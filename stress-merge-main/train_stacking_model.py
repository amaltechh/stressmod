"""
NEURO-FUSION STACKING ENSEMBLE - TRAINING PIPELINE
====================================================
3-Model Architecture:
  Level 1a: XGBoost       → Survey data (5 PSS category scores)
  Level 1b: Random Forest → Wearable data (EDA, HR, TEMP)
  Level 2:  Gradient Boosting (Meta-Learner) → Fuses both predictions

Uses K-Fold Cross-Validation to generate out-of-fold predictions
for the meta-learner to prevent data leakage.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import logging
from datetime import datetime

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")
    print("   Falling back to GradientBoostingClassifier for survey model.")

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
OUTPUT_DIR_SURVEY = 'survey'
OUTPUT_DIR_WEARABLE = 'wearable'

# Setup logging
logging.basicConfig(
    filename='training_stacking.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log(msg):
    """Print and log a message."""
    print(msg)
    logging.info(msg)


def load_data():
    """Load survey and wearable datasets."""
    log("\n" + "=" * 80)
    log("📂 LOADING DATA")
    log("=" * 80)

    try:
        s_df = pd.read_csv('stress_survey_gen.csv')
        w_df = pd.read_csv('stress_prediction_log_gen.csv')
        log(f"✓ Survey Data: {len(s_df):,} records, Columns: {s_df.columns.tolist()}")
        log(f"✓ Wearable Data: {len(w_df):,} records, Columns: {w_df.columns.tolist()}")
        return s_df, w_df
    except FileNotFoundError as e:
        log(f"❌ Error: {e}")
        log("💡 Run 'python generate_data.py' first to create datasets")
        return None, None


def prepare_aligned_data(s_df, w_df):
    """
    Align survey and wearable data.
    Aggregate wearable readings per survey record (5:1 ratio).
    Returns a single DataFrame with both survey features and wearable features.
    """
    log("\n🔬 ALIGNING SURVEY + WEARABLE DATA")

    n_survey = len(s_df)
    wearable_ratio = len(w_df) // n_survey

    log(f"   Survey records: {n_survey:,}")
    log(f"   Wearable records: {len(w_df):,}")
    log(f"   Wearable-to-Survey ratio: {wearable_ratio}:1")

    # Aggregate wearable data: take mean of each chunk of `wearable_ratio` readings
    w_trimmed = w_df.iloc[:n_survey * wearable_ratio].copy()
    w_trimmed['group'] = np.arange(len(w_trimmed)) // wearable_ratio

    w_agg = w_trimmed.groupby('group').agg({
        'EDA': 'mean',
        'HR': 'mean',
        'TEMP': 'mean'
    }).reset_index(drop=True)

    # Build aligned DataFrame
    aligned = pd.DataFrame({
        # Survey features (for XGBoost)
        'Academic_Score': s_df['Academic_Score'].values,
        'Emotional_Score': s_df['Emotional_Score'].values,
        'Social_Score': s_df['Social_Score'].values,
        'Physical_Score': s_df['Physical_Score'].values,
        'Coping_Score': s_df['Coping_Score'].values,
        # Wearable features (for Random Forest)
        'EDA': w_agg['EDA'].values,
        'HR': w_agg['HR'].values,
        'TEMP': w_agg['TEMP'].values,
        # Target
        'Stress_Level': s_df['Stress_Level'].values
    })

    log(f"✓ Aligned dataset: {len(aligned):,} records with {len(aligned.columns)} columns")
    log(f"   Survey features: Academic, Emotional, Social, Physical, Coping")
    log(f"   Wearable features: EDA, HR, TEMP")
    log(f"   Target distribution:\n{aligned['Stress_Level'].value_counts().to_string()}")

    return aligned


def train_stacking_ensemble(aligned_df):
    """
    Train the 3-model stacking ensemble:
    1. XGBoost on survey features
    2. Random Forest on wearable features
    3. GBM meta-learner on combined predictions
    """
    log("\n" + "=" * 80)
    log("🚀 TRAINING STACKING ENSEMBLE")
    log("=" * 80)

    # --- PREPARE FEATURES & LABELS ---
    survey_features = ['Academic_Score', 'Emotional_Score', 'Social_Score',
                       'Physical_Score', 'Coping_Score']
    wearable_features = ['EDA', 'HR', 'TEMP']

    X_survey = aligned_df[survey_features].values
    X_wearable = aligned_df[wearable_features].values
    y = aligned_df['Stress_Level'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    log(f"\n🔢 Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- TRAIN/TEST SPLIT ---
    (X_survey_train, X_survey_test,
     X_wearable_train, X_wearable_test,
     y_train, y_test) = train_test_split(
        X_survey, X_wearable, y_encoded,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    log(f"\n📊 DATA SPLIT")
    log(f"   Training samples: {len(y_train):,}")
    log(f"   Testing samples: {len(y_test):,}")

    # --- INITIALIZE LEVEL-1 MODELS ---
    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        survey_model_name = "XGBoost"
    else:
        # Fallback to GBM if XGBoost not installed
        xgb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE
        )
        survey_model_name = "GradientBoosting (XGBoost fallback)"

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # ========================================================
    # PHASE 1: Generate Out-of-Fold (OOF) Predictions
    # This prevents data leakage in the meta-learner
    # ========================================================
    log(f"\n{'='*80}")
    log(f"📦 PHASE 1: Out-of-Fold Prediction Generation ({N_FOLDS}-Fold CV)")
    log(f"{'='*80}")

    n_classes = len(le.classes_)
    oof_xgb = np.zeros((len(y_train), n_classes))
    oof_rf = np.zeros((len(y_train), n_classes))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_survey_train, y_train)):
        log(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        # Fold split
        X_s_fold_train, X_s_fold_val = X_survey_train[train_idx], X_survey_train[val_idx]
        X_w_fold_train, X_w_fold_val = X_wearable_train[train_idx], X_wearable_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Train XGBoost on survey features for this fold
        xgb_fold = XGBClassifier(**xgb_model.get_params()) if XGBOOST_AVAILABLE else \
                   GradientBoostingClassifier(**xgb_model.get_params())
        xgb_fold.fit(X_s_fold_train, y_fold_train)
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_s_fold_val)

        # Train Random Forest on wearable features for this fold
        rf_fold = RandomForestClassifier(**rf_model.get_params())
        rf_fold.fit(X_w_fold_train, y_fold_train)
        oof_rf[val_idx] = rf_fold.predict_proba(X_w_fold_val)

        # Fold accuracy
        xgb_fold_acc = accuracy_score(y_fold_val, xgb_fold.predict(X_s_fold_val))
        rf_fold_acc = accuracy_score(y_fold_val, rf_fold.predict(X_w_fold_val))
        log(f"   {survey_model_name} (Survey) Fold Acc: {xgb_fold_acc:.1%}")
        log(f"   Random Forest (Wearable) Fold Acc: {rf_fold_acc:.1%}")

    # ========================================================
    # PHASE 2: Train Final Level-1 Models on FULL Training Set
    # ========================================================
    log(f"\n{'='*80}")
    log(f"🎯 PHASE 2: Training Final Level-1 Models (Full Training Set)")
    log(f"{'='*80}")

    # --- Level 1a: XGBoost (Survey) ---
    log(f"\n🤖 Training {survey_model_name} on Survey Features...")
    start = time.time()
    xgb_model.fit(X_survey_train, y_train)
    xgb_time = time.time() - start

    xgb_test_pred = xgb_model.predict(X_survey_test)
    xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
    xgb_test_f1 = f1_score(y_test, xgb_test_pred, average='weighted')
    log(f"   Training Time: {xgb_time:.2f}s")
    log(f"   Test Accuracy: {xgb_test_acc:.1%}")
    log(f"   Test F1-Score: {xgb_test_f1:.1%}")
    log(f"\n   Classification Report ({survey_model_name} - Survey):")
    log(classification_report(y_test, xgb_test_pred, target_names=le.classes_))

    # --- Level 1b: Random Forest (Wearable) ---
    log(f"\n🌲 Training Random Forest on Wearable Features...")
    start = time.time()
    rf_model.fit(X_wearable_train, y_train)
    rf_time = time.time() - start

    rf_test_pred = rf_model.predict(X_wearable_test)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    rf_test_f1 = f1_score(y_test, rf_test_pred, average='weighted')
    log(f"   Training Time: {rf_time:.2f}s")
    log(f"   Test Accuracy: {rf_test_acc:.1%}")
    log(f"   Test F1-Score: {rf_test_f1:.1%}")
    log(f"\n   Classification Report (Random Forest - Wearable):")
    log(classification_report(y_test, rf_test_pred, target_names=le.classes_))

    # ========================================================
    # PHASE 3: Train Meta-Learner (GBM) on OOF Predictions
    # ========================================================
    log(f"\n{'='*80}")
    log(f"⚡ PHASE 3: Training GBM Meta-Learner (Fusion Model)")
    log(f"{'='*80}")

    # Build meta-features from OOF predictions
    meta_train = np.hstack([oof_xgb, oof_rf])  # Shape: (n_train, 6)
    log(f"   Meta-features shape: {meta_train.shape}")
    log(f"   Features: [XGB_P_High, XGB_P_Low, XGB_P_Med, RF_P_High, RF_P_Low, RF_P_Med]")

    # Build meta-features for test set using the final Level-1 models
    xgb_test_probs = xgb_model.predict_proba(X_survey_test)
    rf_test_probs = rf_model.predict_proba(X_wearable_test)
    meta_test = np.hstack([xgb_test_probs, rf_test_probs])

    # Train GBM Meta-Learner
    gbm_meta = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    start = time.time()
    gbm_meta.fit(meta_train, y_train)
    meta_time = time.time() - start

    meta_pred = gbm_meta.predict(meta_test)
    meta_acc = accuracy_score(y_test, meta_pred)
    meta_f1 = f1_score(y_test, meta_pred, average='weighted')
    meta_precision = precision_score(y_test, meta_pred, average='weighted')
    meta_recall = recall_score(y_test, meta_pred, average='weighted')

    log(f"\n   Training Time: {meta_time:.2f}s")
    log(f"   Test Accuracy: {meta_acc:.1%}")
    log(f"   Test F1-Score: {meta_f1:.1%}")
    log(f"   Test Precision: {meta_precision:.1%}")
    log(f"   Test Recall: {meta_recall:.1%}")
    log(f"\n   Classification Report (GBM Meta-Learner - Stacked):")
    log(classification_report(y_test, meta_pred, target_names=le.classes_))

    # ========================================================
    # PHASE 4: Compare All Models
    # ========================================================
    log(f"\n{'='*80}")
    log(f"📊 FINAL COMPARISON")
    log(f"{'='*80}")

    results = pd.DataFrame([
        {
            'Model': f'{survey_model_name} (Survey)',
            'Data Source': 'PSS Survey (5 categories)',
            'Accuracy': xgb_test_acc,
            'F1-Score': xgb_test_f1,
            'Training Time (s)': xgb_time
        },
        {
            'Model': 'Random Forest (Wearable)',
            'Data Source': 'Sensors (EDA, HR, TEMP)',
            'Accuracy': rf_test_acc,
            'F1-Score': rf_test_f1,
            'Training Time (s)': rf_time
        },
        {
            'Model': 'GBM Meta-Learner (Stacked)',
            'Data Source': 'Fused Predictions (6 probs)',
            'Accuracy': meta_acc,
            'F1-Score': meta_f1,
            'Training Time (s)': meta_time
        }
    ])

    log(f"\n{results.to_string(index=False)}")

    # Highlight improvement
    best_single = max(xgb_test_f1, rf_test_f1)
    improvement = meta_f1 - best_single
    log(f"\n⭐ Stacking Improvement over best single model: {improvement:+.1%}")

    # ========================================================
    # PHASE 5: Save Models
    # ========================================================
    log(f"\n{'='*80}")
    log(f"💾 SAVING MODELS")
    log(f"{'='*80}")

    os.makedirs(OUTPUT_DIR_SURVEY, exist_ok=True)
    os.makedirs(OUTPUT_DIR_WEARABLE, exist_ok=True)

    # Save Level 1a: XGBoost (Survey)
    xgb_path = f'{OUTPUT_DIR_SURVEY}/trained_xgb_survey.pkl'
    joblib.dump(xgb_model, xgb_path)
    log(f"   ✓ {survey_model_name} (Survey) → {xgb_path}")

    # Save Level 1b: Random Forest (Wearable)
    rf_path = f'{OUTPUT_DIR_WEARABLE}/trained_rf_wearable.pkl'
    joblib.dump(rf_model, rf_path)
    log(f"   ✓ Random Forest (Wearable) → {rf_path}")

    # Save Level 2: GBM Meta-Learner
    meta_path = f'{OUTPUT_DIR_WEARABLE}/trained_gbm_meta.pkl'
    joblib.dump(gbm_meta, meta_path)
    log(f"   ✓ GBM Meta-Learner (Fusion) → {meta_path}")

    # Save Label Encoder (needed for decoding predictions in live app)
    le_path = f'{OUTPUT_DIR_WEARABLE}/label_encoder.pkl'
    joblib.dump(le, le_path)
    log(f"   ✓ Label Encoder → {le_path}")

    # Save results
    results.to_csv('training_stacking_results.csv', index=False)
    log(f"   ✓ Results table → training_stacking_results.csv")

    log(f"\n{'='*80}")
    log(f"✅ STACKING ENSEMBLE TRAINING COMPLETE")
    log(f"{'='*80}")
    log(f"\nArchitecture:")
    log(f"   Level 1a: {survey_model_name} → Survey (5 PSS scores) → 3 class probabilities")
    log(f"   Level 1b: Random Forest → Wearable (EDA, HR, TEMP) → 3 class probabilities")
    log(f"   Level 2:  GBM Meta-Learner → Fused (6 probabilities) → Final prediction")
    log(f"\nTo use in live app: python -m streamlit run live_app.py --server.port 8530")

    return results


def main():
    log(f"\n{'='*80}")
    log(f"🧠 NEURO-FUSION STACKING ENSEMBLE TRAINER")
    log(f"{'='*80}")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    s_df, w_df = load_data()
    if s_df is None:
        return

    # Check that survey data has category scores
    required_cols = ['Academic_Score', 'Emotional_Score', 'Social_Score',
                     'Physical_Score', 'Coping_Score']
    missing = [c for c in required_cols if c not in s_df.columns]
    if missing:
        log(f"❌ Survey data is missing columns: {missing}")
        log(f"💡 Re-run 'python generate_data.py' to regenerate with category scores.")
        return

    # Align data
    aligned = prepare_aligned_data(s_df, w_df)

    # Train stacking ensemble
    results = train_stacking_ensemble(aligned)


if __name__ == "__main__":
    main()
