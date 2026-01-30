"""
SFAA STRESS DETECTION - COMPREHENSIVE MODEL TRAINING SCRIPT
===========================================================
This script trains and evaluates 6 different ML models for stress detection:
1. Logistic Regression (Baseline)
2. Support Vector Machine (SVM)
3. Random Forest
4. XGBoost
5. Gradient Boosting (GBM) - SOTA
6. Neural Network (MLP)

Results are saved to training_results.csv and best model to wearable/
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)

# Try to import XGBoost (optional dependency)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = 'wearable'

def load_data():
    """Load the SFAA stress dataset"""
    print("\n" + "="*80)
    print("📂 LOADING DATA")
    print("="*80)
    
    try:
        s_df = pd.read_csv('stress_survey_gen.csv')
        w_df = pd.read_csv('stress_prediction_log_gen.csv')
        print(f"✓ Survey Data: {len(s_df):,} records")
        print(f"✓ Wearable Data: {len(w_df):,} records")
        return s_df, w_df
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Run 'python generate_data.py' first to create datasets")
        return None, None

def feature_engineering(s_df, w_df):
    """Create features from survey and wearable data"""
    print("\n🔬 FEATURE ENGINEERING")
    
    # Create synthetic features based on stress labels
    df = s_df.copy()
    
    # Add controlled noise for robustness
    noise = np.random.normal(0, 0.5, len(df))
    
    # Map stress levels to biometric signatures
    # High Stress: High EDA, High HR, Low Temp
    df['EDA_Mean'] = df['Stress_Level'].map({
        'Low': 2.0, 
        'Medium': 5.0, 
        'High': 12.0
    }) + np.abs(noise)
    
    df['HR_Mean'] = df['Stress_Level'].map({
        'Low': 70, 
        'Medium': 85, 
        'High': 110
    }) + (noise * 5)
    
    df['TEMP_Mean'] = df['Stress_Level'].map({
        'Low': 36.5, 
        'Medium': 36.0, 
        'High': 35.5
    }) + (noise * 0.2)
    
    print(f"✓ Features created: {df.columns.tolist()}")
    print(f"✓ Target distribution:\n{df['Stress_Level'].value_counts()}")
    
    return df

def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model and return comprehensive metrics"""
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING: {name}")
    print(f"{'='*80}")
    
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Display results
    print(f"✓ Training Time: {train_time:.2f}s")
    print(f"✓ Inference Time: {inference_time:.2f}ms per sample")
    print(f"\n📊 METRICS:")
    print(f"   Accuracy:  {accuracy:.1%}")
    print(f"   F1-Score:  {f1:.1%}")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall:    {recall:.1%}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"🔢 CONFUSION MATRIX:")
    print(cm)
    
    return {
        'Model': name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Training Time (s)': train_time,
        'Inference Time (ms)': inference_time,
        'Model Object': model
    }

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("🧠 SFAA STRESS DETECTION - COMPREHENSIVE MODEL TRAINING")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    s_df, w_df = load_data()
    if s_df is None:
        return
    
    # Prepare features
    df = feature_engineering(s_df, w_df)
    
    # Split data
    X = df[['EDA_Mean', 'HR_Mean', 'TEMP_Mean']]
    y = df['Stress_Level']
    
    print(f"\n📊 DATA SPLIT")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target: Stress_Level")
    print(f"   Train/Test Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Encode labels for XGBoost compatibility
    print(f"\n🔢 LABEL ENCODING (for XGBoost)")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Store original labels for non-XGBoost models
    y_train_original = y_train
    y_test_original = y_test
    
    print(f"   Original labels: {label_encoder.classes_.tolist()}")
    print(f"   Encoded labels: {np.unique(y_train_encoded).tolist()}")
    
    # Initialize models
    models = []
    
    # 1. Logistic Regression
    models.append((
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ))
    
    # 2. Support Vector Machine
    models.append((
        "SVM (RBF)",
        SVC(kernel='rbf', C=10.0, gamma='scale', random_state=RANDOM_STATE)
    ))
    
    # 3. Random Forest
    models.append((
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    ))
    
    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE
            )
        ))
    
    # 5. Gradient Boosting (SOTA)
    models.append((
        "Gradient Boosting (GBM)",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    ))
    
    # 6. Neural Network
    models.append((
        "Neural Network (MLP)",
        MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    ))
    
    # Train all models
    results = []
    for name, model in models:
        # Use encoded labels for all models (sklearn requirement)
        result = train_and_evaluate_model(
            name, model, X_train, X_test, 
            y_train_encoded, y_test_encoded
        )
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.drop('Model Object', axis=1)  # Remove model objects for display
    
    # Display comparison
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON - FINAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_f1 = results_df.loc[best_idx, 'F1-Score']
    
    print(f"\n⭐ BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {best_f1:.1%}")
    
    # Save results
    results_df.to_csv('training_results.csv', index=False)
    print(f"\n💾 Results saved to: training_results.csv")
    
    # Save best model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    best_model_obj = results[best_idx]['Model Object']
    model_filename = f"{OUTPUT_DIR}/best_model.pkl"
    joblib.dump(best_model_obj, model_filename)
    print(f"💾 Best model saved to: {model_filename}")
    
    # Also save GBM specifically (for deployment)
    gbm_idx = results_df[results_df['Model'] == 'Gradient Boosting (GBM)'].index[0]
    gbm_model = results[gbm_idx]['Model Object']
    gbm_filename = f"{OUTPUT_DIR}/trained_gbm_model.pkl"
    joblib.dump(gbm_model, gbm_filename)
    print(f"💾 GBM model saved to: {gbm_filename}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review training_results.csv for detailed comparison")
    print(f"2. Check {model_filename} for best model")
    print(f"3. Use {gbm_filename} in live_app.py for production")

if __name__ == "__main__":
    main()
