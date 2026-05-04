# Neuro-Fusion Stress Analyzer (SFAA v18.0)
## Presentation Outline / Agenda

---

### Slide 1 — Title Slide
- **Project Title**: Neuro-Fusion Stress Analyzer – A Clinical-Grade Biometric & Psychometric Stress Detection System
- **Subtitle**: Integrating Machine Learning with Indian Knowledge Systems (IKS)
- **Team / Author**: Amal Benny Joseph
- **Institution / Date**

---

### Slide 2 — Problem Statement
- Stress is the leading cause of mental health deterioration among students and professionals
- Traditional stress assessment relies solely on self-reported surveys → **subjective bias**
- No affordable tool combines wearable biosensor data with psychological evaluation
- Gap: Lack of systems that fuse objective biometrics with subjective perception

---

### Slide 3 — Objective
1. Build an AI-powered stress detection system that merges **biological signals** (EDA, Heart Rate, Temperature) with **psychological surveys** (PSS-based)
2. Apply a **weighted Neuro-Fusion algorithm** (60% Body + 40% Mind) to remove subjective bias
3. Map stress diagnoses to **Ayurvedic (IKS) therapeutic interventions** (Pranayama, Asana, Diet, Mantra)
4. Provide a transparent, explainable clinical dashboard

---

### Slide 4 — Literature Survey / Background
- **Perceived Stress Scale (PSS)** — Cohen et al. (1983)
- **Polyvagal Theory** — Stephen Porges (autonomic nervous system & stress)
- **Allostatic Load Model** — Bruce McEwen (cumulative stress burden)
- **Tridosha Framework** — Ayurveda (Vata, Pitta, Kapha imbalance mapping)
- **EDA as a Stress Biomarker** — Boucsein (2012), skin conductance response

---

### Slide 5 — Proposed System Architecture
- *[Insert Block Diagram here]*
- 5-Layer Architecture:
  1. **Data Acquisition Layer** — Wearable sensors + PSS Survey + Demo Presets
  2. **Data Layer** — Synthetic data generation (20K survey + 100K sensor records)
  3. **Model Training Pipeline** — 6 competing ML algorithms
  4. **Neuro-Fusion Engine** — Weighted score fusion
  5. **Presentation Layer** — Clinical App + Batch Dashboard

---

### Slide 6 — Technology Stack
| Component        | Technology                              |
|:-----------------|:----------------------------------------|
| Frontend / UI    | Streamlit (Python)                      |
| Styling          | Custom CSS (Glassmorphism)              |
| ML Engine        | Scikit-learn (GradientBoostingClassifier)|
| Benchmarks       | Logistic Reg, SVM, RF, XGBoost, MLP    |
| Visualization    | Plotly (interactive) + Matplotlib (reports) |
| Data Handling    | Pandas, NumPy                           |
| Model Storage    | Joblib (.pkl serialization)             |

---

### Slide 7 — Data Collection & Generation
- **Survey Data**: 20,000 synthetic PSS records across 5 stress categories (Academic, Emotional, Social, Physical, Coping)
- **Sensor Data**: 100,000 simulated wearable readings (EDA µS, Heart Rate BPM, Skin Temperature °C)
- Stress-correlated noise injection for realistic distributions
- Script: `generate_data.py`

---

### Slide 8 — Machine Learning Pipeline
- Feature Engineering → Label Encoding → 80/20 Stratified Split
- **6 Models Trained & Compared**:
  1. Logistic Regression
  2. Support Vector Machine (RBF Kernel)
  3. Random Forest (100 trees)
  4. XGBoost
  5. **Gradient Boosting Machine (GBM) ⭐ — Deployed**
  6. MLP Neural Network (64→32→16)
- Evaluation: Accuracy, F1-Score, Precision, Recall
- Best model selected by F1-Score and exported as `.pkl`

---

### Slide 9 — Neuro-Fusion Algorithm
- **Mind Score (40%)**: Normalized PSS survey average across 5 categories (0–1)
- **Body Score (60%)**: GBM `predict_proba` → weighted sum: `P(High)×0.9 + P(Med)×0.5 + P(Low)×0.2`
- **Final Score** = `0.6 × Body + 0.4 × Mind`
- Classification:
  - < 0.4 → 🟢 **Low Stress** (Kapha / Eustress)
  - 0.4 – 0.7 → 🟡 **Medium Stress** (Vata / Allostatic)
  - > 0.7 → 🔴 **High Stress** (Pitta / Burnout)

---

### Slide 10 — IKS Integration (Ayurvedic Remedies)
| Stress Level | Dosha     | Pranayama       | Asana           | Diet                | Mantra        |
|:-------------|:----------|:----------------|:----------------|:--------------------|:--------------|
| Low          | Kapha     | Kapalbhati      | Surya Namaskar  | Ginger + Turmeric   | Gayatri       |
| Medium       | Vata      | Nadi Shodhana   | Vrikshasana     | Chamomile / Tulsi   | So Hum        |
| High         | Pitta     | Bhramari        | Shavasana       | Coconut Water       | Om Shanti     |

---

### Slide 11 — Live Application Demo
- **Clinical App** (`live_app.py` — Port 8530)
  - 6-tab data acquisition (Academic, Emotional, Social, Physical, Coping, Biometrics)
  - One-click clinical report generation
  - PDF/PNG downloadable reports
  - Dr. Zen AI Chatbot (sidebar)
  - Session history & telemetry tracking
- **Batch Analysis Dashboard** (`merge.py` — Port 8501)
  - Data Inspector (dictionary, stats, quality, DNA)
  - Analysis Engine (confusion matrix, temporal stability, radar, Sankey, feature impact)
  - AI Masterclass (neurobiology, ML, signal processing educational content)

---

### Slide 12 — Results & Analysis
- Model comparison table (Accuracy, F1, Precision, Recall, Training Time)
- GBM selected as production model
- Confusion Matrix visualization
- Feature importance ranking
- Somatic Mismatch detection (when Mind and Body scores disagree)

---

### Slide 13 — Explainability & Transparency
- XAI Dashboard built into the application
- Clinical frameworks explained to end-user:
  - Polyvagal Theory mapping
  - Allostatic Load indicators
  - Tridosha balance visualization
- Full pipeline transparency (data → model → score → diagnosis)

---

### Slide 14 — Limitations
1. Currently uses **synthetic data** — real-world validation pending
2. No live wearable API integration yet (simulated sensor inputs)
3. Single-session analysis only — no longitudinal trend tracking
4. No multi-user authentication or cloud database
5. IKS remedies are informational, not clinically prescribed

---

### Slide 15 — Future Scope
1. **Real Wearable Integration** — Fitbit / Apple Health / Google Fit APIs
2. **LSTM / Time-Series Models** — Longitudinal stress trend analysis
3. **Multi-User Support** — Authentication + SQLite/Cloud database
4. **Clinical Validation** — IRB-approved study comparing against cortisol biomarkers
5. **Mobile App** — Port from Streamlit to Flutter for smartphone deployment
6. **Federated Learning** — Privacy-preserving model training across institutions

---

### Slide 16 — Conclusion
- Successfully built a **dual-input stress analyzer** combining biometrics and psychology
- The **60/40 Neuro-Fusion formula** corrects for subjective bias in self-reporting
- Integrated **Indian Knowledge Systems (IKS)** for holistic, culturally-rooted therapy
- Delivered a fully functional, transparent, and explainable clinical dashboard

---

### Slide 17 — References
- Cohen, S., Kamarck, T., & Mermelstein, R. (1983). *A Global Measure of Perceived Stress.* JHSB.
- Porges, S.W. (2011). *The Polyvagal Theory.* Norton.
- McEwen, B.S. (1998). *Stress, Adaptation, and Disease: Allostasis and Allostatic Load.* Annals of the NYAS.
- Boucsein, W. (2012). *Electrodermal Activity.* Springer.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.

---

### Slide 18 — Q&A / Demo
- Live demonstration of the application
- Open floor for questions

---
