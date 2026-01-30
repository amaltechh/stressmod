# 🧬 Stress Fusion AI (v14.0 Cyberpunk Edition)

**Advanced Biometric Synchronization & Analysis System**  
*4th Year Main Project*

| Team Member | Role |
| :--- | :--- |
| **Amal Benny Joseph** | Team Lead, Model Developer & Full Stack Architect |
| **Adarsh K Sundaresan** | SFAA Implementation, Core Logic |
| **Alstin Gloria Chacko** | Data Forensic Audit, Validation |
| **Akhil S Nambiar** | Visualization & Documentation |

---

## 🌟 Overview
**Stress Fusion AI** is a next-generation dashboard that bridges the gap between subjective stress (surveys) and objective physiology (wearables).

**New in v14.0 (Cyberpunk Edition):**
*   **Neon Glassmorphism UI**: A futuristic dark mode with 'Rajdhani' typography.
*   **Interactive Simulation**: Visualizes the training process with real-time epoch tracking.
*   **Forensic Auditing**: Automated detection of sensor failures (flatlines, disconnects).

---

## 🧩 The 3-Module Architecture

The application is split into three powerful independent engines:

### 1. 📂 Data Content Inspector (v13.8)
*Before you analyze, you must audit.*
*   **Health Scorecard**: Grades your dataset quality (0-100) based on missing values and sync ratios.
*   **Statistical Deep Dive**: Automatically calculates **Skewness** and **Kurtosis** to check for Gaussian distribution.
*   **Signal Quality Check**: Detects "dead sensors" (flatlines) and loose contacts (zero values).
*   **Dataset DNA**: A raw correlation matrix to identify redundant features.

### 2. ⚡ Analysis Engine (v13.5)
*The heart of the AI.*
*   **Class Radar Envelope**: A polar chart allowing you to see the "Shape" of the model's accuracy (Precision vs. Recall).
*   **Temporal Stability**: Tracks the model's confidence over time to detect "Fatigue" or drift.
*   **Sankey Error Flow**: Visually traces where Ground Truth labels are being misclassified.
*   **Biometric Separation**: Overlays signal distributions (e.g., EDA) for Low/Med/High stress to prove separability.

### 3. 🧠 AI Masterclass
*The textbook built into the app.*
*   **Neurobiology**: Learn about the HPA Axis, Vagal Tone, and Vasoconstriction.
*   **Machine Learning**: Deep dives into Random Forest "Bagging", "Gini Impurity", and "OOB Error".
*   **Signal Processing**: visual guides on Sliding Windows and Nyquist Sampling.

---

## 🛠️ Installation & Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn plotly streamlit
```

### 2. Run the Analysis Dashboard (Batch Mode)
```bash
python -m streamlit run merge.py
```

### 3. Run the "Guru Module" (Live User Mode)
**New in Phase 4:** Calculate your personal stress score and get accurate **Yoga & Ayurveda** remedies.
```bash
python -m streamlit run live_app.py
```

---

## 🧪 Technical Stack
*   **Frontend**: Streamlit + Custom CSS (Glassmorphism/Neon)
*   **Visualization**: Plotly Express / Graph Objects
*   **Logic**: Scikit-Learn (Random Forest Encapsulation)
*   **Pandas**: High-performance time-series alignment (SFAA Core)

> *"Stress is not an emotion. It is a measurable algorithm."*
