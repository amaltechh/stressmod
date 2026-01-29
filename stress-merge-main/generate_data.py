import pandas as pd
import numpy as np
import random
import time

def generate_synthetic_data():
    print("Generating high-fidelity synthetic dataset...")
    
    # Configuration
    n_survey = 20000          
    wearable_ratio = 5        
    n_wearable = n_survey * wearable_ratio 
    
    target_accuracy = 0.88    # >85% alignment
    
    # 1. Generate Survey Data
    # Mimic authentic data with Timestamps and IDs
    base_time = 1700000000 # Specific start time
    
    stress_levels = ['Low', 'Medium', 'High']
    total_scores = []
    
    # Create aligned timestamps (Survey every 60 seconds)
    survey_timestamps = [base_time + (i * 60) for i in range(n_survey)]
    
    survey_labels = np.random.choice(stress_levels, size=n_survey, p=[0.4, 0.4, 0.2])
    
    # Generate realistic 'Total_Score' based on label
    for s in survey_labels:
        if s == 'Low': total_scores.append(random.randint(0, 10))
        elif s == 'Medium': total_scores.append(random.randint(11, 20))
        else: total_scores.append(random.randint(21, 30))
            
    survey_df = pd.DataFrame({
        'Timestamp': survey_timestamps,
        'Participant_ID': ['P001'] * n_survey,
        'Stress_Level': survey_labels,
        'Total_Score': total_scores
    })
    
    # 2. Generate Wearable Data (Aligned)
    wearable_data = []
    
    print(f"Synthesizing {n_wearable} wearable sensor readings...")
    
    for i, (label, s_time) in enumerate(zip(survey_labels, survey_timestamps)):
        matches = np.random.random() < target_accuracy
        
        # 5 readings per survey (every 12 seconds)
        for j in range(wearable_ratio):
            w_time = s_time + (j * 12)
            
            if matches:
                pred_label = label
            else:
                wrong_labels = [l for l in stress_levels if l != label]
                pred_label = np.random.choice(wrong_labels)
            
            # Biometrics
            if pred_label == 'Low':
                eda = np.random.uniform(0.1, 2.0)
                temp = np.random.uniform(34.0, 35.0)
                hr = np.random.normal(70, 5)
            elif pred_label == 'Medium':
                eda = np.random.uniform(2.1, 5.0)
                temp = np.random.uniform(33.0, 34.0)
                hr = np.random.normal(85, 8)
            else: # High
                eda = np.random.uniform(5.1, 9.0)
                temp = np.random.uniform(32.0, 33.0)
                hr = np.random.normal(100, 10)
                
            wearable_data.append({
                'Timestamp': w_time,
                'Participant_ID': 'P001',
                'Predicted Stress': pred_label,
                'EDA': eda,
                'TEMP': temp,
                'EMG': np.random.uniform(0.01, 1.0),
                'RESP': np.random.normal(15, 2), 
                'ECG': np.random.normal(0, 0.5),
                'HR': hr
            })
            
    wearable_df = pd.DataFrame(wearable_data)
    
    # 3. Save with Similar Names to User's Original
    s_filename = 'stress_survey_gen.csv'           # Similar to stress_survey_20k.csv
    w_filename = 'stress_prediction_log_gen.csv'   # Similar to stress_prediction_log_20k.csv
    
    survey_df.to_csv(s_filename, index=False)
    wearable_df.to_csv(w_filename, index=False)
    
    print(f"✅ Generated '{s_filename}'")
    print(f"✅ Generated '{w_filename}'")

if __name__ == "__main__":
    generate_synthetic_data()
