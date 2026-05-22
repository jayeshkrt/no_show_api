# Medical Appointment No-Show Prediction Model

## Use Case
Predict medical appointment no-shows to enable proactive reminders and optimize healthcare resource allocation.

**Output:** No-Show Probability with Binary Classification  
**Application:** Proactive appointment reminders and scheduling optimization

---

## Project Overview

This project builds a **Binary Classification Random Forest Model** to predict whether patients will miss their medical appointments. The model is trained on historical appointment data and uses patient demographics, appointment details, and environmental factors to identify high-risk no-show cases.

---

## Model Details

**Algorithm:** Random Forest Classifier  
**Hyperparameters:**
- n_estimators: 300
- max_depth: 12
- min_samples_leaf: 10
- class_weight: balanced_subsample (handles class imbalance)
- random_state: 42

**Optimization:** Threshold tuning for F1-score (optimal threshold: 0.51)

---

## Features Used

### Patient Demographics
- Gender
- Age, Age Groups (under_12_years_old, over_60_years_old)
- Disability Type
- Patient Needs Companion

### Appointment Details
- Specialty (appointment type)
- Appointment Time
- Appointment Date (ordinal format)
- Appointment Shift (morning, afternoon, etc.)
- Appointment Month & Year
- ICD Code (diagnosis)

### Environmental/Contextual Factors
- City
- Date of Birth (ordinal format)
- Entry Service Date (ordinal format)
- Average Temperature (day)
- Average Rain (day)
- Max Temperature (day)
- Max Rain (day)
- Rainy Day Before (boolean)
- Storm Day Before (boolean)
- Rain Intensity
- Heat Intensity

---

## Data Processing Pipeline

1. **Data Loading:** Read appointment records from CSV
2. **Data Cleaning:** Remove leakage column (no_show_reason)
3. **Date Conversion:** Convert date columns to ordinal format (numeric)
4. **Categorical Encoding:** LabelEncoder for 11 categorical features
5. **Boolean Conversion:** Convert yes/no columns to 0/1
6. **Imputation:** SimpleImputer with median strategy for missing values
7. **Train-Test Split:** 80% train / 20% test (random_state=42)
8. **Threshold Optimization:** Find best threshold (0.20-0.79) based on F1-score

---

## Model Performance

```
Best Threshold: 0.51
Best F1-Score: 0.3385

Classification Report:
              precision    recall  f1-score   support
           0       0.94      0.85      0.89      8982
           1       0.26      0.49      0.34       937
    
    accuracy                           0.82      9919
   macro avg       0.60      0.67      0.62      9919
weighted avg       0.88      0.82      0.84      9919
```

### Interpretation
- **Overall Accuracy:** 82% - solid baseline
- **Recall (No-Shows):** 49% - catches ~half of actual no-shows for proactive outreach
- **Precision (No-Shows):** 26% - 26% of alerts are correct (74% false positives)
- **Specificity (Show-ups):** 85% - reliably identifies patients who will attend

### Class Imbalance
- Total Records: 9,919
- Show-ups: 8,982 (90.6%)
- No-Shows: 937 (9.4%)

---

## Installation & Setup

Install required libraries:

```bash
C:\Users\XYZ\AppData\Local\Microsoft\WindowsApps\python3.11.exe -m pip install -r requirements.txt
```

Or use your Python installation:

```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- requests
- joblib

---

## Running the Model

Execute the training and evaluation script:

```bash
python model.py
```

This will:
1. Load and preprocess the appointment data
2. Train the Random Forest model
3. Optimize threshold for best F1-score
4. Generate classification report
5. Save the trained model as `no_show_model.pkl`

---

## Output Files

- **no_show_model.pkl:** Serialized trained model for future predictions
- **Console Output:** Best threshold, F1-score, and classification metrics

---

## Files in This Project

- **model.py** - Main training script with data preprocessing and model evaluation
- **medical-appointments-no-show-en.csv** - Historical appointment dataset (~10k records)
- **requirements.txt** - Python package dependencies
- **readme.md** - This file

---

## Recommendations for Improvement

1. **Increase Recall** - Lower decision threshold to catch more no-shows
2. **Reduce False Positives** - Feature engineering to improve precision
3. **Feature Enhancement** - Add appointment history, previous cancellations
4. **Class Imbalance** - Consider SMOTE, undersampling, or cost-sensitive learning
5. **Alternative Models** - Evaluate XGBoost, LightGBM for better performance
6. **Hyperparameter Tuning** - GridSearchCV for optimal parameter selection
7. **Cross-Validation** - Implement k-fold cross-validation for robustness

---

## Model Deployment Considerations

✓ **Suitable for:** Proactive reminder systems with acceptable false positive rate  
✓ **Next Steps:** Integrate into scheduling system, monitor in production  
⚠ **Alert Fatigue Risk:** Low precision may require user feedback mechanisms  
⚠ **Continuous Monitoring:** Retrain model periodically with new data