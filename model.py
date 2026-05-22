import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\Owner\Documents\jaybhaiyamtech\medical-appointments-no-show-en.csv")

# Drop leakage column
df = df.drop(columns=["no_show_reason"])

# Target
y = df["no_show"].map({"yes": 1, "no": 0})
X = df.drop(columns=["no_show"])

# Convert date columns
date_cols = ["appointment_date", "date_of_birth", "entry_service_date"]
for col in date_cols:
    X[col] = pd.to_datetime(X[col], errors="coerce")
    X[col] = X[col].map(lambda x: x.toordinal() if pd.notnull(x) else None)

# Encode categorical columns
cat_cols = [
    "specialty", "appointment_time", "gender", "city",
    "icd", "appointment_shift", "rain_intensity", "heat_intensity",
    "disability", "appointment_month", "appointment_year"
]

le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Convert boolean-like columns
bool_cols = ["under_12_years_old", "over_60_years_old", "patient_needs_companion"]
for col in bool_cols:
    X[col] = X[col].astype(int)

# Impute missing values
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# probability prediction
from sklearn.metrics import f1_score

y_prob = model.predict_proba(X_test)[:, 1]

best_t = 0
best_f1 = 0

for t in [i/100 for i in range(20, 80)]:
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best threshold:", best_t)
print("Best F1:", best_f1)
y_pred = (y_prob >= best_t).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "no_show_model.pkl")