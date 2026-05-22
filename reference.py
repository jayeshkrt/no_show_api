import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

df = pd.read_csv("your_file.csv")
df = df.drop(columns=["no_show_reason"])

y = df["no_show"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["no_show"])

date_cols = ["appointment_date", "date_of_birth", "entry_service_date"]
for col in date_cols:
    X[col] = pd.to_datetime(X[col], errors="coerce")
    X[col] = X[col].map(lambda x: x.toordinal() if pd.notnull(x) else None)

cat_cols = ["specialty","appointment_time","gender","city","icd","appointment_shift","rain_intensity","heat_intensity"]
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

bool_cols = ["under_12_years_old","over_60_years_old","patient_needs_companion"]
for col in bool_cols:
    X[col] = X[col].astype(int)

X = SimpleImputer(strategy="median").fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight=class_weights,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))