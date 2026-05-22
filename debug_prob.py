import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

MODEL = joblib.load('no_show_model.pkl')
df = pd.read_csv('medical-appointments-no-show-en.csv')
CAT_COLS = ['specialty','appointment_time','gender','city','icd','appointment_shift','rain_intensity','heat_intensity','disability','appointment_month']
encoders = {
    col: {v: i for i, v in enumerate(df[col].fillna('missing').astype(str).str.lower().unique())}
    for col in CAT_COLS
}

def normalize_category(col, value):
    if value is None:
        return encoders[col].get('missing', 0)
    return encoders[col].get(str(value).strip().lower(), encoders[col].get('missing', 0))


def to_ordinal(value):
    dt = pd.to_datetime(value, errors='coerce')
    return float(dt.toordinal()) if not pd.isna(dt) else np.nan


def bool_to_int(value):
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        text = str(value).strip().lower()
        return 1 if text in {'yes', 'true', '1', 'y', 'present'} else 0

TRAIN_FEATURES = []
for _, r in df.iterrows():
    TRAIN_FEATURES.append([
        normalize_category('specialty', r['specialty']),
        normalize_category('appointment_time', r['appointment_time']),
        normalize_category('gender', r['gender']),
        to_ordinal(r['appointment_date']),
        normalize_category('city', r['city']),
        normalize_category('icd', r['icd']),
        normalize_category('appointment_shift', r['appointment_shift']),
        normalize_category('rain_intensity', r['rain_intensity']),
        normalize_category('heat_intensity', r['heat_intensity']),
        normalize_category('disability', r['disability']),
        normalize_category('appointment_month', r['appointment_month']),
        r['appointment_year'] if pd.notnull(r['appointment_year']) else np.nan,
        bool_to_int(r['under_12_years_old']),
        bool_to_int(r['over_60_years_old']),
        bool_to_int(r['patient_needs_companion']),
        to_ordinal(r['date_of_birth']),
        to_ordinal(r['entry_service_date']),
        r['age'] if pd.notnull(r['age']) else np.nan,
        r['average_temp_day'] if pd.notnull(r['average_temp_day']) else np.nan,
        r['average_rain_day'] if pd.notnull(r['average_rain_day']) else np.nan,
        r['max_temp_day'] if pd.notnull(r['max_temp_day']) else np.nan,
        r['max_rain_day'] if pd.notnull(r['max_rain_day']) else np.nan,
        r['rainy_day_before'] if pd.notnull(r['rainy_day_before']) else np.nan,
        r['storm_day_before'] if pd.notnull(r['storm_day_before']) else np.nan,
    ])

IMPUTER = SimpleImputer(strategy='median')
IMPUTER.fit(np.array(TRAIN_FEATURES, dtype=float))

base = {
    'specialty': 'physiotherapy',
    'appointment_time': '13:20',
    'gender': 'M',
    'appointment_date': '2024-07-30',
    'city': 'B. CAMBORIU',
    'icd': 'I67',
    'appointment_shift': 'afternoon',
    'appointment_month': 'july',
    'appointment_year': 2024,
    'under_12_years_old': 0,
    'over_60_years_old': 1,
    'patient_needs_companion': 0,
    'date_of_birth': '1956-01-01',
    'entry_service_date': '2024-01-01',
    'age': 68,
    'average_temp_day': 20.0,
    'average_rain_day': 0.0,
    'max_temp_day': 25.0,
    'max_rain_day': 0.0,
    'rainy_day_before': 0,
    'storm_day_before': 0,
    'rain_intensity': 'no_rain',
    'heat_intensity': 'mild',
    'disability': None,
}


def make_x(row):
    return np.array([
        normalize_category('specialty', row['specialty']),
        normalize_category('appointment_time', row['appointment_time']),
        normalize_category('gender', row['gender']),
        to_ordinal(row['appointment_date']),
        normalize_category('city', row['city']),
        normalize_category('icd', row['icd']),
        normalize_category('appointment_shift', row['appointment_shift']),
        normalize_category('rain_intensity', row['rain_intensity']),
        normalize_category('heat_intensity', row['heat_intensity']),
        normalize_category('disability', row.get('disability')),
        normalize_category('appointment_month', row['appointment_month']),
        row['appointment_year'],
        bool_to_int(row['under_12_years_old']),
        bool_to_int(row['over_60_years_old']),
        bool_to_int(row['patient_needs_companion']),
        to_ordinal(row['date_of_birth']),
        to_ordinal(row['entry_service_date']),
        row['age'],
        row['average_temp_day'],
        row['average_rain_day'],
        row['max_temp_day'],
        row['max_rain_day'],
        row['rainy_day_before'],
        row['storm_day_before'],
    ], dtype=float).reshape(1, -1)

for col, vals in [
    ('city', ['B. CAMBORIU', 'Unknown']),
    ('gender', ['M', 'F']),
    ('appointment_time', ['13:20', '08:00', '09:20']),
    ('specialty', ['physiotherapy', 'psychotherapy']),
]:
    for v in vals:
        row = base.copy()
        row[col] = v
        x = make_x(row)
        xp = IMPUTER.transform(x)
        print(col, v, MODEL.predict_proba(xp)[0, 1])
