from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.impute import SimpleImputer

MODEL_PATH = "no_show_model.pkl"
DATA_PATH = "medical-appointments-no-show-en.csv"

app = FastAPI(title="No-show Prediction API")

@app.api_route("/", methods=["GET", "POST"])
def read_root():
    return {
        "service": "No-show Prediction API",
        "message": "Use POST /predict to get a no-show prediction.",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Columns used by the original model pipeline
DATE_COLS = ["appointment_date", "date_of_birth", "entry_service_date"]
CAT_COLS = [
    "specialty",
    "appointment_time",
    "gender",
    "city",
    "icd",
    "appointment_shift",
    "rain_intensity",
    "heat_intensity",
    "disability",
    "appointment_month",
]
BOOL_COLS = ["under_12_years_old", "over_60_years_old", "patient_needs_companion"]
NUM_COLS = [
    "age",
    "average_temp_day",
    "average_rain_day",
    "max_temp_day",
    "max_rain_day",
    "rainy_day_before",
    "storm_day_before",
]

MODEL = joblib.load(MODEL_PATH)
DF = pd.read_csv(DATA_PATH)
# NOTE: we will fit the imputer on a numeric-transformed version of the
# training data (after encoding / date conversion) below. Do not fit on raw
# mixed-type DataFrame because median strategy requires numeric input.

CATEGORY_ENCODERS: Dict[str, Dict[str, int]] = {}
for col in CAT_COLS:
    values = DF[col].fillna("missing").astype(str).str.lower()
    unique_values = list(values.unique())
    CATEGORY_ENCODERS[col] = {v: i for i, v in enumerate(unique_values)}
IMPUTER = None
# The numeric training matrix and imputer will be built after helper
# functions (normalize_category, to_ordinal, bool_to_int) are defined.


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return None


def normalize_category(col: str, value: Any) -> int:
    if value is None:
        value_str = "missing"
    else:
        value_str = str(value).strip().lower()
    encoder = CATEGORY_ENCODERS.get(col, {})
    return encoder.get(value_str, encoder.get("missing", 0))


def appointment_shift_from_time(value: Optional[str]) -> str:
    if not value:
        return "missing"
    try:
        dt = pd.to_datetime(value, format="%H:%M", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return "missing"
        hour = dt.hour
        return "morning" if hour < 12 else "afternoon"
    except Exception:
        return "missing"


MONTH_LOOKUP = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "aug",
    9: "sept",
    10: "oct",
    11: "nov",
    12: "dec",
}


def month_from_date(value: Optional[str]) -> str:
    dt = parse_date(value)
    if dt is None or pd.isna(dt):
        return "missing"
    return MONTH_LOOKUP.get(dt.month, "missing")


def year_from_date(value: Optional[str]) -> Optional[int]:
    dt = parse_date(value)
    return int(dt.year) if dt is not None and not pd.isna(dt) else None


def to_ordinal(value: Optional[str]) -> Optional[float]:
    dt = parse_date(value)
    return float(dt.toordinal()) if dt is not None and not pd.isna(dt) else None


def bool_to_int(value: Optional[Any]) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        text = str(value).strip().lower()
        return 1 if text in {"yes", "true", "1", "y", "present"} else 0


# Build numeric training matrix for the imputer by applying the same
# transformations used at inference time (category -> int, dates -> ordinal,
# booleans -> int). This ensures the SimpleImputer with median strategy can
# be fitted without errors.
TRAIN_FEATURES = []
for _, r in DF.iterrows():
    x = []
    x.append(normalize_category("specialty", r.get("specialty")))
    x.append(normalize_category("appointment_time", r.get("appointment_time")))
    x.append(normalize_category("gender", r.get("gender")))
    x.append(to_ordinal(r.get("appointment_date")))
    x.append(normalize_category("city", r.get("city")))
    x.append(normalize_category("icd", r.get("icd")))
    x.append(normalize_category("appointment_shift", r.get("appointment_shift")))
    x.append(normalize_category("rain_intensity", r.get("rain_intensity")))
    x.append(normalize_category("heat_intensity", r.get("heat_intensity")))
    x.append(normalize_category("disability", r.get("disability")))
    x.append(normalize_category("appointment_month", r.get("appointment_month")))
    x.append(r.get("appointment_year") if pd.notnull(r.get("appointment_year")) else np.nan)
    x.append(bool_to_int(r.get("under_12_years_old")))
    x.append(bool_to_int(r.get("over_60_years_old")))
    x.append(bool_to_int(r.get("patient_needs_companion")))
    x.append(to_ordinal(r.get("date_of_birth")))
    x.append(to_ordinal(r.get("entry_service_date")))
    x.append(r.get("age") if pd.notnull(r.get("age")) else np.nan)
    x.append(r.get("average_temp_day") if pd.notnull(r.get("average_temp_day")) else np.nan)
    x.append(r.get("average_rain_day") if pd.notnull(r.get("average_rain_day")) else np.nan)
    x.append(r.get("max_temp_day") if pd.notnull(r.get("max_temp_day")) else np.nan)
    x.append(r.get("max_rain_day") if pd.notnull(r.get("max_rain_day")) else np.nan)
    x.append(r.get("rainy_day_before") if pd.notnull(r.get("rainy_day_before")) else np.nan)
    x.append(r.get("storm_day_before") if pd.notnull(r.get("storm_day_before")) else np.nan)
    TRAIN_FEATURES.append(x)

TRAIN_X = np.array(TRAIN_FEATURES, dtype=float)
IMPUTER = SimpleImputer(strategy="median")
IMPUTER.fit(TRAIN_X)


class PredictRequest(BaseModel):
    specialty: Optional[str] = None
    appointment_time: str
    gender: str
    appointment_date: str
    department: Optional[str] = None
    last_appointments: Optional[List[Dict[str, Any]]] = None
    history_status: Optional[str] = None
    city: Optional[str] = None
    icd: Optional[str] = None
    appointment_shift: Optional[str] = None
    appointment_month: Optional[str] = None
    appointment_year: Optional[int] = None
    disability: Optional[str] = None
    date_of_birth: Optional[str] = None
    entry_service_date: Optional[str] = None
    age: Optional[float] = None
    under_12_years_old: Optional[int] = None
    over_60_years_old: Optional[int] = None
    patient_needs_companion: Optional[int] = None
    average_temp_day: Optional[float] = None
    average_rain_day: Optional[float] = None
    max_temp_day: Optional[float] = None
    max_rain_day: Optional[float] = None
    rainy_day_before: Optional[int] = None
    storm_day_before: Optional[int] = None
    rain_intensity: Optional[str] = None
    heat_intensity: Optional[str] = None

    class Config:
        extra = "ignore"


@app.post("/predict")
def predict(payload: PredictRequest):
    appointment_date = payload.appointment_date
    appointment_month = payload.appointment_month or month_from_date(appointment_date)
    appointment_year = payload.appointment_year or year_from_date(appointment_date)
    appointment_shift = payload.appointment_shift or appointment_shift_from_time(payload.appointment_time)
    specialty_value = payload.specialty or payload.department
    if not specialty_value:
        raise HTTPException(status_code=400, detail="Either specialty or department must be provided")

    features = {
        "specialty": specialty_value,
        "appointment_time": payload.appointment_time,
        "gender": payload.gender,
        "appointment_date": appointment_date,
        "city": payload.city,
        "icd": payload.icd,
        "appointment_shift": appointment_shift,
        "rain_intensity": payload.rain_intensity,
        "heat_intensity": payload.heat_intensity,
        "disability": payload.disability,
        "appointment_month": appointment_month,
        "appointment_year": appointment_year,
        "under_12_years_old": payload.under_12_years_old,
        "over_60_years_old": payload.over_60_years_old,
        "patient_needs_companion": payload.patient_needs_companion,
        "date_of_birth": payload.date_of_birth,
        "entry_service_date": payload.entry_service_date,
        "age": payload.age,
        "average_temp_day": payload.average_temp_day,
        "average_rain_day": payload.average_rain_day,
        "max_temp_day": payload.max_temp_day,
        "max_rain_day": payload.max_rain_day,
        "rainy_day_before": payload.rainy_day_before,
        "storm_day_before": payload.storm_day_before,
    }

    x = []
    x.append(normalize_category("specialty", features["specialty"]))
    x.append(normalize_category("appointment_time", features["appointment_time"]))
    x.append(normalize_category("gender", features["gender"]))
    x.append(to_ordinal(features["appointment_date"]))
    x.append(normalize_category("city", features["city"]))
    x.append(normalize_category("icd", features["icd"]))
    x.append(normalize_category("appointment_shift", features["appointment_shift"]))
    x.append(normalize_category("rain_intensity", features["rain_intensity"]))
    x.append(normalize_category("heat_intensity", features["heat_intensity"]))
    x.append(normalize_category("disability", features["disability"]))
    x.append(normalize_category("appointment_month", features["appointment_month"]))
    x.append(features["appointment_year"] if features["appointment_year"] is not None else np.nan)
    x.append(bool_to_int(features["under_12_years_old"]))
    x.append(bool_to_int(features["over_60_years_old"]))
    x.append(bool_to_int(features["patient_needs_companion"]))
    x.append(to_ordinal(features["date_of_birth"]))
    x.append(to_ordinal(features["entry_service_date"]))
    x.append(features["age"] if features["age"] is not None else np.nan)
    x.append(features["average_temp_day"] if features["average_temp_day"] is not None else np.nan)
    x.append(features["average_rain_day"] if features["average_rain_day"] is not None else np.nan)
    x.append(features["max_temp_day"] if features["max_temp_day"] is not None else np.nan)
    x.append(features["max_rain_day"] if features["max_rain_day"] is not None else np.nan)
    x.append(features["rainy_day_before"] if features["rainy_day_before"] is not None else np.nan)
    x.append(features["storm_day_before"] if features["storm_day_before"] is not None else np.nan)

    x = np.array(x, dtype=float).reshape(1, -1)
    x = IMPUTER.transform(x)

    probability = MODEL.predict_proba(x)[0, 1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": float(probability),
        "threshold": 0.5,
        "payload": payload.dict(exclude_none=True),
    }
