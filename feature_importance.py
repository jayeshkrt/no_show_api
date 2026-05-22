import joblib

model = joblib.load('no_show_model.pkl')
feature_names = [
    'specialty', 'appointment_time', 'gender', 'appointment_date_ordinal',
    'city', 'icd', 'appointment_shift', 'rain_intensity', 'heat_intensity',
    'disability', 'appointment_month', 'appointment_year', 'under_12_years_old',
    'over_60_years_old', 'patient_needs_companion', 'date_of_birth_ordinal',
    'entry_service_date_ordinal', 'age', 'average_temp_day', 'average_rain_day',
    'max_temp_day', 'max_rain_day', 'rainy_day_before', 'storm_day_before'
]
importances = model.feature_importances_
for n, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f'{n}: {imp:.6f}')
