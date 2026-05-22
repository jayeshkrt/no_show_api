Deploying the No-show Prediction API to Railway

Quick summary
- This repo contains `no_show_api.py`, `no_show_model.pkl`, and `medical-appointments-no-show-en.csv`.
- Start command (used in `Procfile`):
  `uvicorn no_show_api:app --host 0.0.0.0 --port $PORT`

Preflight checklist
- `requirements.txt` includes: fastapi, uvicorn, pandas, scikit-learn, joblib, numpy, requests
- `no_show_model.pkl` is present in the repo root (or provide a download URL and update `no_show_api.py` to fetch it on startup)

Railway deploy steps
1. Push your repo to GitHub (if not already):

```bash
git init
git add .
git commit -m "prepare for railway deploy"
git branch -M main
# replace <your-github-repo-url> with your repo
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Create a Railway account at https://railway.app and connect your GitHub account.
3. New Project → Deploy from GitHub → select this repository.
4. Railway will detect a Python project. In Service Settings confirm the following:
   - Start Command (if not auto-filled): `uvicorn no_show_api:app --host 0.0.0.0 --port $PORT`
   - Build Command: leave default (Railway uses `pip install -r requirements.txt`).
5. Add environment variables if needed (e.g., `MODEL_PATH`, `DATA_PATH`) via Railway dashboard.
6. Deploy. Railway will provide a public HTTPS URL (e.g. `https://<project>.up.railway.app`).

Notes & recommendations
- If `no_show_model.pkl` is large or sensitive, store it in object storage (S3) and download it at app startup using an env var URL and credentials.
- Free Railway projects may sleep after inactivity; consider upgrading if you need always-on availability.
- For secure Apex callouts, create a Named Credential in Salesforce pointing to the Railway HTTPS URL.

Apex callout (example) — use a Named Credential `NoShowAPI` in Salesforce
```apex
HttpRequest req = new HttpRequest();
req.setEndpoint('callout:NoShowAPI/predict');
req.setMethod('POST');
req.setHeader('Content-Type','application/json');
Map<String, Object> payload = new Map<String, Object>{
  'specialty' => 'physiotherapy',
  'appointment_time' => '13:20',
  'gender' => 'M',
  'appointment_date' => '2024-07-30',
  'age' => 68,
  'city' => 'B. CAMBORIU'
};
req.setBody(JSON.serialize(payload));
Http http = new Http();
HTTPResponse res = http.send(req);
System.debug(res.getStatus() + ': ' + res.getBody());
```

If you'd like, I can:
- push these files to GitHub for you (you must provide the remote URL or grant access), or
- walk you through connecting this repo to Railway and completing the deploy; or
- implement an automatic model download at startup (if you prefer not to commit `no_show_model.pkl`).
