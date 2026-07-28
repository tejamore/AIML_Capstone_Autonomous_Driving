# Installation & Deployment Guide

## 1. Prerequisites

- Python 3.11 or 3.12
- pip
- git (to push the project to GitHub, which Render deploys from)
- A free [Render](https://render.com) account, for live hosting
- A free [GitHub](https://github.com) account

## 2. Run locally

```bash
# 1. Unzip / clone the project, then:
cd capstone

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-download the vehicle-detection model weights (~25MB, needs internet)
cd app
python download_models.py

# 5. Start the app
python app.py
```

Open **http://localhost:5000** in your browser. You should see:
- `/` — project overview
- `/detect` — upload an image and click "Detect vehicles"
- `/analytics` — the EDA dashboard (loads `data/Tesla-Deaths.csv` automatically)

To use the real Tesla incident dataset instead of the bundled synthetic
sample, replace `data/Tesla-Deaths.csv` with your CSV (keep the same column
names — see `data/generate_sample_data.py` for the schema), then restart the
app.

To use your own image dataset for Part 1, place images in
`data/sample_images/` — the notebook's Part 1 loop reads from that folder.

## 3. Deploy live on Render

### Option A — one-click blueprint (recommended)

1. Push this project to a **new GitHub repository** (public or private):
   ```bash
   git init
   git add .
   git commit -m "AIML capstone: autonomous driving"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to the [Render Dashboard](https://dashboard.render.com/) → **New** →
   **Blueprint**.
3. Connect the GitHub repository you just pushed. Render will detect
   `render.yaml` automatically and pre-fill:
   - Build command: `pip install -r requirements.txt && python app/download_models.py`
   - Start command: `gunicorn --chdir app app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - Plan: Free
4. Click **Apply** / **Create Web Service**. The first build takes a few
   minutes (installs dependencies + downloads the ~25MB model file).
5. Once the deploy finishes, Render gives you a URL like
   `https://aiml-capstone-autonomous-driving.onrender.com` — that's your
   live app.

### Option B — manual web service (no render.yaml)

1. Render Dashboard → **New** → **Web Service** → connect your repo.
2. Environment: **Python 3**
3. Build command:
   ```
   pip install -r requirements.txt && python app/download_models.py
   ```
4. Start command:
   ```
   gunicorn --chdir app app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
   ```
5. Instance type: **Free**
6. Click **Create Web Service**.

### Notes on the free tier

- Render's free web services **spin down after ~15 minutes of inactivity**
  and take ~30–60 seconds to wake up on the next request — this is expected,
  not a bug.
- Free tier has 512MB RAM. The MobileNet-SSD + OpenCV footprint fits
  comfortably; avoid adding PyTorch/TensorFlow unless you upgrade the plan.
- `/api/health` is wired up as the Render health-check path in `render.yaml`.

## 4. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/detect` shows "Detection failed: HTTP Error ..." | Model weights couldn't download. Confirm the build command ran `download_models.py`, or that your network/firewall allows GitHub raw content. |
| `/analytics` shows "Could not load analytics" | `data/Tesla-Deaths.csv` is missing or has different column names — check `data/generate_sample_data.py` for the expected schema. |
| App works locally but not on Render | Check the Render **Logs** tab for the real stack trace; most often it's a missing build step or a RAM limit on the free tier. |
| Slow first request after idling | Normal free-tier cold start; consider Render's paid "Starter" plan to avoid spin-down. |

## 5. Updating the deployed app

Any `git push` to the connected branch triggers an automatic redeploy on
Render (auto-deploy is on by default for Blueprints).
