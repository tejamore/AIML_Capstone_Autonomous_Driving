# AIML Capstone — Autonomous Driving

A two-part capstone project on autonomous vehicles (AV) and intelligent transport
systems (ITS), packaged as a deployable Flask web app.

- **Part 1 — Object Detection:** upload a road image and a MobileNet-SSD deep
  learning model predicts the vehicle type (car / bus / motorbike / bicycle /
  train) and localizes it with a bounding box.
- **Part 2 — Data Science:** an EDA dashboard analyzing Tesla Autopilot
  incident records and their effect on road safety.

## Live demo

Once deployed on Render (see `INSTALLATION.md`), the app is available at:

```
https://<your-service-name>.onrender.com
```

Routes:
| Route | Description |
|---|---|
| `/` | Project overview |
| `/detect` | Upload an image, run vehicle detection |
| `/analytics` | Tesla Autopilot / road-safety EDA dashboard |
| `/api/health` | JSON health check |

## Project structure

```
capstone/
├── app/
│   ├── app.py               # Flask application (routes)
│   ├── detector.py          # Part 1: MobileNet-SSD vehicle detection
│   ├── analysis.py          # Part 2: EDA over Tesla-Deaths.csv
│   ├── download_models.py   # Pre-fetch model weights (used in build step)
│   ├── templates/           # index.html, detect.html, analytics.html, base.html
│   ├── static/style.css     # UI styling
│   └── models/              # (created at runtime) cached MobileNet-SSD weights
├── data/
│   ├── Tesla-Deaths.csv     # sample/synthetic dataset (replace with the real one)
│   └── generate_sample_data.py
├── notebooks/
│   └── AIML_Capstone_Autonomous_Driving.ipynb   # source-code deliverable
├── requirements.txt
├── render.yaml               # Render deployment blueprint
├── Procfile                  # alternative start command
├── runtime.txt                # pinned Python version
├── INSTALLATION.md
└── README.md
```

## How Part 1 works

Rather than training an object-detection CNN from scratch (which needs a
labelled bounding-box dataset and a GPU), the app uses **transfer learning**:
OpenCV's DNN module running a pre-trained MobileNet-SSD (Pascal VOC classes),
which already includes `car`, `bus`, `motorbike`, `bicycle`, and `train`. This
keeps the deployed service small (~25MB of weights, CPU-only, no PyTorch/
TensorFlow) so it runs comfortably on Render's free tier. The same detection
code is reused in the notebook, including a scaffold for fine-tuning a custom
CNN if you have a GPU and labelled data (see notebook section 1.5).

## How Part 2 works

`app/analysis.py` loads `data/Tesla-Deaths.csv`, cleans it (duplicates,
numeric coercion, date parsing), and answers the brief's EDA questions:
events by year/state/country/weekday, deaths per accident, Tesla driver vs.
occupant deaths, cyclist/pedestrian involvement, collisions with other
vehicles, model distribution, and verified Autopilot-linked deaths.

> **Important:** the bundled `Tesla-Deaths.csv` is a small, randomly generated
> **synthetic sample** so the app and notebook run out-of-the-box. Replace it
> with the real dataset (same column names) before treating any numbers here
> as findings — see `data/generate_sample_data.py` for the exact schema.

## Local quick start

```bash
cd app
pip install -r ../requirements.txt
python download_models.py   # needs internet, one-time (~25MB)
python app.py                # http://localhost:5000
```

Full setup and Render deployment steps: see `INSTALLATION.md`.

## Notebook

`notebooks/AIML_Capstone_Autonomous_Driving.ipynb` mirrors the app's logic
(imports `app/detector.py` and `app/analysis.py` directly) so the notebook
and the deployed app always agree, and contains inline charts/results for
both parts plus write-up prompts.

## Tech stack

Flask · gunicorn · OpenCV (DNN, MobileNet-SSD) · pandas · matplotlib ·
seaborn · NumPy · Pillow — all CPU-only, no GPU or paid services required.
