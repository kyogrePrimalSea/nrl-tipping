# 🏉 NRL Tipping Predictor

A Streamlit web app that predicts NRL match outcomes using XGBoost models trained on historical match data, Elo ratings, and weather features.

---

## What it does

- Predicts the winner of any NRL match with a win probability and confidence label
- Uses **two XGBoost models**:
  - **Form Model** — uses team form stats, Elo ratings, and weather (no odds required)
  - **Combined Model** — adds bookmaker odds for improved accuracy (activated when you enter odds)
- Displays team stat cards, H2H history, weather context, and optional market analysis
- All predictions run locally with no external API calls

### Model Accuracy

| Model | Test Accuracy | Test Set Size |
|---|---|---|
| Form + Elo + Weather | **62.4%** | 425 games |
| Combined (+ Odds) | **64.6%** | 65 games |

---

## Folder Structure

```
nrl-tipping/
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── .gitignore
├── README.md
└── models/
    ├── model_xgb_form.pkl         # Form-only XGBoost model
    ├── model_xgb_combined.pkl     # Combined (odds) XGBoost model
    ├── scaler_form.pkl            # StandardScaler for form model
    ├── scaler_combined.pkl        # StandardScaler for combined model
    ├── feature_columns_form.pkl   # Feature column list for form model
    ├── feature_columns_combined.pkl
    ├── elo_ratings.pkl            # Current Elo ratings per team
    ├── venue_encoder.pkl          # LabelEncoder for venues
    ├── current_team_stats.pkl     # Last-5 form stats per team
    ├── current_h2h_stats.pkl      # Head-to-head records
    ├── team_list.pkl              # List of all NRL teams
    ├── venue_list.pkl             # List of all venues
    ├── venue_monthly_temps.pkl    # Avg monthly temps per venue
    └── team_home_venue.pkl        # Each team's home ground
```

---

## Run Locally

**Requirements:** Python 3.10+

```bash
# 1. Clone / download the repo
cd nrl-tipping

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (the `models/` pkl files are small enough to include).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, branch, and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud will install dependencies from `requirements.txt` automatically.

> **Note:** If you use private data or API keys, add them via *Settings → Secrets* in the Streamlit Cloud dashboard and access them with `st.secrets`.

---

## How to Retrain (replace pkl files)

All model artefacts live in `models/`. To update the models after a new season:

1. Retrain your XGBoost models in your training notebook/script.
2. Export each artefact with `pickle.dump(obj, open('models/filename.pkl', 'wb'))`.
3. Replace the corresponding `.pkl` files in the `models/` folder.
4. Restart the Streamlit app — `@st.cache_resource` caches artefacts on startup, so a full restart (or *Clear cache* in the app menu) is needed after replacing files.

### Key artefacts to update each season

| File | What to update |
|---|---|
| `elo_ratings.pkl` | Re-run Elo calculation after each round |
| `current_team_stats.pkl` | Rolling last-5 form stats |
| `current_h2h_stats.pkl` | H2H win percentages |
| `model_xgb_*.pkl` | Full retrain if adding new seasons |
| `scaler_*.pkl` | Must be retrained alongside models |

---

## Inputs Explained

| Input | Description |
|---|---|
| Home / Away Team | Select from all current NRL teams |
| Days since last game | Rest days — affects rest_advantage feature |
| Venue | Match venue — encoded and used for home advantage |
| Temperature | Actual forecast temp on game day (°C) |
| Rain | Whether rain is expected |
| Finals | Whether this is a finals match |
| Game month | Auto-detected; used for venue climate lookup |
| Form overrides | Manually set last-5 stats if you have fresher data |
| Odds | Decimal odds from your bookmaker — unlocks Combined Model |

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [XGBoost](https://xgboost.readthedocs.io) — gradient boosting classifier
- [scikit-learn](https://scikit-learn.org) — preprocessing (StandardScaler, LabelEncoder)
- [pandas](https://pandas.pydata.org) / [numpy](https://numpy.org) — data handling
