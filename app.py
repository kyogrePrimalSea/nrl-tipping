import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRL Tipping Predictor",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark header band */
    .header-band {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 12px;
        padding: 28px 32px 20px 32px;
        margin-bottom: 28px;
        border-left: 6px solid #e94560;
    }
    .header-band h1 { color: #ffffff; margin: 0 0 4px 0; font-size: 2.4rem; }
    .header-band p  { color: #a0aec0; margin: 0; font-size: 1.05rem; }

    /* Tip box */
    .tip-box {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border: 2px solid #e94560;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
        margin-bottom: 20px;
    }
    .tip-box .tip-label { color: #a0aec0; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; }
    .tip-box .tip-team  { color: #ffffff; font-size: 2.6rem; font-weight: 800; margin: 6px 0; }
    .tip-box .tip-conf  { font-size: 1.1rem; margin: 4px 0; }
    .tip-box .tip-model { color: #718096; font-size: 0.82rem; margin-top: 10px; }

    /* Stat cards */
    .stat-card {
        background: #1a202c;
        border-radius: 12px;
        padding: 20px 22px;
        border: 1px solid #2d3748;
        margin-bottom: 16px;
        height: 100%;
    }
    .stat-card h3 { color: #e2e8f0; margin: 0 0 14px 0; font-size: 1.15rem; border-bottom: 2px solid #e94560; padding-bottom: 8px; }
    .stat-row { display: flex; justify-content: space-between; align-items: center; margin: 7px 0; }
    .stat-label { color: #718096; font-size: 0.88rem; }
    .stat-value { color: #e2e8f0; font-weight: 600; font-size: 0.92rem; }
    .positive { color: #68d391; }
    .negative { color: #fc8181; }
    .neutral  { color: #a0aec0; }

    /* Info boxes */
    .info-box {
        background: #1a202c;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2d3748;
        margin-bottom: 16px;
    }
    .info-box h4 { color: #e2e8f0; margin: 0 0 12px 0; font-size: 1rem; }

    /* Prob bar labels */
    .prob-labels { display: flex; justify-content: space-between; color: #a0aec0; font-size: 0.82rem; margin-top: 4px; }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #e94560, #c53030);
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 14px 48px;
        border-radius: 10px;
        border: none;
        width: 100%;
        letter-spacing: 1px;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }

    /* Section divider */
    .section-div { border-top: 1px solid #2d3748; margin: 24px 0; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Artifact loading (cached) ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load(name):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_all_artifacts():
    return {
        "elo_ratings":         _load("elo_ratings.pkl"),
        "team_stats":          _load("current_team_stats.pkl"),
        "h2h_stats":           _load("current_h2h_stats.pkl"),
        "venue_encoder":       _load("venue_encoder.pkl"),
        "venue_monthly_temps": _load("venue_monthly_temps.pkl"),
        "team_home_venue":     _load("team_home_venue.pkl"),
        "scaler_form":         _load("scaler_form.pkl"),
        "scaler_comb":         _load("scaler_combined.pkl"),
        "model_form":          _load("model_xgb_form.pkl"),
        "model_comb":          _load("model_xgb_combined.pkl"),
        "X_form_cols":         _load("feature_columns_form.pkl"),
        "X_comb_cols":         _load("feature_columns_combined.pkl"),
        "team_list":           _load("team_list.pkl"),
        "venue_list":          _load("venue_list.pkl"),
    }


artifacts = load_all_artifacts()

team_list  = sorted(artifacts["team_list"])
venue_list = sorted(artifacts["venue_list"])

# ── Core prediction function ───────────────────────────────────────────────────

def predict_game(
    home_team,
    away_team,
    venue,
    home_days_rest,
    away_days_rest,
    is_finals,
    match_day_temp,
    rain_flag,
    game_month=None,
    home_odds=None,
    away_odds=None,
    total_score=None,
    home_avg_scored=None,
    home_avg_conceded=None,
    home_avg_margin=None,
    home_win_pct=None,
    away_avg_scored=None,
    away_avg_conceded=None,
    away_avg_margin=None,
    away_win_pct=None,
    h2h_home_win_pct=None,
):
    art = artifacts
    elo_ratings         = art["elo_ratings"]
    team_stats          = art["team_stats"]
    h2h_stats           = art["h2h_stats"]
    venue_encoder       = art["venue_encoder"]
    v_monthly_temps     = art["venue_monthly_temps"]
    t_home_venue        = art["team_home_venue"]
    scaler_form         = art["scaler_form"]
    scaler_comb         = art["scaler_comb"]
    model_form          = art["model_form"]
    model_comb          = art["model_comb"]
    X_form_cols         = art["X_form_cols"]
    X_comb_cols         = art["X_comb_cols"]

    if game_month is None:
        game_month = datetime.now().month

    # Elo
    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)
    elo_diff = home_elo - away_elo

    # Form stats
    h_stats = team_stats[team_stats["team"] == home_team]
    a_stats = team_stats[team_stats["team"] == away_team]

    home_avg_scored   = home_avg_scored   if home_avg_scored   is not None else (h_stats["avg_scored"].values[0]   if len(h_stats) > 0 else 20.0)
    home_avg_conceded = home_avg_conceded if home_avg_conceded is not None else (h_stats["avg_conceded"].values[0] if len(h_stats) > 0 else 20.0)
    home_avg_margin   = home_avg_margin   if home_avg_margin   is not None else (h_stats["avg_margin"].values[0]   if len(h_stats) > 0 else 0.0)
    home_win_pct      = home_win_pct      if home_win_pct      is not None else (h_stats["win_pct"].values[0]      if len(h_stats) > 0 else 0.5)
    away_avg_scored   = away_avg_scored   if away_avg_scored   is not None else (a_stats["avg_scored"].values[0]   if len(a_stats) > 0 else 20.0)
    away_avg_conceded = away_avg_conceded if away_avg_conceded is not None else (a_stats["avg_conceded"].values[0] if len(a_stats) > 0 else 20.0)
    away_avg_margin   = away_avg_margin   if away_avg_margin   is not None else (a_stats["avg_margin"].values[0]   if len(a_stats) > 0 else 0.0)
    away_win_pct      = away_win_pct      if away_win_pct      is not None else (a_stats["win_pct"].values[0]      if len(a_stats) > 0 else 0.5)

    # H2H
    if h2h_home_win_pct is None:
        h2h_key = "_".join(sorted([home_team, away_team]))
        team1   = h2h_key.split("_")[0]
        h2h_row = h2h_stats[h2h_stats["h2h_key"] == h2h_key]
        if len(h2h_row) > 0:
            t1_pct           = h2h_row["team1_win_pct"].values[0]
            h2h_home_win_pct = t1_pct if home_team == team1 else 1 - t1_pct
        else:
            h2h_home_win_pct = 0.5

    rest_advantage = home_days_rest - away_days_rest

    # Venue encoding
    try:
        venue_enc = venue_encoder.transform([venue])[0]
    except Exception:
        venue_enc = 0

    # Temperature differential
    away_home_v    = t_home_venue.get(away_team)
    away_home_temp = (
        v_monthly_temps[away_home_v][game_month]
        if away_home_v and away_home_v in v_monthly_temps
        else 20.0
    )
    temp_diff = match_day_temp - away_home_temp

    form_features = {
        "home_avg_scored_last5":   home_avg_scored,
        "home_avg_conceded_last5": home_avg_conceded,
        "away_avg_scored_last5":   away_avg_scored,
        "away_avg_conceded_last5": away_avg_conceded,
        "home_avg_margin_last5":   home_avg_margin,
        "away_avg_margin_last5":   away_avg_margin,
        "home_win_pct_last5":      home_win_pct,
        "away_win_pct_last5":      away_win_pct,
        "h2h_home_win_pct_last5":  h2h_home_win_pct,
        "rest_advantage":          rest_advantage,
        "is_finals":               int(is_finals),
        "venue_encoded":           venue_enc,
        "home_elo":                home_elo,
        "away_elo":                away_elo,
        "elo_diff":                elo_diff,
        "rain_flag":               int(rain_flag),
        "match_day_temp":          match_day_temp,
        "temp_differential":       temp_diff,
    }

    use_combined = home_odds is not None and away_odds is not None
    if use_combined:
        margin_o    = (1 / home_odds) + (1 / away_odds)
        home_impl   = (1 / home_odds) / margin_o
        away_impl   = (1 / away_odds) / margin_o
        certainty   = abs(home_impl - 0.5)
        total_close = total_score if total_score is not None else 40.0
        all_features = {
            **form_features,
            "home_implied_prob": home_impl,
            "away_implied_prob": away_impl,
            "market_certainty":  certainty,
            "total_score_close": total_close,
        }
        X        = pd.DataFrame([all_features])[X_comb_cols]
        X_scaled = scaler_comb.transform(X)
        prob     = model_comb.predict_proba(X_scaled)[0][1]
        model_used = "Combined (Form + Elo + Odds + Weather)"
    else:
        X        = pd.DataFrame([form_features])[X_form_cols]
        X_scaled = scaler_form.transform(X)
        prob     = model_form.predict_proba(X_scaled)[0][1]
        model_used = "Form + Elo + Weather"

    tip        = home_team if prob > 0.5 else away_team
    confidence = abs(prob - 0.5) * 2
    if confidence > 0.3:
        conf_label = "🟢 Strong"
    elif confidence > 0.15:
        conf_label = "🟡 Moderate"
    else:
        conf_label = "🔴 Lean"

    return {
        "home_team":            home_team,
        "away_team":            away_team,
        "venue":                venue,
        "game_month":           game_month,
        "home_win_probability": round(prob, 3),
        "away_win_probability": round(1 - prob, 3),
        "tip":                  tip,
        "confidence":           conf_label,
        "model_used":           model_used,
        "home_elo":             round(home_elo, 1),
        "away_elo":             round(away_elo, 1),
        "temp_on_day":          match_day_temp,
        "away_home_climate":    round(away_home_temp, 1),
        "temp_differential":    round(temp_diff, 1),
        "rain":                 bool(rain_flag),
        "h2h_home_win_pct":     round(h2h_home_win_pct, 3),
        "home_avg_scored":      round(home_avg_scored, 1),
        "home_avg_conceded":    round(home_avg_conceded, 1),
        "home_avg_margin":      round(home_avg_margin, 1),
        "home_win_pct":         round(home_win_pct, 3),
        "away_avg_scored":      round(away_avg_scored, 1),
        "away_avg_conceded":    round(away_avg_conceded, 1),
        "away_avg_margin":      round(away_avg_margin, 1),
        "away_win_pct":         round(away_win_pct, 3),
        # pass-through for market analysis
        "home_implied_prob":    round((1 / home_odds) / ((1 / home_odds) + (1 / away_odds)), 3) if use_combined else None,
        "away_implied_prob":    round((1 / away_odds) / ((1 / home_odds) + (1 / away_odds)), 3) if use_combined else None,
        "market_certainty":     round(abs((1 / home_odds) / ((1 / home_odds) + (1 / away_odds)) - 0.5), 3) if use_combined else None,
    }


# ── Helper renderers ──────────────────────────────────────────────────────────

def margin_color(val):
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    return "neutral"


def render_team_card(label, team, elo, win_pct, avg_scored, avg_conceded, avg_margin):
    elo_delta = elo - 1500
    delta_sign = "+" if elo_delta >= 0 else ""
    margin_cls = margin_color(avg_margin)
    margin_sign = "+" if avg_margin >= 0 else ""

    st.markdown(
        f"""
        <div class="stat-card">
            <h3>{'🏠 ' if label == 'Home' else '✈️ '}{team}
                <span style="font-size:0.7rem;color:#718096;font-weight:400;margin-left:8px;">{label}</span>
            </h3>
            <div class="stat-row">
                <span class="stat-label">Elo Rating</span>
                <span class="stat-value">{elo:.0f}
                    <span style="font-size:0.78rem;color:{'#68d391' if elo_delta >= 0 else '#fc8181'}">
                        ({delta_sign}{elo_delta:.0f})
                    </span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win Rate (last 5)</span>
                <span class="stat-value">{win_pct * 100:.0f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(win_pct), text="")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Scored", f"{avg_scored:.1f}")
    col_b.metric("Avg Conceded", f"{avg_conceded:.1f}")
    col_c.metric(
        "Avg Margin",
        f"{margin_sign}{avg_margin:.1f}",
        delta=f"{margin_sign}{avg_margin:.1f}",
        delta_color="normal",
    )


def render_prob_bar(home_team, away_team, home_prob, away_prob):
    home_pct = int(round(home_prob * 100))
    away_pct = int(round(away_prob * 100))
    st.progress(float(home_prob), text="")
    st.markdown(
        f"""
        <div class="prob-labels">
            <span>🏠 {home_team} &nbsp;<strong style="color:#e2e8f0">{home_pct}%</strong></span>
            <span><strong style="color:#e2e8f0">{away_pct}%</strong>&nbsp; {away_team} ✈️</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-band">
        <h1>🏉 NRL Tipping Predictor</h1>
        <p>Powered by XGBoost + Elo Ratings + Weather Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input Section ─────────────────────────────────────────────────────────────
col_home, col_away = st.columns(2)

with col_home:
    st.markdown("#### 🏠 Home Team")
    home_team      = st.selectbox("Home Team", team_list, key="home_team", label_visibility="collapsed")
    home_days_rest = st.number_input("Days since last game", min_value=1, max_value=21, value=7, key="home_rest")

with col_away:
    st.markdown("#### ✈️ Away Team")
    away_team      = st.selectbox("Away Team", team_list, index=1, key="away_team", label_visibility="collapsed")
    away_days_rest = st.number_input("Days since last game", min_value=1, max_value=21, value=7, key="away_rest")

# Validate same team
if home_team == away_team:
    st.warning("⚠️ Home and Away teams are the same — please select different teams.")

st.markdown("---")

col_venue, col_temp = st.columns([2, 1])
with col_venue:
    venue = st.selectbox("📍 Venue", venue_list)
with col_temp:
    match_day_temp = st.number_input("🌡️ Temperature on game day (°C)", min_value=-5, max_value=45, value=20)

col_rain, col_finals, col_month = st.columns(3)
with col_rain:
    rain_flag = st.toggle("🌧️ Rain expected?", value=False)
with col_finals:
    is_finals = st.toggle("🏆 Finals game?", value=False)
with col_month:
    game_month = st.number_input("📅 Game month (1–12)", min_value=1, max_value=12, value=datetime.now().month)

st.markdown("---")

# ── Optional Form Overrides ───────────────────────────────────────────────────
with st.expander("📊 Override Form Stats (optional — uses historical defaults if blank)"):
    st.caption("Leave at 0 to use the model's stored historical values for each team.")
    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown(f"**{home_team} overrides**")
        o_home_scored    = st.number_input("Home Avg Scored (last 5)",   min_value=0.0, max_value=80.0, value=0.0, step=0.1, key="o_hs")
        o_home_conceded  = st.number_input("Home Avg Conceded (last 5)", min_value=0.0, max_value=80.0, value=0.0, step=0.1, key="o_hc")
        o_home_margin    = st.number_input("Home Avg Margin (last 5)",   min_value=-80.0, max_value=80.0, value=0.0, step=0.1, key="o_hm")
        o_home_win_pct   = st.number_input("Home Win % (0–1)",           min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="o_hwp")

    with oc2:
        st.markdown(f"**{away_team} overrides**")
        o_away_scored    = st.number_input("Away Avg Scored (last 5)",   min_value=0.0, max_value=80.0, value=0.0, step=0.1, key="o_as")
        o_away_conceded  = st.number_input("Away Avg Conceded (last 5)", min_value=0.0, max_value=80.0, value=0.0, step=0.1, key="o_ac")
        o_away_margin    = st.number_input("Away Avg Margin (last 5)",   min_value=-80.0, max_value=80.0, value=0.0, step=0.1, key="o_am")
        o_away_win_pct   = st.number_input("Away Win % (0–1)",           min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="o_awp")

    o_h2h = st.number_input(
        "H2H Home Win % (0–1, leave 0 for auto)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="o_h2h"
    )

    # Convert 0 → None (means "use defaults")
    def _none_if_zero(v):
        return None if v == 0.0 else v

    home_avg_scored   = _none_if_zero(o_home_scored)
    home_avg_conceded = _none_if_zero(o_home_conceded)
    home_avg_margin   = _none_if_zero(o_home_margin)
    home_win_pct_ov   = _none_if_zero(o_home_win_pct)
    away_avg_scored   = _none_if_zero(o_away_scored)
    away_avg_conceded = _none_if_zero(o_away_conceded)
    away_avg_margin   = _none_if_zero(o_away_margin)
    away_win_pct_ov   = _none_if_zero(o_away_win_pct)
    h2h_override      = _none_if_zero(o_h2h)

# ── Optional Odds ─────────────────────────────────────────────────────────────
with st.expander("💰 Enter Odds (optional — enables Combined Model)"):
    st.caption("Leave blank to use the Form + Elo + Weather model only.")
    odds_col1, odds_col2, odds_col3 = st.columns(3)
    with odds_col1:
        home_odds_raw = st.text_input(f"🏠 {home_team} odds (decimal)", placeholder="e.g. 1.75", key="home_odds")
    with odds_col2:
        away_odds_raw = st.text_input(f"✈️ {away_team} odds (decimal)", placeholder="e.g. 2.10", key="away_odds")
    with odds_col3:
        total_score_raw = st.text_input("Total score line (optional)", placeholder="e.g. 40.5", key="total_score")

    def _parse_float(s):
        try:
            v = float(s.strip())
            return v if v > 0 else None
        except Exception:
            return None

    home_odds   = _parse_float(home_odds_raw)   if home_odds_raw   else None
    away_odds   = _parse_float(away_odds_raw)   if away_odds_raw   else None
    total_score = _parse_float(total_score_raw) if total_score_raw else None

    if (home_odds is None) != (away_odds is None):
        st.warning("Please enter BOTH home and away odds, or leave both blank.")
        home_odds = away_odds = None

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🏉  PREDICT", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict_clicked:
    if home_team == away_team:
        st.error("Cannot predict: Home and Away teams must be different.")
        st.stop()

    with st.spinner("Running prediction..."):
        result = predict_game(
            home_team=home_team,
            away_team=away_team,
            venue=venue,
            home_days_rest=int(home_days_rest),
            away_days_rest=int(away_days_rest),
            is_finals=is_finals,
            match_day_temp=float(match_day_temp),
            rain_flag=rain_flag,
            game_month=int(game_month),
            home_odds=home_odds,
            away_odds=away_odds,
            total_score=total_score,
            home_avg_scored=home_avg_scored,
            home_avg_conceded=home_avg_conceded,
            home_avg_margin=home_avg_margin,
            home_win_pct=home_win_pct_ov,
            away_avg_scored=away_avg_scored,
            away_avg_conceded=away_avg_conceded,
            away_avg_margin=away_avg_margin,
            away_win_pct=away_win_pct_ov,
            h2h_home_win_pct=h2h_override,
        )

    st.markdown("<div class='section-div'></div>", unsafe_allow_html=True)

    # ── 1. TIP BOX ────────────────────────────────────────────────────────────
    home_pct = int(round(result["home_win_probability"] * 100))
    away_pct = int(round(result["away_win_probability"] * 100))

    st.markdown(
        f"""
        <div class="tip-box">
            <div class="tip-label">Tip of the match</div>
            <div class="tip-team">{result['tip']}</div>
            <div class="tip-conf">{result['confidence']}</div>
            <div class="tip-model">Model: {result['model_used']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability bar
    st.markdown("**Win Probabilities**")
    render_prob_bar(home_team, away_team, result["home_win_probability"], result["away_win_probability"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. TEAM STAT CARDS ────────────────────────────────────────────────────
    st.markdown("### 📊 Team Stats")
    card_home, card_away = st.columns(2)

    with card_home:
        render_team_card(
            label="Home",
            team=home_team,
            elo=result["home_elo"],
            win_pct=result["home_win_pct"],
            avg_scored=result["home_avg_scored"],
            avg_conceded=result["home_avg_conceded"],
            avg_margin=result["home_avg_margin"],
        )

    with card_away:
        render_team_card(
            label="Away",
            team=away_team,
            elo=result["away_elo"],
            win_pct=result["away_win_pct"],
            avg_scored=result["away_avg_scored"],
            avg_conceded=result["away_avg_conceded"],
            avg_margin=result["away_avg_margin"],
        )

    st.markdown("<div class='section-div'></div>", unsafe_allow_html=True)

    # ── 3. HEAD TO HEAD ───────────────────────────────────────────────────────
    st.markdown("### 🤝 Head to Head (last 5 meetings)")
    h2h_home = result["h2h_home_win_pct"]
    h2h_away = round(1 - h2h_home, 3)
    h2h_home_pct = int(round(h2h_home * 100))
    h2h_away_pct = int(round(h2h_away * 100))

    leader = home_team if h2h_home >= 0.5 else away_team
    leader_pct = max(h2h_home_pct, h2h_away_pct)
    trailer_pct = min(h2h_home_pct, h2h_away_pct)

    st.markdown(
        f"""
        <div class="info-box">
            <h4>🤝 H2H Record</h4>
            <p style="color:#a0aec0;margin:0 0 10px 0">
                <strong style="color:#e2e8f0">{leader}</strong> lead H2H
                <strong style="color:#e2e8f0">{leader_pct}% – {trailer_pct}%</strong>
                (last 5 meetings)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h2h_col1, h2h_col2 = st.columns(2)
    with h2h_col1:
        st.markdown(f"🏠 **{home_team}** {h2h_home_pct}%")
        st.progress(float(h2h_home))
    with h2h_col2:
        st.markdown(f"✈️ **{away_team}** {h2h_away_pct}%")
        st.progress(float(h2h_away))

    st.markdown("<div class='section-div'></div>", unsafe_allow_html=True)

    # ── 4. WEATHER CONTEXT ────────────────────────────────────────────────────
    st.markdown("### 🌤️ Weather Context")
    td = result["temp_differential"]
    if td > 2:
        temp_arrow = "🔴 ↑ Warmer than away team's home climate (potential disadvantage for away)"
    elif td < -2:
        temp_arrow = "🔵 ↓ Colder than away team's home climate (potential disadvantage for away)"
    else:
        temp_arrow = "⚪ Similar to away team's usual climate"

    wcol1, wcol2, wcol3 = st.columns(3)
    wcol1.metric("🌡️ Game Temp", f"{result['temp_on_day']}°C")
    wcol2.metric(f"☀️ {away_team} Home Climate", f"{result['away_home_climate']}°C")
    wcol3.metric("↕️ Differential", f"{td:+.1f}°C")

    st.markdown(
        f"""
        <div class="info-box" style="margin-top:12px">
            <p style="margin:0;color:#a0aec0">{temp_arrow}</p>
            <p style="margin:6px 0 0 0;color:#a0aec0">
                Rain: {'🌧️ Yes' if result['rain'] else '☀️ No'}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 5. MARKET ANALYSIS (odds only) ────────────────────────────────────────
    if result["home_implied_prob"] is not None:
        st.markdown("<div class='section-div'></div>", unsafe_allow_html=True)
        st.markdown("### 💰 Market Analysis")

        mkt_certainty = result["market_certainty"]
        if mkt_certainty > 0.2:
            cert_label = "🟢 High confidence market"
        elif mkt_certainty > 0.1:
            cert_label = "🟡 Moderate confidence market"
        else:
            cert_label = "🔴 Coin flip market"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {home_team} Implied Prob", f"{result['home_implied_prob']*100:.1f}%")
        m2.metric(f"✈️ {away_team} Implied Prob", f"{result['away_implied_prob']*100:.1f}%")
        m3.metric("Market Certainty", f"{mkt_certainty*100:.1f}%")

        st.markdown(
            f"""
            <div class="info-box" style="margin-top:12px">
                <p style="margin:0;color:#a0aec0">{cert_label}</p>
                <p style="margin:6px 0 0 0;color:#718096;font-size:0.85rem">
                    Model used: <strong style="color:#a0aec0">{result['model_used']}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption(
        f"Prediction generated {datetime.now().strftime('%d %b %Y %H:%M')} · "
        "Form model: 62.4% acc (425-game test) · "
        "Combined model: 64.6% acc (65-game test)"
    )
