"""
dashboard.py — Streamlit dashboard for the Short Squeeze Scanner

Run with:  streamlit run dashboard.py

Design philosophy: raw trading terminal aesthetic — dark, dense, data-first.
Feels like something a trader built for themselves, not a polished product.
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings("ignore")
load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Squeeze Scanner",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark terminal aesthetic ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #0d0f12;
    color: #c8ccd4;
}

.stApp { background-color: #0d0f12; }

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: -0.5px;
}

.metric-card {
    background: #151820;
    border: 1px solid #1e2535;
    border-left: 3px solid #e84040;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 8px;
}

.metric-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5a6478;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #e0e4ef;
}

.metric-sub {
    font-size: 11px;
    color: #5a6478;
    margin-top: 2px;
}

.ticker-row-high {
    background: rgba(232, 64, 64, 0.08);
    border-left: 3px solid #e84040;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 2px;
}

.ticker-row-med {
    background: rgba(255, 180, 0, 0.06);
    border-left: 3px solid #ffb400;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 2px;
}

.bear-case-box {
    background: #111418;
    border: 1px solid #1e2535;
    border-radius: 4px;
    padding: 20px 24px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: #a8b0c0;
    white-space: pre-wrap;
}

.signal-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.badge-red { background: rgba(232,64,64,0.2); color: #e84040; border: 1px solid rgba(232,64,64,0.3); }
.badge-yellow { background: rgba(255,180,0,0.15); color: #ffb400; border: 1px solid rgba(255,180,0,0.3); }
.badge-gray { background: rgba(90,100,120,0.2); color: #5a6478; border: 1px solid #1e2535; }

.section-header {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #5a6478;
    border-bottom: 1px solid #1e2535;
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
}

div[data-testid="metric-container"] {
    background: #151820;
    border: 1px solid #1e2535;
    border-radius: 4px;
    padding: 10px 14px;
}

div[data-testid="metric-container"] label {
    font-size: 10px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #5a6478 !important;
}

.stSelectbox > div > div { background: #151820; border-color: #1e2535; }
.stDataFrame { background: #151820; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1a1f2e;
}

/* Remove Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

.run-button > button {
    background: #e84040 !important;
    color: white !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  DB helpers
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    db_path = os.getenv("DB_PATH", "data/squeeze_scanner.db")
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def load_latest_predictions(engine, days: int = 7) -> pd.DataFrame:
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = f"""
            SELECT * FROM prediction_results
            WHERE run_date >= '{cutoff.strftime('%Y-%m-%d')}'
            ORDER BY run_date DESC, squeeze_probability DESC
        """
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


def load_ticker_history(engine, ticker: str, days: int = 90) -> pd.DataFrame:
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = f"""
            SELECT * FROM ticker_snapshots
            WHERE ticker = '{ticker}' AND snapshot_date >= '{cutoff}'
            ORDER BY snapshot_date ASC
        """
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()


def load_watchlist() -> pd.DataFrame:
    try:
        return pd.read_csv("data/watchlist.csv")
    except Exception:
        return pd.DataFrame(columns=["ticker", "sector", "short_thesis"])


# ──────────────────────────────────────────────────────────────
#  Chart helpers
# ──────────────────────────────────────────────────────────────

CHART_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0f12",
    plot_bgcolor="#0d0f12",
    font=dict(family="IBM Plex Mono", color="#c8ccd4", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1a1f2e", linecolor="#1a1f2e", showgrid=True),
    yaxis=dict(gridcolor="#1a1f2e", linecolor="#1a1f2e", showgrid=True),
)


def price_chart(ticker: str) -> go.Figure:
    try:
        df = yf.Ticker(ticker).history(period="3mo")
        if df.empty:
            return go.Figure()
        df = df.reset_index()

        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#26a65b",
            decreasing_line_color="#e84040",
            name=ticker,
            showlegend=False,
        ))

        # Volume bars
        colors = ["#26a65b" if c >= o else "#e84040"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df["Date"],
            y=df["Volume"],
            marker_color=colors,
            marker_opacity=0.4,
            name="Volume",
            yaxis="y2",
            showlegend=False,
        ))

        fig.update_layout(
            **CHART_TEMPLATE,
            height=320,
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        tickformat=".2s", title="Volume"),
            xaxis_rangeslider_visible=False,
            title=dict(text=f"{ticker} — 3 months", x=0, font=dict(size=12)),
        )
        return fig
    except Exception:
        return go.Figure()


def score_history_chart(df_pred: pd.DataFrame, ticker: str) -> go.Figure:
    sub = df_pred[df_pred["ticker"] == ticker].sort_values("run_date")
    if sub.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["run_date"],
        y=sub["squeeze_probability"],
        mode="lines+markers",
        line=dict(color="#e84040", width=2),
        marker=dict(size=6, color="#e84040"),
        name="Squeeze prob",
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#ffb400",
                  annotation_text="alert threshold", annotation_font_size=10)

    fig.update_layout(
        **CHART_TEMPLATE,
        height=200,
        yaxis=dict(**CHART_TEMPLATE["yaxis"], tickformat=".0%", range=[0, 1]),
        title=dict(text="Score history", x=0, font=dict(size=11)),
    )
    return fig


def watchlist_heatmap(df_pred: pd.DataFrame) -> go.Figure:
    """Bubble chart: x=SI%, y=vol spike, size=score, color=score"""
    if df_pred.empty:
        return go.Figure()

    latest = df_pred.sort_values("run_date").groupby("ticker").last().reset_index()

    try:
        features = latest["features_json"].apply(
            lambda x: json.loads(x) if pd.notna(x) else {}
        )
        latest["si_pct"] = features.apply(lambda x: x.get("short_interest_pct") or 0)
        latest["vol_spike"] = features.apply(lambda x: x.get("volume_spike_ratio") or 1)
    except Exception:
        latest["si_pct"] = 0
        latest["vol_spike"] = 1

    fig = px.scatter(
        latest,
        x="si_pct",
        y="vol_spike",
        size="squeeze_probability",
        color="squeeze_probability",
        text="ticker",
        color_continuous_scale=[(0, "#1e2535"), (0.5, "#ffb400"), (1, "#e84040")],
        size_max=50,
        labels={
            "si_pct": "Short Interest %",
            "vol_spike": "Volume Spike Ratio",
            "squeeze_probability": "Score",
        },
    )
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.update_layout(
        **CHART_TEMPLATE,
        height=380,
        coloraxis_colorbar=dict(tickformat=".0%"),
        title=dict(text="Signal map — SI% vs Volume spike", x=0, font=dict(size=12)),
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────

engine = get_engine()
watchlist_df = load_watchlist()
all_tickers = watchlist_df["ticker"].tolist() if not watchlist_df.empty else []

with st.sidebar:
    st.markdown("### 📉 SQUEEZE SCANNER")
    st.markdown("<div class='section-header'>CONTROLS</div>", unsafe_allow_html=True)

    selected_ticker = st.selectbox(
        "Inspect ticker",
        options=all_tickers,
        index=0 if all_tickers else 0,
    )

    days_back = st.slider("History window (days)", 1, 30, 7)

    st.markdown("<div class='section-header'>THRESHOLDS</div>", unsafe_allow_html=True)
    alert_thresh = st.slider("Alert threshold", 0.40, 0.90, 0.65, 0.05)
    moderate_thresh = st.slider("Moderate threshold", 0.30, 0.70, 0.45, 0.05)

    st.markdown("<div class='section-header'>RUN</div>", unsafe_allow_html=True)

    st.markdown("<div class='run-button'>", unsafe_allow_html=True)
    if st.button("▶  RUN SCAN NOW", use_container_width=True):
        with st.spinner("Scanning..."):
            try:
                import subprocess, sys
                result = subprocess.run(
                    [sys.executable, "run_scanner.py", "--no-ai"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    st.success("Scan complete")
                else:
                    st.error(f"Scan failed:\n{result.stderr[:300]}")
            except Exception as e:
                st.error(f"Could not run scan: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>STATUS</div>", unsafe_allow_html=True)
    model_exists = os.path.exists("models/squeeze_model.pkl")
    db_exists = os.path.exists(os.getenv("DB_PATH", "data/squeeze_scanner.db"))
    st.markdown(f"Model: {'✅ loaded' if model_exists else '❌ not trained'}")
    st.markdown(f"DB: {'✅ connected' if db_exists else '❌ not found'}")
    st.markdown(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ──────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ──────────────────────────────────────────────────────────────

df_pred = load_latest_predictions(engine, days=days_back)

# ── Header ────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-size:20px;margin-bottom:4px;'>SHORT SQUEEZE SCANNER</h1>"
    f"<div style='font-size:11px;color:#5a6478;margin-bottom:20px;'>"
    f"Personal watchlist monitor — {datetime.now().strftime('%A, %B %d %Y  %H:%M')}"
    f"</div>",
    unsafe_allow_html=True
)

# ── KPI row ───────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_tickers = len(all_tickers)
alerts = len(df_pred[df_pred["squeeze_probability"] >= alert_thresh]) if not df_pred.empty else 0
moderate = len(df_pred[(df_pred["squeeze_probability"] >= moderate_thresh) &
                        (df_pred["squeeze_probability"] < alert_thresh)]) if not df_pred.empty else 0
avg_score = df_pred["squeeze_probability"].mean() if not df_pred.empty else 0
top_ticker = df_pred.sort_values("squeeze_probability", ascending=False).iloc[0]["ticker"] \
    if not df_pred.empty else "—"

with col1:
    st.metric("Watchlist", total_tickers)
with col2:
    st.metric("🔴 Alerts", alerts)
with col3:
    st.metric("🟡 Moderate", moderate)
with col4:
    st.metric("Avg score", f"{avg_score:.0%}" if avg_score else "—")
with col5:
    st.metric("Top signal", top_ticker)

st.markdown("---")

# ── Two column layout ─────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="medium")

with left:
    # ── Signal ranking table ──────────────────────────────────
    st.markdown("<div class='section-header'>SIGNAL RANKING</div>", unsafe_allow_html=True)

    if df_pred.empty:
        st.info("No scan data yet. Run the scanner first: `python run_scanner.py`")
    else:
        latest = df_pred.sort_values("run_date").groupby("ticker").last().reset_index()
        latest = latest.sort_values("squeeze_probability", ascending=False)

        for _, row in latest.iterrows():
            prob = row["squeeze_probability"]
            ticker = row["ticker"]

            if prob >= alert_thresh:
                badge = f"<span class='signal-badge badge-red'>ALERT  {prob:.0%}</span>"
                row_class = "ticker-row-high"
            elif prob >= moderate_thresh:
                badge = f"<span class='signal-badge badge-yellow'>WATCH  {prob:.0%}</span>"
                row_class = "ticker-row-med"
            else:
                badge = f"<span class='signal-badge badge-gray'>NEUTRAL  {prob:.0%}</span>"
                row_class = ""

            features = {}
            try:
                features = json.loads(row.get("features_json") or "{}")
            except Exception:
                pass

            si = features.get("short_interest_pct") or "—"
            vol = round(features.get("volume_spike_ratio") or 0, 1)
            borrow = features.get("borrow_rate_pct") or "—"

            st.markdown(
                f"<div class='{row_class}' style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<div>"
                f"<span style='font-size:15px;font-weight:600;color:#e0e4ef;'>{ticker}</span>"
                f"&nbsp;&nbsp;<span style='font-size:11px;color:#5a6478;'>SI:{si}%  Vol:{vol}x  Borrow:{borrow}%</span>"
                f"</div>"
                f"{badge}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Heatmap ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>SIGNAL MAP</div>", unsafe_allow_html=True)
    if not df_pred.empty:
        st.plotly_chart(watchlist_heatmap(df_pred), use_container_width=True)


with right:
    # ── Selected ticker deep-dive ─────────────────────────────
    if selected_ticker:
        st.markdown(
            f"<div class='section-header'>DEEP DIVE — {selected_ticker}</div>",
            unsafe_allow_html=True
        )

        ticker_pred = df_pred[df_pred["ticker"] == selected_ticker]
        latest_pred = ticker_pred.sort_values("run_date").iloc[-1] \
            if not ticker_pred.empty else None

        if latest_pred is not None:
            prob = latest_pred["squeeze_probability"]
            color = "#e84040" if prob >= alert_thresh else "#ffb400" if prob >= moderate_thresh else "#5a6478"
            st.markdown(
                f"<div style='font-size:36px;font-weight:700;color:{color};letter-spacing:-1px;'>"
                f"{prob:.0%}"
                f"<span style='font-size:13px;font-weight:400;color:#5a6478;margin-left:8px;'>squeeze score</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            features = {}
            try:
                features = json.loads(latest_pred.get("features_json") or "{}")
            except Exception:
                pass

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Short Int.", f"{features.get('short_interest_pct') or '—'}%")
            with m2:
                st.metric("Vol Spike", f"{round(features.get('volume_spike_ratio') or 0, 1)}x")
            with m3:
                st.metric("RSI-14", round(features.get("rsi_14") or 0, 1))

        # Price chart
        st.plotly_chart(price_chart(selected_ticker), use_container_width=True)

        # Score history
        if not df_pred.empty:
            st.plotly_chart(score_history_chart(df_pred, selected_ticker), use_container_width=True)

        # Bear case
        if latest_pred is not None and pd.notna(latest_pred.get("bear_case_summary")):
            st.markdown("<div class='section-header'>BEAR CASE</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='bear-case-box'>{latest_pred['bear_case_summary']}</div>",
                unsafe_allow_html=True
            )

        # Watchlist context
        wl_row = watchlist_df[watchlist_df["ticker"] == selected_ticker]
        if not wl_row.empty:
            st.markdown("<div class='section-header'>WHY WATCHING</div>", unsafe_allow_html=True)
            row = wl_row.iloc[0]
            st.markdown(
                f"<div style='font-size:12px;color:#8090a8;line-height:1.6;'>"
                f"<b style='color:#c8ccd4;'>Sector:</b> {row.get('sector', '—')}<br>"
                f"<b style='color:#c8ccd4;'>Thesis:</b> {row.get('short_thesis', '—')}"
                f"</div>",
                unsafe_allow_html=True
            )
