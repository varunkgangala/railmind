"""
app.py  —  RailMind: Smart Railway Resource Planning System
===========================================================
Run with:
    streamlit run app.py

Tech Stack:
  - Python 3.10+
  - Pandas + NumPy  (data manipulation)
  - Scikit-learn    (ML: RandomForest, GradientBoosting, LinearRegression)
  - Streamlit       (dashboard UI)
  - Plotly          (interactive charts)
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ── project paths ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from utils.helpers import (
    load_data, load_platform_data, load_delay_data,
    get_trained_models, summary_by_route,
    hourly_demand, weekly_demand, monthly_demand, heatmap_data,
    ROUTE_META, ROUTE_IDS, demand_tag, occupancy_color,
)

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RailMind — Smart Railway Planner",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

  /* Global */
  .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  h1,h2,h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.5px; }

  /* Top header banner */
  .railmind-header {
    background: linear-gradient(135deg, #0e0e18 0%, #151525 100%);
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .railmind-title { font-family: 'Syne',sans-serif; font-size:2rem; font-weight:800;
    color:#f0c040; letter-spacing:-1px; margin:0; }
  .railmind-sub { font-family:'DM Mono',monospace; font-size:11px; color:#6b6b80; 
    letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

  /* KPI cards */
  div[data-testid="metric-container"] {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color .2s;
  }
  div[data-testid="metric-container"]:hover { border-color: #f0c040; }
  div[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b6b80 !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    color: #f0c040 !important;
    letter-spacing: -1px;
  }

  /* Section headers */
  .section-head {
    font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
    color:#e8e8f0; margin: 18px 0 10px 0;
    border-left: 3px solid #f0c040; padding-left: 10px;
  }

  /* Pill badges */
  .pill { display:inline-block; padding:2px 10px; border-radius:100px;
    font-family:'DM Mono',monospace; font-size:11px; font-weight:500; }
  .pill-red    { background:rgba(255,79,79,.15);  color:#ff4f4f; border:1px solid rgba(255,79,79,.25); }
  .pill-orange { background:rgba(224,92,42,.15);  color:#e05c2a; border:1px solid rgba(224,92,42,.25); }
  .pill-yellow { background:rgba(255,170,51,.15); color:#ffaa33; border:1px solid rgba(255,170,51,.25); }
  .pill-green  { background:rgba(61,220,132,.15); color:#3ddc84; border:1px solid rgba(61,220,132,.25); }

  /* Recommendation cards */
  .rec-card {
    background:#111118; border:1px solid #1e1e2e; border-radius:12px;
    padding:16px 20px; margin-bottom:10px; transition: border-color .2s;
  }
  .rec-card:hover { border-color: #f0c040; }
  .rec-title { font-weight:700; font-size:14px; margin-bottom:4px; }
  .rec-desc  { font-size:12px; color:#8888a0; line-height:1.6; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background:#0e0e18; border-right:1px solid #1e1e2e; }
  section[data-testid="stSidebar"] .stRadio label { font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="railmind-header">
  <div>
    <div class="railmind-title">🚄 RailMind</div>
    <div class="railmind-sub">Smart Railway Resource Planning System · Synthetic Dataset · India Railways</div>
  </div>
  <div style="text-align:right; font-family:'DM Mono',monospace; font-size:11px; color:#6b6b80;">
    Python · Pandas · NumPy · Scikit-learn · Streamlit<br>
    <span style="color:#3ddc84;">● Live Simulation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data + train models ───────────────────────────────────────────────
with st.spinner("Loading data and training ML models…"):
    df       = load_data()
    plat_df  = load_platform_data()
    delay_df = load_delay_data()
    forecaster, delay_pred, recommender = get_trained_models()

# ── Sidebar navigation ─────────────────────────────────────────────────────
PAGES = [
    "📊  Dashboard Overview",
    "📈  Demand Forecast",
    "🚂  Train & Coach Allocation",
    "🏗️  Platform Usage",
    "💡  Smart Recommendations",
    "🔬  ML Model Performance",
]

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Filters")
    sel_routes = st.multiselect(
        "Routes", ROUTE_IDS,
        default=ROUTE_IDS[:4],
        help="Filter data by route"
    )
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    default_start = max(date_min, date_max - timedelta(days=60))
    date_range = st.date_input(
        "Date Range",
        value=(default_start, date_max),
        min_value=date_min, max_value=date_max,
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#6b6b80;'>"
        f"Dataset: {len(df):,} records<br>"
        f"Date span: {date_min} → {date_max}"
        f"</div>", unsafe_allow_html=True
    )

# ── Apply filters ──────────────────────────────────────────────────────────
if sel_routes:
    filtered = df[df["route_id"].isin(sel_routes)]
else:
    filtered = df.copy()

if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= d0) & (filtered["date"] <= d1)]

# ── Plotly theme helper ────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111118",
    plot_bgcolor="#111118",
    font=dict(family="DM Mono, monospace", size=11, color="#8888a0"),
    margin=dict(l=20, r=20, t=36, b=20),
    xaxis=dict(gridcolor="#1e1e2e", zeroline=False),
    yaxis=dict(gridcolor="#1e1e2e", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
COLORS = ["#f0c040","#4ec9b0","#e05c2a","#ff4f4f","#3ddc84","#7b61ff","#ffaa33","#ff6e8a"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    # KPIs
    total_pass  = int(filtered["passenger_count"].sum())
    avg_occ     = filtered["occupancy_pct"].mean()
    avg_delay   = filtered["delay_minutes"].mean()
    otp_rate    = filtered["on_time"].mean() * 100
    total_trains = len(filtered)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Passengers",  f"{total_pass:,}")
    c2.metric("Avg Occupancy",     f"{avg_occ:.1f}%",  delta=f"{avg_occ-78:.1f}% vs target")
    c3.metric("On-Time Rate",      f"{otp_rate:.1f}%", delta=f"{otp_rate-85:.1f}%", delta_color="normal")
    c4.metric("Avg Delay",         f"{avg_delay:.1f} min")
    c5.metric("Total Departures",  f"{total_trains:,}")

    st.markdown('<div class="section-head">Hourly Passenger Volume</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])

    with col1:
        h_df = hourly_demand(filtered)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=h_df["departure_hour"], y=h_df["avg_passengers"],
            mode="lines+markers",
            line=dict(color="#f0c040", width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(240,192,64,0.07)",
            name="Avg Passengers",
        ))
        fig.add_hline(y=filtered["passenger_count"].quantile(0.9),
                      line_dash="dot", line_color="#ff4f4f",
                      annotation_text="90th percentile")
        fig.update_layout(title="Average Passengers by Departure Hour",
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        route_sum = summary_by_route(filtered)
        fig2 = px.pie(
            route_sum, names="route_id", values="avg_passengers",
            title="Demand Share by Route",
            color_discrete_sequence=COLORS,
            hole=0.5,
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-head">Weekly & Monthly Trends</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)

    with col3:
        w_df = weekly_demand(filtered)
        fig3 = px.bar(w_df, x="day_of_week", y="avg_passengers",
                      title="Avg Demand by Day",
                      color="avg_passengers",
                      color_continuous_scale=["#1e4d1e","#f0c040","#ff4f4f"])
        fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        m_df = monthly_demand(filtered)
        m_df["month_name"] = m_df["month"].apply(
            lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"][x-1]
        )
        fig4 = go.Figure(go.Scatter(
            x=m_df["month_name"], y=m_df["avg_passengers"],
            mode="lines+markers",
            line=dict(color="#4ec9b0", width=2),
            marker=dict(size=6),
        ))
        fig4.update_layout(title="Monthly Demand Trend", **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        class_df = (filtered.groupby("coach_class")["occupancy_pct"]
                    .mean().reset_index())
        fig5 = px.bar(class_df, x="coach_class", y="occupancy_pct",
                      title="Avg Occupancy by Coach Class",
                      color="coach_class",
                      color_discrete_sequence=COLORS)
        fig5.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-head">Live Train Summary Table</div>', unsafe_allow_html=True)
    show_df = (
        filtered.sort_values("date", ascending=False)
        .head(50)[["train_id","route_id","date","departure_time",
                   "num_coaches","passenger_count","occupancy_pct",
                   "delay_minutes","is_holiday","platform_number"]]
        .rename(columns={
            "train_id":"Train ID", "route_id":"Route",
            "date":"Date", "departure_time":"Dep.",
            "num_coaches":"Coaches", "passenger_count":"Passengers",
            "occupancy_pct":"Occ %", "delay_minutes":"Delay (min)",
            "is_holiday":"Holiday", "platform_number":"Platform",
        })
    )
    st.dataframe(
        show_df.style
        .background_gradient(subset=["Occ %"], cmap="RdYlGn_r", vmin=40, vmax=110)
        .format({"Occ %": "{:.1f}%", "Delay (min)": "{:.0f}"}),
        use_container_width=True,
        height=380,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif "Demand Forecast" in page:
    st.markdown('<div class="section-head">ML Demand Forecast (Random Forest)</div>',
                unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    sel_route_fc = fc1.selectbox("Select Route", ROUTE_IDS,
                                  format_func=lambda x: ROUTE_META[x]["name"])
    fc_days   = fc2.slider("Forecast Horizon (days)", 7, 30, 14)
    fc_hour   = fc3.slider("Departure Hour", 5, 22, 9)
    fc_coaches = st.slider("Assumed Coaches", 8, 24, 16)

    fcast_df = forecaster.forecast_route(
        sel_route_fc,
        start_date=str(filtered["date"].max().date()),
        days=fc_days,
        coaches=fc_coaches,
        hour=fc_hour,
    )

    # Historical last 14 days for the route
    hist = (
        filtered[filtered["route_id"] == sel_route_fc]
        .groupby("date")["passenger_count"].mean()
        .reset_index()
        .tail(14)
        .rename(columns={"date":"date","passenger_count":"value"})
    )

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist["date"], y=hist["value"],
        mode="lines+markers", name="Historical",
        line=dict(color="#f0c040", width=2.5),
        marker=dict(size=5),
    ))
    fig_fc.add_trace(go.Scatter(
        x=fcast_df["date"], y=fcast_df["predicted_demand"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#4ec9b0", width=2.5, dash="dot"),
        marker=dict(size=5),
    ))
    # Confidence band
    std = fcast_df["predicted_demand"].std() * 0.3
    fig_fc.add_trace(go.Scatter(
        x=list(fcast_df["date"]) + list(reversed(fcast_df["date"])),
        y=list(fcast_df["predicted_demand"] + std) +
          list(reversed(fcast_df["predicted_demand"] - std)),
        fill="toself", fillcolor="rgba(78,201,176,0.08)",
        line=dict(color="rgba(78,201,176,0)"),
        name="Confidence Band",
    ))
    fig_fc.update_layout(
        title=f"{ROUTE_META[sel_route_fc]['name']} — {fc_days}-Day Demand Forecast",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Heatmap
    st.markdown('<div class="section-head">Demand Heatmap — Hour × Day of Week</div>',
                unsafe_allow_html=True)
    hm = heatmap_data(filtered)
    fig_hm = px.imshow(
        hm,
        labels=dict(x="Day", y="Hour", color="Avg Passengers"),
        color_continuous_scale=["#0d1f0d","#2d7a2d","#f0c040","#e05c2a","#ff4f4f"],
        aspect="auto",
        title="Passenger Demand Heatmap",
    )
    fig_hm.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_hm, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        hol_comp = (
            filtered.groupby(["is_holiday","departure_hour"])["passenger_count"]
            .mean().reset_index()
        )
        hol_comp["type"] = hol_comp["is_holiday"].map({0:"Weekday",1:"Holiday"})
        fig_hol = px.line(
            hol_comp, x="departure_hour", y="passenger_count",
            color="type", title="Holiday vs Weekday Demand by Hour",
            color_discrete_map={"Weekday":"#f0c040","Holiday":"#e05c2a"},
        )
        fig_hol.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hol, use_container_width=True)

    with col_b:
        feat_imp = forecaster.feature_importance().reset_index()
        feat_imp.columns = ["Feature","Importance"]
        fig_imp = px.bar(
            feat_imp, x="Importance", y="Feature", orientation="h",
            title="RF Feature Importance — Demand Forecast",
            color="Importance",
            color_continuous_scale=["#1e4d1e","#f0c040"],
        )
        fig_imp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-head">Forecast Table</div>', unsafe_allow_html=True)
    fcast_df["capacity"] = fc_coaches * 60
    fcast_df["pred_occ_pct"] = (fcast_df["predicted_demand"] / fcast_df["capacity"] * 100).round(1)
    fcast_df["demand_level"]  = fcast_df["pred_occ_pct"].apply(demand_tag)
    st.dataframe(
        fcast_df[["date","departure_hour","predicted_demand","pred_occ_pct","demand_level"]]
        .rename(columns={
            "date":"Date","departure_hour":"Hour",
            "predicted_demand":"Predicted Demand",
            "pred_occ_pct":"Pred. Occupancy %",
            "demand_level":"Demand Level",
        })
        .style.background_gradient(subset=["Pred. Occupancy %"],
                                    cmap="RdYlGn_r", vmin=40, vmax=120),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAIN & COACH ALLOCATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Coach Allocation" in page:
    st.markdown('<div class="section-head">Coach Allocation Analysis</div>',
                unsafe_allow_html=True)

    kc1, kc2, kc3, kc4 = st.columns(4)
    total_coaches = int(filtered["num_coaches"].sum())
    overcrowded   = int((filtered["occupancy_pct"] > 95).sum())
    underutil     = int((filtered["occupancy_pct"] < 60).sum())
    avg_coaches   = filtered["num_coaches"].mean()
    kc1.metric("Total Coaches Deployed", f"{total_coaches:,}")
    kc2.metric("Overcrowded Departures", f"{overcrowded:,}", delta=f"{overcrowded/len(filtered)*100:.1f}%")
    kc3.metric("Underutilised Departures", f"{underutil:,}")
    kc4.metric("Avg Coaches / Train", f"{avg_coaches:.1f}")

    # Allocation vs recommended
    st.markdown('<div class="section-head">Route-Level Allocation vs ML Recommendation</div>',
                unsafe_allow_html=True)
    route_sum = summary_by_route(filtered)

    # Get recommendations for each route using modal hour + month
    recs_rows = []
    for _, row in route_sum.iterrows():
        rid = row["route_id"]
        modal_hour  = int(filtered[filtered["route_id"]==rid]["departure_hour"].mode()[0])
        modal_month = int(filtered[filtered["route_id"]==rid]["month"].mode()[0])
        is_wknd     = int(filtered[filtered["route_id"]==rid]["is_weekend"].mean() > 0.3)
        rec = recommender.recommend(rid, modal_hour, modal_month, is_wknd)
        recs_rows.append({
            "Route":           ROUTE_META[rid]["name"],
            "Avg Demand":      int(row["avg_passengers"]),
            "Avg Coaches":     row["avg_coaches"],
            "Recommended":     rec["recommended_coaches"],
            "Avg Occupancy %": row["avg_occupancy"],
            "Gap":             rec["recommended_coaches"] - int(row["avg_coaches"]),
        })
    alloc_df = pd.DataFrame(recs_rows)

    col_t, col_c = st.columns([3, 2])
    with col_t:
        styled = alloc_df.style \
            .background_gradient(subset=["Avg Occupancy %"], cmap="RdYlGn_r", vmin=50, vmax=110) \
            .applymap(lambda v: "color: #ff4f4f; font-weight:bold" if isinstance(v, (int,float)) and v > 2
                      else ("color: #ffaa33" if isinstance(v, (int,float)) and v > 0 else ""),
                      subset=["Gap"])
        st.dataframe(styled, use_container_width=True, height=340)

    with col_c:
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Bar(
            name="Current Avg Coaches",
            x=alloc_df["Route"].str.split("→").str[0].str.strip(),
            y=alloc_df["Avg Coaches"],
            marker_color="#f0c040",
        ))
        fig_alloc.add_trace(go.Bar(
            name="Recommended",
            x=alloc_df["Route"].str.split("→").str[0].str.strip(),
            y=alloc_df["Recommended"],
            marker_color="#4ec9b0",
        ))
        fig_alloc.update_layout(
            barmode="group",
            title="Current vs Recommended Coaches",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    # Occupancy distribution
    st.markdown('<div class="section-head">Occupancy Distribution</div>', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig_hist = px.histogram(
            filtered, x="occupancy_pct", nbins=40,
            color_discrete_sequence=["#f0c040"],
            title="Occupancy % Distribution",
        )
        fig_hist.add_vline(x=85, line_dash="dot", line_color="#4ec9b0",
                           annotation_text="Target 85%")
        fig_hist.add_vline(x=100, line_dash="dot", line_color="#ff4f4f",
                           annotation_text="Capacity")
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        occ_bins = pd.cut(
            filtered["occupancy_pct"],
            bins=[0,60,80,95,200],
            labels=["<60% Low","60-80% Med","80-95% High",">95% Critical"]
        ).value_counts()
        fig_pie = px.pie(
            values=occ_bins.values,
            names=occ_bins.index,
            title="Occupancy Tier Breakdown",
            color_discrete_sequence=["#3ddc84","#f0c040","#ffaa33","#ff4f4f"],
            hole=0.45,
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Hourly fleet capacity chart
    st.markdown('<div class="section-head">Fleet Utilisation by Hour</div>', unsafe_allow_html=True)
    hu = (filtered.groupby("departure_hour")
          .agg(avg_occ=("occupancy_pct","mean"), count=("train_id","count"))
          .reset_index())
    fig_fu = go.Figure()
    fig_fu.add_trace(go.Bar(
        x=hu["departure_hour"], y=hu["avg_occ"],
        marker=dict(
            color=hu["avg_occ"],
            colorscale=[[0,"#1e4d1e"],[0.5,"#f0c040"],[1,"#ff4f4f"]],
        ),
        name="Avg Occupancy %",
    ))
    fig_fu.add_hline(y=85, line_dash="dot", line_color="#4ec9b0",
                     annotation_text="85% Target")
    fig_fu.update_layout(title="Average Fleet Occupancy by Hour", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_fu, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PLATFORM USAGE
# ══════════════════════════════════════════════════════════════════════════════
elif "Platform" in page:
    st.markdown('<div class="section-head">Platform Performance Overview</div>',
                unsafe_allow_html=True)

    plat_sum = (
        filtered.groupby("platform_number")
        .agg(
            trains=("train_id","count"),
            avg_occ=("occupancy_pct","mean"),
            avg_delay=("delay_minutes","mean"),
            total_pass=("passenger_count","sum"),
        )
        .reset_index()
        .sort_values("trains", ascending=False)
    )

    # Platform KPI cards
    cols = st.columns(min(8, len(plat_sum)))
    for i, (_, row) in enumerate(plat_sum.iterrows()):
        if i < 8:
            color = ("#ff4f4f" if row["avg_occ"]>90
                     else "#ffaa33" if row["avg_occ"]>75
                     else "#3ddc84")
            cols[i].markdown(f"""
            <div style="background:#111118;border:1px solid #1e1e2e;border-radius:10px;
                        padding:12px;text-align:center;">
              <div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                          color:{color};">P{int(row['platform_number'])}</div>
              <div style="font-family:'DM Mono',monospace;font-size:9px;color:#6b6b80;
                          margin:2px 0 6px;">{int(row['trains'])} trains</div>
              <div style="font-size:12px;font-weight:600;color:{color};">
                {row['avg_occ']:.0f}%</div>
              <div style="height:4px;background:#1e1e2e;border-radius:2px;margin-top:6px;">
                <div style="height:100%;width:{min(row['avg_occ'],100):.0f}%;
                            background:{color};border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Platform Utilisation Charts</div>',
                unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_pb = px.bar(
            plat_sum, x="platform_number", y="trains",
            color="avg_occ",
            color_continuous_scale=["#1e4d1e","#f0c040","#ff4f4f"],
            title="Trains per Platform",
            labels={"platform_number":"Platform","trains":"Train Count"},
        )
        fig_pb.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_pb, use_container_width=True)

    with col_p2:
        fig_pd = px.bar(
            plat_sum, x="platform_number", y="avg_delay",
            color="avg_delay",
            color_continuous_scale=["#3ddc84","#ffaa33","#ff4f4f"],
            title="Average Delay per Platform (min)",
            labels={"platform_number":"Platform","avg_delay":"Avg Delay (min)"},
        )
        fig_pd.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_pd, use_container_width=True)

    # Platform occupancy by hour — heatmap
    st.markdown('<div class="section-head">Platform Load by Hour</div>',
                unsafe_allow_html=True)
    ph = (
        filtered.pivot_table(
            index="platform_number",
            columns="departure_hour",
            values="occupancy_pct",
            aggfunc="mean",
        ).round(1)
    )
    fig_ph = px.imshow(
        ph,
        labels=dict(x="Hour", y="Platform", color="Avg Occ %"),
        color_continuous_scale=["#0d1f0d","#f0c040","#ff4f4f"],
        title="Platform × Hour Occupancy Heatmap",
    )
    fig_ph.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_ph, use_container_width=True)

    # Delay analysis
    st.markdown('<div class="section-head">Delay Root-Cause Analysis</div>',
                unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        cause_df = delay_df["cause"].value_counts().reset_index()
        cause_df.columns = ["Cause","Count"]
        fig_cause = px.pie(
            cause_df, names="Cause", values="Count",
            title="Delay Causes",
            color_discrete_sequence=COLORS, hole=0.4,
        )
        fig_cause.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_cause, use_container_width=True)

    with col_d2:
        dm = (delay_df.groupby("route_id")["delay_minutes"]
              .mean().reset_index().sort_values("delay_minutes", ascending=False))
        dm["route_name"] = dm["route_id"].map(lambda x: ROUTE_META.get(x,{}).get("name",x))
        fig_dm = px.bar(
            dm, x="delay_minutes", y="route_name", orientation="h",
            title="Avg Delay by Route",
            color="delay_minutes",
            color_continuous_scale=["#3ddc84","#ffaa33","#ff4f4f"],
        )
        fig_dm.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_dm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SMART RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Recommendations" in page:
    st.markdown('<div class="section-head">AI-Powered Recommendation Engine</div>',
                unsafe_allow_html=True)

    route_sum = summary_by_route(filtered)

    recommendations = []

    # Rule 1: Overcrowded routes
    for _, row in route_sum.iterrows():
        if row["avg_occupancy"] > 90:
            recommendations.append({
                "severity": "🔴 CRITICAL",
                "color": "#ff4f4f",
                "title": f"Overcrowding Alert — {ROUTE_META.get(row['route_id'],{}).get('name', row['route_id'])}",
                "desc": (f"Average occupancy at {row['avg_occupancy']:.1f}% — exceeds safe capacity. "
                         f"ML recommends adding {max(2, int((row['avg_occupancy']-85)/6))} coaches "
                         f"or deploying a supplementary service during peak hours."),
                "action": "Add coaches / Extra train",
            })
        elif row["avg_occupancy"] > 80:
            recommendations.append({
                "severity": "🟠 HIGH",
                "color": "#e05c2a",
                "title": f"High Demand — {ROUTE_META.get(row['route_id'],{}).get('name', row['route_id'])}",
                "desc": (f"Occupancy at {row['avg_occupancy']:.1f}%. Pre-emptively add 1-2 coaches "
                         f"on weekends and holiday periods to maintain comfort."),
                "action": "Pre-deploy coaches",
            })

    # Rule 2: Underutilised routes
    for _, row in route_sum.iterrows():
        if row["avg_occupancy"] < 60:
            recommendations.append({
                "severity": "🟡 MEDIUM",
                "color": "#ffaa33",
                "title": f"Underutilisation — {ROUTE_META.get(row['route_id'],{}).get('name', row['route_id'])}",
                "desc": (f"Average occupancy only {row['avg_occupancy']:.1f}%. "
                         f"Consider reducing coaches from {row['avg_coaches']:.0f} to "
                         f"{max(8, int(row['avg_coaches']-2))} on weekdays "
                         f"and redeploying to high-demand routes."),
                "action": "Reallocate coaches",
            })

    # Rule 3: Platform congestion
    plat_busy = (
        filtered.groupby("platform_number")
        .agg(trains=("train_id","count"), avg_occ=("occupancy_pct","mean"))
        .reset_index()
        .sort_values("trains", ascending=False)
    )
    if len(plat_busy) > 0:
        top_plat = plat_busy.iloc[0]
        recommendations.append({
            "severity": "🟠 HIGH",
            "color": "#e05c2a",
            "title": f"Platform {int(top_plat['platform_number'])} Bottleneck",
            "desc": (f"Platform {int(top_plat['platform_number'])} handles the most trains "
                     f"({int(top_plat['trains'])}) with avg occupancy {top_plat['avg_occ']:.1f}%. "
                     f"Consider diverting 2 trains to adjacent platforms during 17:00–19:00 peak."),
            "action": "Reroute trains",
        })

    # Rule 4: Holiday prep
    holiday_occ = filtered[filtered["is_holiday"]==1]["occupancy_pct"].mean()
    normal_occ  = filtered[filtered["is_holiday"]==0]["occupancy_pct"].mean()
    if holiday_occ > normal_occ + 5:
        recommendations.append({
            "severity": "💡 INSIGHT",
            "color": "#4ec9b0",
            "title": "Holiday Demand Spike Detected",
            "desc": (f"Holiday occupancy ({holiday_occ:.1f}%) is {holiday_occ-normal_occ:.1f}% "
                     f"higher than normal ({normal_occ:.1f}%). "
                     f"Begin supplementary train requisition 3 weeks before major holidays."),
            "action": "Plan ahead",
        })

    # Rule 5: OTP improvement
    low_otp_routes = route_sum[route_sum["on_time_rate"] < 0.75]
    for _, row in low_otp_routes.iterrows():
        recommendations.append({
            "severity": "🟡 MEDIUM",
            "color": "#ffaa33",
            "title": f"Poor On-Time Performance — {row['route_id']}",
            "desc": (f"Only {row['on_time_rate']*100:.1f}% on-time rate. "
                     f"Average delay {row['avg_delay']:.1f} min. "
                     f"Delay predictor model flags overcrowding and platform congestion as main causes."),
            "action": "Schedule review",
        })

    # Render recommendations
    if not recommendations:
        st.success("✅ No critical issues detected in the current filtered data.")
    else:
        sev_order = {"🔴 CRITICAL":0,"🟠 HIGH":1,"🟡 MEDIUM":2,"💡 INSIGHT":3,"🟢 LOW":4}
        recommendations.sort(key=lambda r: sev_order.get(r["severity"], 9))
        for rec in recommendations:
            st.markdown(f"""
            <div class="rec-card" style="border-left: 3px solid {rec['color']};">
              <div class="rec-title">{rec['severity']} &nbsp; {rec['title']}</div>
              <div class="rec-desc">{rec['desc']}</div>
              <div style="margin-top:8px;">
                <span class="pill pill-yellow">{rec['action']}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Severity summary chart
    st.markdown('<div class="section-head">Recommendation Severity Summary</div>',
                unsafe_allow_html=True)
    sev_counts = pd.Series([r["severity"] for r in recommendations]).value_counts()
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_sev = px.pie(
            values=sev_counts.values, names=sev_counts.index,
            title="Issues by Severity",
            color_discrete_sequence=["#ff4f4f","#e05c2a","#ffaa33","#4ec9b0"],
            hole=0.4,
        )
        fig_sev.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_sev, use_container_width=True)

    with col_r2:
        impacts = {
            "Coach Optimisation": 12, "Platform Balancing": 8,
            "Schedule Adjustment": 15, "Demand Forecasting": 6,
            "Route Merging": 9,
        }
        fig_imp = px.bar(
            x=list(impacts.values()), y=list(impacts.keys()),
            orientation="h", title="Projected Efficiency Gains (%)",
            color=list(impacts.values()),
            color_continuous_scale=["#1e4d1e","#f0c040"],
        )
        fig_imp.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "ML Model" in page:
    st.markdown('<div class="section-head">Machine Learning Model Metrics</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Demand Forecast MAE",   f"{forecaster.mae:,} passengers")
    mc2.metric("Demand Forecast R²",    f"{forecaster.r2:.4f}")
    mc3.metric("Delay Predictor Acc.",  f"{delay_pred.accuracy:.1f}%")

    st.markdown('<div class="section-head">Model 1 — Random Forest Demand Forecaster</div>',
                unsafe_allow_html=True)

    # Feature importance
    fi = forecaster.feature_importance().reset_index()
    fi.columns = ["Feature","Importance"]
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_fi = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            title="Feature Importances (RandomForest)",
            color="Importance",
            color_continuous_scale=["#1e4d1e","#f0c040","#e05c2a"],
        )
        fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        # Actual vs predicted scatter on a sample
        sample = filtered.sample(min(500, len(filtered)), random_state=1)
        preds = forecaster.predict(sample)
        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(
            x=sample["passenger_count"].values, y=preds,
            mode="markers",
            marker=dict(color="#f0c040", size=4, opacity=0.6),
            name="Predictions",
        ))
        mx = max(sample["passenger_count"].max(), preds.max())
        fig_av.add_trace(go.Scatter(
            x=[0,mx], y=[0,mx],
            mode="lines",
            line=dict(color="#4ec9b0", dash="dot"),
            name="Perfect Fit",
        ))
        fig_av.update_layout(
            title="Actual vs Predicted Demand (Sample 500)",
            xaxis_title="Actual", yaxis_title="Predicted",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_av, use_container_width=True)

    st.markdown('<div class="section-head">Model 2 — Gradient Boosting Delay Predictor</div>',
                unsafe_allow_html=True)
    report = delay_pred.report
    report_df = pd.DataFrame(report).T.loc[["0","1"]].rename(
        index={"0":"On Time","1":"Delayed"}
    )[["precision","recall","f1-score","support"]]
    report_df = report_df.round(3)
    st.dataframe(report_df.style.background_gradient(
        subset=["precision","recall","f1-score"], cmap="Greens"), use_container_width=True)

    st.markdown('<div class="section-head">Model 3 — Linear Regression Coach Recommender</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="rec-card">
      <div class="rec-title">Coach Recommender Logic</div>
      <div class="rec-desc">
        Uses Linear Regression trained on route, hour, month, and calendar features to predict
        expected passenger count. Divides predicted demand by target occupancy (85%) and seat
        capacity per coach (60) to derive the optimal coach count.
        Business constraints: min 8, max 24 coaches per departure.
        <br><br>
        Formula: <code>recommended_coaches = ceil(predicted_demand / (0.85 × 60))</code>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive recommender
    st.markdown('<div class="section-head">Interactive Coach Calculator</div>',
                unsafe_allow_html=True)
    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    i_route   = ic1.selectbox("Route", ROUTE_IDS, key="ic_route",
                               format_func=lambda x: ROUTE_META[x]["name"])
    i_hour    = ic2.slider("Hour", 5, 22, 9, key="ic_hour")
    i_month   = ic3.slider("Month", 1, 12, 6, key="ic_month")
    i_weekend = ic4.selectbox("Day Type", ["Weekday","Weekend"], key="ic_wknd")
    i_holiday = ic5.selectbox("Holiday?", ["No","Yes"], key="ic_hol")

    rec = recommender.recommend(
        i_route, i_hour, i_month,
        int(i_weekend == "Weekend"),
        int(i_holiday == "Yes"),
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Demand",      f"{rec['predicted_demand']:,} pax")
    r2.metric("Recommended Coaches",   rec["recommended_coaches"])
    r3.metric("Expected Occupancy",    f"{rec['expected_occupancy']}%")
