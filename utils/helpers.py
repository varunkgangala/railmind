"""
utils/helpers.py
----------------
Data loading, caching, preprocessing, and shared chart helpers
for the RailMind Streamlit app.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Data Loading ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = os.path.join(DATA_DIR, "railway_data.csv")
    if not os.path.exists(path):
        _generate_and_save()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_platform_data():
    path = os.path.join(DATA_DIR, "platform_data.csv")
    if not os.path.exists(path):
        _generate_and_save()
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_delay_data():
    path = os.path.join(DATA_DIR, "delay_data.csv")
    if not os.path.exists(path):
        _generate_and_save()
    return pd.read_csv(path, parse_dates=["date"])


def _generate_and_save():
    """Auto-generate data if CSVs don't exist."""
    import sys
    sys.path.insert(0, DATA_DIR)
    from generate_data import (
        generate_main_dataset,
        generate_platform_data,
        generate_delay_records,
    )
    main_df  = generate_main_dataset(days=180)
    plat_df  = generate_platform_data(main_df)
    delay_df = generate_delay_records(main_df)
    main_df.to_csv( os.path.join(DATA_DIR, "railway_data.csv"),  index=False)
    plat_df.to_csv( os.path.join(DATA_DIR, "platform_data.csv"), index=False)
    delay_df.to_csv(os.path.join(DATA_DIR, "delay_data.csv"),    index=False)


# ── Model Training ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_trained_models():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.ml_models import DemandForecaster, DelayPredictor, CoachRecommender

    df = load_data()
    forecaster   = DemandForecaster().fit(df)
    delay_pred   = DelayPredictor().fit(df)
    recommender  = CoachRecommender().fit(df)
    return forecaster, delay_pred, recommender


# ── KPI Helpers ───────────────────────────────────────────────
def kpi_card(col, label, value, delta=None, delta_color="normal"):
    col.metric(label=label, value=value, delta=delta,
               delta_color=delta_color)


def occupancy_color(pct: float) -> str:
    if pct >= 95:  return "#ff4f4f"
    if pct >= 80:  return "#ffaa33"
    return "#3ddc84"


def demand_tag(pct: float) -> str:
    if pct >= 95:  return "🔴 CRITICAL"
    if pct >= 80:  return "🟠 HIGH"
    if pct >= 60:  return "🟡 MEDIUM"
    return "🟢 LOW"


# ── Route Metadata ────────────────────────────────────────────
ROUTE_META = {
    "MUM-DEL": {"name": "Mumbai → Delhi",     "dist": 1384},
    "DEL-CHE": {"name": "Delhi → Chennai",    "dist": 2180},
    "BLR-HYD": {"name": "Bengaluru → Hyd.",  "dist":  570},
    "KOL-DEL": {"name": "Kolkata → Delhi",   "dist": 1453},
    "MUM-GOA": {"name": "Mumbai → Goa",       "dist":  590},
    "CHE-BLR": {"name": "Chennai → Bengaluru","dist":  362},
    "DEL-JAI": {"name": "Delhi → Jaipur",    "dist":  303},
    "HYD-MUM": {"name": "Hyderabad → Mumbai","dist":  711},
}

ROUTE_IDS = list(ROUTE_META.keys())


# ── Preprocessing ─────────────────────────────────────────────
def summary_by_route(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("route_id")
        .agg(
            avg_passengers=("passenger_count", "mean"),
            avg_occupancy=("occupancy_pct", "mean"),
            avg_coaches=("num_coaches", "mean"),
            avg_delay=("delay_minutes", "mean"),
            total_trains=("train_id", "count"),
            on_time_rate=("on_time", "mean"),
        )
        .reset_index()
        .round(1)
    )


def hourly_demand(df: pd.DataFrame, route_id: str = None) -> pd.DataFrame:
    if route_id:
        df = df[df["route_id"] == route_id]
    return (
        df.groupby("departure_hour")["passenger_count"]
        .mean()
        .reset_index()
        .rename(columns={"passenger_count": "avg_passengers"})
    )


def weekly_demand(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    agg = (
        df.groupby("day_of_week")["passenger_count"]
        .mean()
        .reindex(order)
        .reset_index()
        .rename(columns={"passenger_count": "avg_passengers"})
    )
    return agg


def monthly_demand(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month")["passenger_count"]
        .mean()
        .reset_index()
        .rename(columns={"passenger_count": "avg_passengers"})
    )


def heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hour × DayOfWeek average passengers pivot."""
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = (
        df.pivot_table(
            index="departure_hour",
            columns="day_of_week",
            values="passenger_count",
            aggfunc="mean",
        )
        .reindex(columns=order)
    )
    return pivot.round(0)
