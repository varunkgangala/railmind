"""
models/ml_models.py
-------------------
Machine-learning layer for RailMind.

Models:
  1. DemandForecaster   – RandomForestRegressor to predict passenger_count
  2. DelayPredictor     – GradientBoostingClassifier to flag likely delays
  3. CoachRecommender   – Rule + regression hybrid to suggest coach count
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. DEMAND FORECASTER
# ─────────────────────────────────────────────────────────────
class DemandForecaster:
    """
    Predicts passenger_count given route, time, and calendar features.
    Uses RandomForestRegressor for non-linear demand patterns.
    """

    FEATURES = [
        "departure_hour", "day_of_week_num", "month",
        "is_weekend", "is_holiday",
        "distance_km", "num_coaches",
        "route_encoded",
    ]

    def __init__(self):
        self.model   = RandomForestRegressor(
            n_estimators=150, max_depth=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        self.le_route = LabelEncoder()
        self.le_dow   = LabelEncoder()
        self.trained  = False
        self.mae      = None
        self.r2       = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["route_encoded"]   = self.le_route.transform(df["route_id"])
        df["day_of_week_num"] = self.le_dow.transform(df["day_of_week"])
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.le_route.fit(df["route_id"])
        self.le_dow.fit(df["day_of_week"])
        df = self._encode(df)

        X = df[self.FEATURES]
        y = df["passenger_count"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        self.mae = round(mean_absolute_error(y_te, preds), 1)
        self.r2  = round(r2_score(y_te, preds), 4)
        self.trained = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = self._encode(df)
        return self.model.predict(df[self.FEATURES]).astype(int)

    def forecast_route(
        self,
        route_id: str,
        start_date: str,
        days: int = 14,
        coaches: int = 16,
        hour: int = 9,
    ) -> pd.DataFrame:
        """Return a day-by-day demand forecast for a given route."""
        dates = pd.date_range(start_date, periods=days, freq="D")
        rows = []
        for dt in dates:
            rows.append({
                "date":          dt.strftime("%Y-%m-%d"),
                "route_id":      route_id,
                "departure_hour": hour,
                "day_of_week":   dt.strftime("%A"),
                "month":         dt.month,
                "is_weekend":    int(dt.weekday() >= 5),
                "is_holiday":    0,
                "distance_km":   self._route_dist(route_id),
                "num_coaches":   coaches,
            })
        df_in = pd.DataFrame(rows)
        df_in["predicted_demand"] = self.predict(df_in)
        return df_in

    def _route_dist(self, route_id: str) -> int:
        dist_map = {
            "MUM-DEL": 1384, "DEL-CHE": 2180, "BLR-HYD": 570,
            "KOL-DEL": 1453, "MUM-GOA":  590, "CHE-BLR": 362,
            "DEL-JAI":  303, "HYD-MUM":  711,
        }
        return dist_map.get(route_id, 800)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.FEATURES
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────
# 2. DELAY PREDICTOR
# ─────────────────────────────────────────────────────────────
class DelayPredictor:
    """
    Binary classifier: will a train be delayed (>5 min)?
    Uses GradientBoostingClassifier.
    """

    FEATURES = [
        "departure_hour", "day_of_week_num", "month",
        "is_weekend", "is_holiday",
        "occupancy_pct", "num_coaches",
        "platform_number", "route_encoded",
    ]

    def __init__(self):
        self.model    = GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        )
        self.le_route = LabelEncoder()
        self.le_dow   = LabelEncoder()
        self.trained  = False
        self.accuracy = None

    def _encode(self, df):
        df = df.copy()
        df["route_encoded"]   = self.le_route.transform(df["route_id"])
        df["day_of_week_num"] = self.le_dow.transform(df["day_of_week"])
        return df

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["delayed"] = (df["delay_minutes"] > 5).astype(int)
        self.le_route.fit(df["route_id"])
        self.le_dow.fit(df["day_of_week"])
        df = self._encode(df)

        X = df[self.FEATURES]
        y = df["delayed"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        self.accuracy = round(accuracy_score(y_te, preds) * 100, 1)
        self.report   = classification_report(y_te, preds, output_dict=True)
        self.trained  = True
        return self

    def predict_proba_delay(self, df: pd.DataFrame) -> np.ndarray:
        df = self._encode(df)
        return self.model.predict_proba(df[self.FEATURES])[:, 1]


# ─────────────────────────────────────────────────────────────
# 3. COACH RECOMMENDER
# ─────────────────────────────────────────────────────────────
class CoachRecommender:
    """
    Recommends optimal number of coaches per route/time-slot.
    Uses a linear regression calibrated on historical data,
    then applies a business-rule cap.
    """
    SEATS_PER_COACH = 60       # average across classes
    TARGET_OCC_PCT  = 85       # target occupancy

    def __init__(self):
        self.model   = LinearRegression()
        self.le_route = LabelEncoder()
        self.trained  = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        self.le_route.fit(df["route_id"])
        df["route_encoded"] = self.le_route.transform(df["route_id"])
        feats = ["route_encoded", "departure_hour", "month",
                 "is_weekend", "is_holiday"]
        X = df[feats]
        y = df["passenger_count"]
        self.model.fit(X, y)
        self.trained = True
        return self

    def recommend(
        self,
        route_id: str,
        hour: int,
        month: int,
        is_weekend: int = 0,
        is_holiday: int = 0,
    ) -> dict:
        route_enc = self.le_route.transform([route_id])[0]
        X = pd.DataFrame([{
            "route_encoded": route_enc,
            "departure_hour": hour,
            "month":          month,
            "is_weekend":     is_weekend,
            "is_holiday":     is_holiday,
        }])
        pred_demand = max(0, self.model.predict(X)[0])
        required_seats    = pred_demand / (self.TARGET_OCC_PCT / 100)
        recommended_coaches = int(np.ceil(required_seats / self.SEATS_PER_COACH))
        recommended_coaches = max(8, min(recommended_coaches, 24))

        return {
            "route_id":             route_id,
            "hour":                 hour,
            "predicted_demand":     int(pred_demand),
            "recommended_coaches":  recommended_coaches,
            "expected_occupancy":   round(
                pred_demand / (recommended_coaches * self.SEATS_PER_COACH) * 100, 1
            ),
        }

    def bulk_recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recommended_coaches column to a route-summary dataframe."""
        results = []
        for _, row in df.iterrows():
            rec = self.recommend(
                row["route_id"], row["departure_hour"],
                row["month"], row["is_weekend"], row["is_holiday"]
            )
            results.append(rec["recommended_coaches"])
        df = df.copy()
        df["recommended_coaches"] = results
        df["coach_gap"] = df["recommended_coaches"] - df["num_coaches"]
        return df
