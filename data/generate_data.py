"""
generate_data.py
----------------
Generates a realistic synthetic railway dataset for RailMind.
Run once to create: railway_data.csv, delay_data.csv, platform_data.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────
ROUTES = [
    {"id": "MUM-DEL", "src": "Mumbai",    "dst": "Delhi",     "dist_km": 1384, "base_demand": 4200},
    {"id": "DEL-CHE", "src": "Delhi",     "dst": "Chennai",   "dist_km": 2180, "base_demand": 3100},
    {"id": "BLR-HYD", "src": "Bengaluru", "dst": "Hyderabad", "dist_km":  570, "base_demand": 2800},
    {"id": "KOL-DEL", "src": "Kolkata",   "dst": "Delhi",     "dist_km": 1453, "base_demand": 3400},
    {"id": "MUM-GOA", "src": "Mumbai",    "dst": "Goa",       "dist_km":  590, "base_demand": 2200},
    {"id": "CHE-BLR", "src": "Chennai",   "dst": "Bengaluru", "dist_km":  362, "base_demand": 3600},
    {"id": "DEL-JAI", "src": "Delhi",     "dst": "Jaipur",    "dist_km":  303, "base_demand": 4800},
    {"id": "HYD-MUM", "src": "Hyderabad", "dst": "Mumbai",    "dist_km":  711, "base_demand": 2600},
]

TRAIN_CLASSES = ["1AC", "2AC", "3AC", "SL", "GEN"]
CLASS_CAPACITY = {"1AC": 18, "2AC": 46, "3AC": 64, "SL": 72, "GEN": 90}

# Indian public holidays (approximate)
HOLIDAYS_2024 = [
    "2024-01-26", "2024-03-25", "2024-04-14", "2024-05-23",
    "2024-08-15", "2024-10-02", "2024-10-12", "2024-10-31",
    "2024-11-15", "2024-12-25",
]

# Peak hour multiplier (hour → multiplier)
HOURLY_PATTERN = {
    5: 0.35, 6: 0.55, 7: 0.85, 8: 1.60, 9: 1.85,
    10: 1.40, 11: 1.10, 12: 1.05, 13: 0.90, 14: 0.95,
    15: 1.15, 16: 1.40, 17: 1.75, 18: 1.90, 19: 1.65,
    20: 1.25, 21: 0.95, 22: 0.65, 23: 0.40,
}


def is_holiday(dt):
    return dt.strftime("%Y-%m-%d") in HOLIDAYS_2024


def demand_for(route_base, dt, coaches, hour):
    multiplier = HOURLY_PATTERN.get(hour, 0.8)
    if dt.weekday() >= 5:           # weekend
        multiplier *= 1.25
    if is_holiday(dt):              # holiday
        multiplier *= 1.45
    # seasonal: summer (Apr-Jun) and Diwali season (Oct-Nov) spike
    if dt.month in [4, 5, 6]:
        multiplier *= 1.15
    if dt.month in [10, 11]:
        multiplier *= 1.30
    base = route_base * multiplier
    noise = np.random.normal(0, base * 0.08)
    raw = max(0, int(base + noise))
    capacity = coaches * 60          # avg 60 seats/coach across classes
    return raw, round(raw / capacity * 100, 1) if capacity > 0 else 0.0


def generate_main_dataset(days=180):
    """Main operational dataset: one row per train-departure."""
    records = []
    start_date = datetime(2024, 1, 1)
    train_counter = 12000

    for day_offset in range(days):
        dt = start_date + timedelta(days=day_offset)
        is_wknd = dt.weekday() >= 5
        is_hol   = is_holiday(dt)

        for route in ROUTES:
            # 3-6 departures per route per day
            n_departures = random.randint(3, 6)
            departure_hours = sorted(random.sample(range(5, 23), n_departures))

            for hour in departure_hours:
                minute = random.choice([0, 15, 30, 45])
                dep_time = dt.replace(hour=hour, minute=minute)

                coaches = random.randint(10, 22)
                primary_class = random.choices(
                    TRAIN_CLASSES, weights=[5, 15, 30, 35, 15]
                )[0]

                demand, occ_pct = demand_for(
                    route["base_demand"], dt, coaches, hour
                )

                platform = random.randint(1, 8)
                delay_min = 0
                if occ_pct > 95:
                    delay_min = random.randint(5, 35)
                elif occ_pct > 80:
                    delay_min = random.choices([0, random.randint(3, 15)],
                                               weights=[70, 30])[0]
                else:
                    delay_min = random.choices([0, random.randint(1, 8)],
                                               weights=[85, 15])[0]

                train_counter += random.randint(1, 5)
                records.append({
                    "train_id":         str(train_counter),
                    "route_id":         route["id"],
                    "source":           route["src"],
                    "destination":      route["dst"],
                    "distance_km":      route["dist_km"],
                    "date":             dt.strftime("%Y-%m-%d"),
                    "departure_time":   dep_time.strftime("%H:%M"),
                    "departure_hour":   hour,
                    "day_of_week":      dt.strftime("%A"),
                    "month":            dt.month,
                    "is_weekend":       int(is_wknd),
                    "is_holiday":       int(is_hol),
                    "num_coaches":      coaches,
                    "coach_class":      primary_class,
                    "platform_number":  platform,
                    "passenger_count":  demand,
                    "seat_capacity":    coaches * 60,
                    "occupancy_pct":    occ_pct,
                    "delay_minutes":    delay_min,
                    "on_time":          int(delay_min == 0),
                })

    df = pd.DataFrame(records)
    return df


def generate_platform_data(main_df):
    """Aggregated platform utilisation per day."""
    plat = (
        main_df.groupby(["date", "platform_number"])
        .agg(
            trains_scheduled=("train_id", "count"),
            avg_occupancy=("occupancy_pct", "mean"),
            avg_delay=("delay_minutes", "mean"),
            total_passengers=("passenger_count", "sum"),
        )
        .reset_index()
    )
    plat["avg_occupancy"] = plat["avg_occupancy"].round(1)
    plat["avg_delay"]     = plat["avg_delay"].round(1)
    return plat


def generate_delay_records(main_df):
    """Delay summary with root-cause tags."""
    causes = ["Track Congestion", "Platform Unavailable",
              "Mechanical Issue", "Crew Delay", "Signal Fault",
              "Weather", "Passenger Boarding"]
    delayed = main_df[main_df["delay_minutes"] > 0].copy()
    delayed["cause"] = [random.choice(causes) for _ in range(len(delayed))]
    return delayed[["train_id", "route_id", "date",
                     "departure_time", "delay_minutes",
                     "occupancy_pct", "platform_number", "cause"]]


if __name__ == "__main__":
    print("Generating synthetic railway dataset …")
    main_df   = generate_main_dataset(days=180)
    plat_df   = generate_platform_data(main_df)
    delay_df  = generate_delay_records(main_df)

    import os
    out = os.path.join(os.path.dirname(__file__))
    main_df.to_csv(  f"{out}/railway_data.csv",  index=False)
    plat_df.to_csv(  f"{out}/platform_data.csv", index=False)
    delay_df.to_csv( f"{out}/delay_data.csv",    index=False)

    print(f"✅  railway_data.csv  — {len(main_df):,} rows")
    print(f"✅  platform_data.csv — {len(plat_df):,} rows")
    print(f"✅  delay_data.csv    — {len(delay_df):,} rows")
