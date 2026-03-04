# 🚄 RailMind — Smart Railway Resource Planning System

> A data-driven Streamlit dashboard for intelligent railway resource allocation, demand forecasting, and operational planning.

---

## 📌 Problem Statement

Manual railway planning leads to inefficient resource usage, overcrowded trains, idle coaches, and cascading delays. **RailMind** uses historical and synthetic operational data combined with machine learning to help planners make smarter decisions.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📊 Dashboard Overview | Live KPIs, hourly/weekly/monthly demand trends, route comparison |
| 📈 Demand Forecast | 14-day ML forecast per route with confidence bands + demand heatmap |
| 🚂 Train & Coach Allocation | Occupancy analysis, ML-powered coach recommendations, gap analysis |
| 🏗️ Platform Usage | Platform load heatmaps, dwell time, delay root-cause analysis |
| 💡 Smart Recommendations | Auto-generated prioritised recommendations from ML + business rules |
| 🔬 ML Model Performance | Feature importance, actual vs predicted scatter, interactive coach calculator |

---

## 🧠 Machine Learning Models

### 1. Demand Forecaster — `RandomForestRegressor`
- **Target**: `passenger_count`
- **Features**: departure_hour, day_of_week, month, is_weekend, is_holiday, distance_km, num_coaches, route_encoded
- **Metric**: MAE, R²

### 2. Delay Predictor — `GradientBoostingClassifier`
- **Target**: Binary — will the train be delayed >5 min?
- **Features**: occupancy_pct, platform_number, hour, route, calendar features
- **Metric**: Accuracy, Precision, Recall, F1

### 3. Coach Recommender — `LinearRegression` + Business Rules
- **Logic**: Predict demand → divide by target occupancy (85%) × seats/coach (60)
- **Formula**: `recommended_coaches = ceil(predicted_demand / (0.85 × 60))`
- **Constraints**: min 8, max 24 coaches

---

## 🗂️ Project Structure

```
railmind/
├── app.py                   # Main Streamlit app (entry point)
├── requirements.txt         # Python dependencies
├── README.md
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   ├── railway_data.csv     # Auto-generated on first run
│   ├── platform_data.csv    # Auto-generated on first run
│   └── delay_data.csv       # Auto-generated on first run
├── models/
│   └── ml_models.py         # DemandForecaster, DelayPredictor, CoachRecommender
└── utils/
    └── helpers.py           # Data loading, caching, preprocessing
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/railmind.git
cd railmind
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

> **Note:** The synthetic dataset (railway_data.csv, platform_data.csv, delay_data.csv) is auto-generated on first run. No manual data download needed.

---

## 📊 Dataset

All data is **100% synthetic** — generated using realistic patterns for Indian Railways:

| Field | Description |
|---|---|
| `train_id` | Unique train identifier |
| `route_id` | Source–Destination code (e.g., MUM-DEL) |
| `date` | Departure date |
| `departure_time` | Scheduled departure time |
| `departure_hour` | Hour of departure (for ML features) |
| `num_coaches` | Number of coaches deployed |
| `passenger_count` | Simulated passenger count |
| `occupancy_pct` | % of seat capacity used |
| `delay_minutes` | Departure delay in minutes |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_holiday` | 1 if Indian public holiday |
| `platform_number` | Platform assigned (1–8) |
| `coach_class` | Primary class (1AC/2AC/3AC/SL/GEN) |

**Routes covered**: Mumbai↔Delhi, Delhi↔Chennai, Bengaluru↔Hyderabad, Kolkata↔Delhi, Mumbai↔Goa, Chennai↔Bengaluru, Delhi↔Jaipur, Hyderabad↔Mumbai

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn (RandomForest, GradientBoosting, LinearRegression) |
| Dashboard UI | Streamlit |
| Charts | Plotly |
| Styling | Custom CSS via `st.markdown` |

---

## 🏆 Evaluation Criteria Mapping

| Criterion | Implementation |
|---|---|
| Problem Understanding | 6 dedicated views covering all problem aspects |
| Innovation | ML-driven dynamic recommendations, interactive coach calculator |
| Quality of Insights | Heatmaps, forecasts with confidence bands, cause analysis |
| Technical Approach | 3 ML models with performance metrics page |
| Ease of Use | Sidebar navigation, global filters, color-coded indicators |
| Presentation | Custom dark theme, KPI cards, gradient tables |

---

## 📝 Assumptions

1. Average seat capacity: 60 per coach (blended across classes)
2. Target occupancy: 85% (standard operational benchmark)
3. Delay threshold: >5 minutes classified as "delayed"
4. Holiday list: 10 major Indian public holidays (2024)
5. Seasonal patterns: Summer (Apr–Jun) +15%, Diwali season (Oct–Nov) +30%
6. Weekend demand multiplier: 1.25×, Holiday multiplier: 1.45×

---

*Submission for Smart Railway Resource Planning Hackathon — Solo Participation*
