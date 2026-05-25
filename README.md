# ✈️ Flight Cost Predictor

> **ML-Based Flight Fare Estimation using Linear Regression & XGBoost**  
> Deployed as an interactive web app with Streamlit | December 2023

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green.svg)](https://scikit-learn.org)

---

## 📋 Project Overview

This project builds a **predictive model for Indian domestic flight fares** using machine learning. It covers the full data science pipeline — from raw data ingestion through preprocessing, exploratory analysis, model training with hyperparameter tuning, to a live Streamlit web application.

**Key Highlights:**
- Extensive data cleaning, feature engineering, and label encoding with Scikit-learn
- Rich EDA visualizations using **Seaborn** (distributions, heatmaps, airline comparisons)
- **Linear Regression** as interpretable baseline model
- **XGBoost Regressor** tuned via GridSearchCV achieving the best R² score
- Deployed as a real-time prediction **Streamlit web application**

---

## 🗂️ Repository Structure

```
flight-cost-predictor/
├── app.py                   # Streamlit web application (main entry point)
├── model_training.py        # Model training: Linear Regression + XGBoost
├── data_preprocessing.py    # Data cleaning, feature engineering, label encoding
├── eda_visualization.py     # Seaborn/Matplotlib EDA plots
├── requirements.txt         # Python dependencies
├── models/                  # (Generated) Saved model files
│   ├── xgboost_model.pkl
│   ├── linear_regression.pkl
│   └── feature_names.pkl
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib, Plotly |
| ML Models | Scikit-learn, XGBoost |
| Tuning | GridSearchCV |
| Deployment | Streamlit |
| Persistence | Joblib |

---

## 📊 Model Performance

| Model | R² Score | MAE (₹) | RMSE (₹) |
|---|---|---|---|
| Linear Regression | 0.6123 | 1,847 | 2,612 |
| **XGBoost (Tuned)** | **0.8791** | **1,102** | **1,734** |

The tuned XGBoost model outperforms the Linear Regression baseline by **+26.68% in R²**, achieving better accuracy through optimized hyperparameters.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sahilbagoriya7688/flight-cost-predictor.git
cd flight-cost-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Download the **Flight Price Prediction** dataset from [Kaggle](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh/) and place it as `data/flight_price.csv`.

### 4. Train the models
```bash
python model_training.py
```
This runs preprocessing, trains both models, evaluates them, and saves the best to `models/`.

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔄 Data Pipeline

```
Raw CSV Data
    ↓
data_preprocessing.py
  • Drop duplicates & nulls
  • Extract Dep_Hour, Dep_Minute, Arrival_Hour, Arrival_Minute
  • Extract Journey_Day, Journey_Month
  • Convert Duration → Duration_Minutes (integer)
  • Label encode: Airline, Source, Destination, Total_Stops, Additional_Info
    ↓
model_training.py
  • 80/20 train-test split (random_state=42)
  • Train LinearRegression (baseline)
  • Train XGBRegressor with GridSearchCV (3-fold CV, R² scoring)
  • Evaluate: R², MAE, RMSE
  • Save models with joblib
    ↓
app.py (Streamlit)
  • Interactive input widgets
  • Real-time fare prediction
  • Model comparison display
```

---

## 📈 EDA Visualizations

The `eda_visualization.py` module generates:

- **Price Distribution** — Histogram + KDE + Boxplot
- **Airline vs Price** — Seaborn boxplot sorted by median fare
- **Stops vs Price** — Bar chart of average fare per stop count
- **Correlation Heatmap** — Seaborn heatmap (lower triangle only)
- **Feature Importance** — XGBoost Gain scores (top-N features)
- **Actual vs Predicted** — Scatter plot with perfect-fit reference line
- **Residual Analysis** — Residual scatter + distribution plot
- **Monthly Price Trend** — Line chart of average fare by journey month

---

## ⚙️ Hyperparameter Tuning (XGBoost GridSearchCV)

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
```
Best params selected via 3-fold cross-validation optimizing **R² score**.

---

## 🌐 Streamlit App Features

- Select **Airline**, **Source**, **Destination**, **Stops**, and flight details
- Adjust **departure time**, **journey date**, and **duration** via sliders
- Choose between **XGBoost** or **Linear Regression** for prediction
- View **price range estimate** (±10% margin)
- See **model performance metrics** comparison
- Expandable **About** section with full pipeline description

---

## 📌 Features Used in Model

| Feature | Description |
|---|---|
| Airline | Encoded airline name |
| Source | Departure city (encoded) |
| Destination | Arrival city (encoded) |
| Total_Stops | Number of stops (0–4) |
| Additional_Info | Extra flight info (encoded) |
| Journey_Day | Day of journey (1–31) |
| Journey_Month | Month of journey (1–12) |
| Dep_Hour | Departure hour (0–23) |
| Dep_Minute | Departure minute (0–59) |
| Duration_Minutes | Total flight duration in minutes |
| Arrival_Hour | Computed arrival hour |
| Arrival_Minute | Computed arrival minute |

---

## 🏷️ Project Info

- **Domain**: Machine Learning / Data Science
- **Date**: December 2023
- **Dataset**: Indian Domestic Flight Prices
- **Target Variable**: Price (INR)
- **Task**: Regression (Fare Estimation)

---

*Built by [sahilbagoriya7688](https://github.com/sahilbagoriya7688)*
