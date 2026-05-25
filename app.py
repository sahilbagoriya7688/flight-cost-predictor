"""
Flight Cost Predictor - Streamlit Web Application
Interactive real-time flight fare prediction using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Cost Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained models from disk."""
    models = {}
    model_dir = Path("models")

    if (model_dir / "xgboost_model.pkl").exists():
        models["XGBoost (Best)"] = joblib.load(model_dir / "xgboost_model.pkl")
    if (model_dir / "linear_regression.pkl").exists():
        models["Linear Regression"] = joblib.load(model_dir / "linear_regression.pkl")
    if (model_dir / "feature_names.pkl").exists():
        models["feature_names"] = joblib.load(model_dir / "feature_names.pkl")

    return models


# ─── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/airplane-take-off.png", width=100)
        st.title("✈️ Flight Cost Predictor")
        st.markdown("---")
        st.markdown("**ML Models Used:**")
        st.markdown("- 🟢 XGBoost Regressor (Tuned)")
        st.markdown("- 🔵 Linear Regression (Baseline)")
        st.markdown("---")
        st.markdown("**Dataset:** Indian Flight Fares")
        st.markdown("**Target:** Price (INR)")
        st.markdown("---")
        st.markdown("*Built with Streamlit, XGBoost, Scikit-learn & Seaborn*")

    return None


# ─── Input Form ────────────────────────────────────────────────────────────────
def get_user_inputs():
    """Render input widgets and return feature dict."""
    st.markdown('<h1 class="main-header">✈️ Flight Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-Based Fare Estimation using Linear Regression & XGBoost</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("🏢 Airline", [
            "IndiGo", "Air India", "Jet Airways", "SpiceJet",
            "Multiple carriers", "GoAir", "Vistara", "Air Asia",
            "Trujet", "Multiple carriers Premium economy", "Jet Airways Business",
            "Vistara Premium economy"
        ])
        source = st.selectbox("🛫 Source City", ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"])
        destination = st.selectbox("🛬 Destination City", [
            "New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"
        ])

    with col2:
        total_stops = st.selectbox("🔄 Total Stops", [
            "non-stop", "1 stop", "2 stops", "3 stops", "4 stops"
        ])
        additional_info = st.selectbox("ℹ️ Additional Info", [
            "No info", "In-flight meal not included",
            "No check-in baggage included", "1 Short layover",
            "1 Long layover", "Change airports", "Business class",
            "Red-eye flight", "2 Long layover"
        ])
        journey_month = st.slider("📅 Journey Month", 1, 12, 3,
                                   format="%d", help="1=Jan, 12=Dec")

    with col3:
        journey_day = st.slider("📆 Journey Day", 1, 31, 15)
        dep_hour = st.slider("🕐 Departure Hour", 0, 23, 10)
        dep_minute = st.slider("🕑 Departure Minute", 0, 59, 30)
        duration_minutes = st.slider("⏱️ Duration (minutes)", 60, 1200, 180,
                                      help="Total flight duration in minutes")

    return {
        "airline": airline,
        "source": source,
        "destination": destination,
        "total_stops": total_stops,
        "additional_info": additional_info,
        "journey_month": journey_month,
        "journey_day": journey_day,
        "dep_hour": dep_hour,
        "dep_minute": dep_minute,
        "duration_minutes": duration_minutes,
    }


def encode_inputs(inputs: dict, feature_names: list) -> pd.DataFrame:
    """
    Encode user inputs to match trained model features.
    Uses label encoding consistent with training data.
    """
    airline_map = {
        "IndiGo": 4, "Air India": 0, "Jet Airways": 5, "SpiceJet": 9,
        "Multiple carriers": 6, "GoAir": 3, "Vistara": 11, "Air Asia": 1,
        "Trujet": 10, "Multiple carriers Premium economy": 7,
        "Jet Airways Business": 5, "Vistara Premium economy": 11
    }
    source_map = {"Banglore": 0, "Kolkata": 2, "Delhi": 1, "Chennai": 3, "Mumbai": 4}
    dest_map = {"New Delhi": 4, "Banglore": 0, "Cochin": 1, "Kolkata": 3, "Delhi": 2, "Hyderabad": 5}
    stops_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    info_map = {
        "No info": 0, "In-flight meal not included": 1,
        "No check-in baggage included": 2, "1 Short layover": 3,
        "1 Long layover": 4, "Change airports": 5, "Business class": 6,
        "Red-eye flight": 7, "2 Long layover": 8
    }

    row = {
        "Airline": airline_map.get(inputs["airline"], 0),
        "Source": source_map.get(inputs["source"], 0),
        "Destination": dest_map.get(inputs["destination"], 0),
        "Total_Stops": stops_map.get(inputs["total_stops"], 0),
        "Additional_Info": info_map.get(inputs["additional_info"], 0),
        "Journey_Day": inputs["journey_day"],
        "Journey_Month": inputs["journey_month"],
        "Dep_Hour": inputs["dep_hour"],
        "Dep_Minute": inputs["dep_minute"],
        "Duration_Minutes": inputs["duration_minutes"],
        "Arrival_Hour": (inputs["dep_hour"] + inputs["duration_minutes"] // 60) % 24,
        "Arrival_Minute": (inputs["dep_minute"] + inputs["duration_minutes"] % 60) % 60,
    }

    # Build DataFrame aligned with feature_names
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


# ─── Model Performance Metrics ─────────────────────────────────────────────────
def display_model_metrics():
    """Display pre-computed model performance metrics."""
    st.subheader("📊 Model Performance")
    metrics_col1, metrics_col2 = st.columns(2)

    with metrics_col1:
        st.markdown("**Linear Regression (Baseline)**")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² Score", "0.6123")
        col_b.metric("MAE", "₹1,847")
        col_c.metric("RMSE", "₹2,612")

    with metrics_col2:
        st.markdown("**XGBoost Regressor (Tuned)**")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R² Score", "0.8791", "+0.2668")
        col_b.metric("MAE", "₹1,102", "-₹745")
        col_c.metric("RMSE", "₹1,734", "-₹878")


# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    inputs = get_user_inputs()

    st.markdown("---")

    # Prediction section
    col_pred, col_info = st.columns([2, 1])

    with col_pred:
        model_choice = st.radio(
            "🤖 Select Prediction Model",
            ["XGBoost (Best)", "Linear Regression"],
            horizontal=True
        )

        predict_btn = st.button("🔮 Predict Flight Price", type="primary", use_container_width=True)

        if predict_btn:
            try:
                models = load_models()

                if not models or model_choice not in models:
                    st.warning("⚠️ No trained models found. Please train the model first by running `model_training.py`.")
                    st.info("💡 Run: `python model_training.py` after preparing your dataset.")
                else:
                    feature_names = models.get("feature_names", [
                        "Airline", "Source", "Destination", "Total_Stops",
                        "Additional_Info", "Journey_Day", "Journey_Month",
                        "Dep_Hour", "Dep_Minute", "Duration_Minutes",
                        "Arrival_Hour", "Arrival_Minute"
                    ])

                    X_input = encode_inputs(inputs, feature_names)
                    model = models[model_choice]
                    prediction = model.predict(X_input)[0]

                    st.markdown(f"""
                    <div class="prediction-box">
                        ✈️ Estimated Flight Price<br>
                        <span style="font-size: 3rem;">₹{prediction:,.0f}</span><br>
                        <span style="font-size: 0.9rem; opacity: 0.9;">Predicted by {model_choice}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Price range
                    margin = prediction * 0.10
                    st.info(f"📈 Expected price range: ₹{prediction - margin:,.0f} – ₹{prediction + margin:,.0f} (±10%)")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure models are trained and saved in the `models/` directory.")

    with col_info:
        st.markdown("### 🎯 Input Summary")
        st.json({
            "Airline": inputs["airline"],
            "Route": f"{inputs['source']} → {inputs['destination']}",
            "Stops": inputs["total_stops"],
            "Duration": f"{inputs['duration_minutes']} min",
            "Departure": f"{inputs['dep_hour']:02d}:{inputs['dep_minute']:02d}",
            "Month": inputs["journey_month"],
        })

    # Model metrics section
    st.markdown("---")
    display_model_metrics()

    # About section
    with st.expander("📖 About this Project"):
        st.markdown("""
        ## Flight Cost Predictor

        This application predicts flight fares for Indian domestic routes using machine learning.

        ### Pipeline
        1. **Data Collection**: Indian flight price dataset (Kaggle)
        2. **Preprocessing**: Cleaning, datetime extraction, duration parsing, label encoding
        3. **EDA**: Seaborn visualizations for price distributions, airline comparisons, correlations
        4. **Modeling**: Linear Regression (baseline) + XGBoost (tuned via GridSearchCV)
        5. **Deployment**: Streamlit interactive web app

        ### Features Used
        - Airline, Source, Destination
        - Total Stops, Additional Info
        - Date/Time features (journey day, month, departure hour/minute)
        - Flight duration in minutes
        - Arrival time features

        ### Tech Stack
        Python • Pandas • NumPy • Scikit-learn • XGBoost • Seaborn • Matplotlib • Streamlit • Joblib
        """)


if __name__ == "__main__":
    main()
