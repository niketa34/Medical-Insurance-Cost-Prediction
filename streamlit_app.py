import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path

# Local helper imports (simple copies for Streamlit)
from src.data import load_data, clean_data, TARGET
from src.features import add_feature_engineering, build_preprocessor
from src.eda import eda_univariate, eda_bivariate, eda_multivariate, eda_outliers_corr

DATA_PATH = "data/insurance.csv"
MODELS_DIR = "models"

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💊", layout="wide")

def load_dataset():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_feature_engineering(df)
    return df

def get_best_model():
    # Priority order (adjust if you want dynamic best pick)
    for name in ["xgb", "rf", "ridge", "lasso", "linear"]:
        p = Path(MODELS_DIR) / f"{name}_model.joblib"
        if p.exists():
            return joblib.load(p), name
    return None, None

def bootstrap_interval(model, X_row, n_boot=200, seed=42):
    # Simple residual-based bootstrap using training-like noise assumption
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X_row)[0]
    noise = rng.normal(0, scale=0.15*base_pred, size=n_boot)  # heuristic 15% std
    sims = base_pred + noise
    lo, hi = np.percentile(sims, [10, 90])
    return float(lo), float(hi)

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Introduction", "EDA analysis", "Predict cost", "About"])

# Introduction
if page == "Introduction":
    st.title("Medical Insurance Cost Prediction")
    st.markdown("""
    This app explores factors influencing medical insurance costs and provides a cost prediction based on user inputs.
    - Cleaned dataset with engineered features
    - Visual EDA using matplotlib and seaborn
    - Multiple regression models trained and tracked with MLflow
    - Interactive cost prediction with optional confidence interval
    """)

    st.subheader("Approach")
    st.markdown("""
    1. Data preprocessing: cleaning, encoding categorical variables, BMI classification, interaction terms (age×smoker).
    2. EDA: univariate, bivariate, multivariate, outliers, correlation heatmap.
    3. Modeling: Linear, Lasso, Ridge, Random Forest, XGBoost — evaluated via RMSE, MAE, R²; experiments logged in MLflow.
    4. App: four pages — intro, EDA insights, prediction form, and about section.
 
    **Technical Stack used for this project:**
    * **Streamlit** (Frontend)
    * **Pandas/NumPy** (Data Processing)
    * **Seaborn/Matplotlib** (Visualization)
    * **Scikit-Learn** (Machine Learning)
                
     """)

# EDA Analysis
elif page == "EDA analysis":
    st.title("Exploratory Data Analysis")
    df = load_dataset()

    st.subheader("Univariate analysis")
    figs = eda_univariate(df)
    for fig in figs:
        st.pyplot(fig)

    st.subheader("Bivariate analysis")
    figs = eda_bivariate(df)
    for fig in figs:
        st.pyplot(fig)

    st.subheader("Multivariate analysis")
    figs = eda_multivariate(df)
    for fig in figs:
        st.pyplot(fig)

    st.subheader("Outliers and correlation")
    figs = eda_outliers_corr(df)
    for fig in figs:
        st.pyplot(fig)

# Predict page
elif page == "Predict cost":
    st.title("Predict Medical Insurance Cost")
    model, model_name = get_best_model()
    if model is None:
        st.warning("No trained model found. Please run train.py to generate model artifacts.")
    else:
        st.success(f"Using trained model: {model_name.upper()}")

        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
            with col2:
                sex = st.selectbox("Sex", ["male", "female"])
                smoker = st.selectbox("Smoker", ["no", "yes"])
            with col3:
                children = st.number_input("Children", min_value=0, max_value=10, value=0)
                region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

            show_ci = st.checkbox("Show confidence interval (approximate)", value=True)
            submitted = st.form_submit_button("Predict")

        if submitted:
            # Build a one-row DataFrame matching training schema
            row = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region
            }])
            row = row.copy()
            # Apply same featurization
            row = add_feature_engineering(row)

            pred = model.predict(row)[0]
            st.metric("Estimated medical insurance cost", f"{pred:,.2f}")

            if show_ci:
                lo, hi = bootstrap_interval(model, row)
                st.caption(f"Approximate 80% interval: {lo:,.2f} — {hi:,.2f} (heuristic)")

            st.divider()
            st.write("Inputs used:")
            st.json({
                "age": age, "sex": sex, "bmi": bmi, "children": children,
                "smoker": smoker, "region": region
            })

# About page
elif page == "About":
    st.title("About this project")
    st.subheader("About the creator")
    st.markdown("""    ### Hello! I'm Nileta Singh
    I am a Data Scientist passionate about building tools that make data accessible to everyone. 
    
    **Connect with me:**
    * [GitHub](https://github.com/niketa34)
    * [LinkedIn](linkedin.com/in/niketa3400)
    """)