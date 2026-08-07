import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Page Configuration (Mobile Responsive Wide Layout)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Churn Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Mobile Responsiveness & Polish
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2E3440;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Pipeline Model & SHAP Explainer
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    pipeline = joblib.load('xgb_churn_pipeline.pkl')
    
    # Preprocessor aur XGBoost Classifier extract kar rahe hain SHAP ke liye
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    return pipeline, preprocessor, explainer, model

try:
    pipeline, preprocessor, explainer, model = load_assets()
except Exception as e:
    st.error(f"❌ Error loading model assets: {str(e)}")

# ---------------------------------------------------------
# 3. Header Section
# ---------------------------------------------------------
st.title("🎯 Customer Churn Risk & Intelligence Center")
st.caption("Enterprise AI-Powered Predictive Analytics with SHAP Model Explainability")
st.markdown("---")

# ---------------------------------------------------------
# 4. Sidebar Controls (Responsive Form Inputs)
# ---------------------------------------------------------
st.sidebar.header("👤 Customer Demographics")

age = st.sidebar.slider("Age", 18, 80, 35)
gender = st.sidebar.selectbox("Gender", ["female", "male", "other"])

st.sidebar.header("💳 Subscription & Billing")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 10)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 85.0, step=5.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 15000.0, round(tenure * monthly_charges, 2))

contract = st.sidebar.selectbox("Contract Type", ["month-to-month", "one year", "two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["electronic check", "mailed check", "bank transfer", "credit card"])

# Dataframe Preparation
input_df = pd.DataFrame([{
    "Age": age,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Gender": gender,
    "Contract": contract,
    "PaymentMethod": payment_method
}])

# ---------------------------------------------------------
# 5. Profile Overview Cards (Responsive Grid)
# ---------------------------------------------------------
st.subheader("📋 Active Customer Snapshot")

m1, m2, m3, m4 = st.columns([1, 1, 1, 1])

m1.metric("Tenure Length", f"{tenure} Mo")
m2.metric("Monthly Billing", f"${monthly_charges:.2f}")
m3.metric("Lifetime Value", f"${total_charges:.2f}")
m4.metric("Contract Type", contract.title())

st.markdown("---")

# ---------------------------------------------------------
# 6. Prediction Engine & SHAP Explanation
# ---------------------------------------------------------
if st.button("🚀 Run Comprehensive Risk Assessment", type="primary", use_container_width=True):
    
    # 1. Pipeline Prediction
    churn_proba = float(pipeline.predict_proba(input_df)[0][1])
    is_churn = int(churn_proba > 0.5)
    proba_pct = round(churn_proba * 100, 2)
    
    st.subheader("📊 Executive Summary")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Gauge Chart for Risk Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba_pct,
            number={'suffix': "%"},
            title={'text': "Calculated Churn Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4B4B" if is_churn else "#00CC96"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(255, 170, 0, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.write("#### ")
        if is_churn == 1:
            st.error(f"### 🔴 High Churn Risk Identified ({proba_pct}%)")
            st.warning("⚠️ **Retention Alert:** Customer shows high vulnerability factors corresponding to historical churn behavior.")
            st.info("💡 **Recommended Action:** Offer an immediate 15% promotional discount upon upgrading to a 1-Year or 2-Year contract.")
        else:
            st.success(f"### 🟢 Healthy Customer Profile ({proba_pct}% Risk)")
            st.write("✅ **Status:** Low probability of cancellation in the current billing cycle.")
            st.info("💡 **Recommended Action:** Eligible for premium cross-sell services or loyalty reward enrollment.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 7. SHAP Explainability Section (Updated Feature Names)
    # ---------------------------------------------------------
    st.subheader("🧠 Model Transparency & Feature Impact (SHAP Values)")
    st.write("This section shows **why** the model made this prediction for this specific customer.")

    try:
        # 1. Feature names extract karein preprocessor se
        raw_feature_names = preprocessor.get_feature_names_out()

        # 2. Preprocessed data ko DataFrame mein convert karein
        transformed_data = preprocessor.transform(input_df)
        
        # 3. Prefixes ('cat__', 'num__', etc.) remove karke clean names banayein
        clean_names = [col.split('__')[-1] for col in raw_feature_names]
        
        processed_df = pd.DataFrame(transformed_data, columns=clean_names)

        # 4. SHAP Values calculate karein
        shap_values = explainer(processed_df)
        
        # 5. Plot SHAP Waterfall Chart
        fig_shap, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        st.pyplot(fig_shap, clear_figure=True)
        
    except Exception as e:
        st.error(f"❌ Could not render SHAP plot: {str(e)}")

    # ---------------------------------------------------------
    # 8. Interactive Feature Comparison Chart
    # ---------------------------------------------------------
    st.write("---")
    st.subheader("📈 Financial vs. Tenure Position Analysis")
    
    comparison_data = pd.DataFrame({
        "Metric": ["Monthly Charges ($)", "Tenure (Months)"],
        "Customer Value": [monthly_charges, tenure],
        "Benchmark Average": [65.0, 32.0]
    })
    
    fig_bar = px.bar(
        comparison_data, 
        x="Metric", 
        y=["Customer Value", "Benchmark Average"],
        barmode="group",
        title="Customer Metrics vs. System Benchmarks",
        color_discrete_sequence=["#FF4B4B", "#1F77B4"]
    )
    fig_bar.update_layout(height=350)
    st.plotly_chart(fig_bar, use_container_width=True)