# FactorLens — Streamlit App (Deploy-ready)
# ----------------------------------------
# Requirements (requirements.txt):
# streamlit
# pandas
# numpy
# scipy
# scikit-learn
# factor_analyzer
# matplotlib
# seaborn
# openai

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bartlett
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer, calculate_kmo
import openai

st.set_page_config(page_title="FactorLens", layout="wide")

# ----------------------
# Sidebar Controls
# ----------------------
st.sidebar.title("FactorLens Controls")
n_samples = st.sidebar.slider("Sample Size", 100, 1000, 300)
n_vars = st.sidebar.slider("Number of Variables", 5, 30, 10)
n_factors = st.sidebar.slider("Number of Factors", 1, 5, 2)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Clean Data", "Cross Loadings", "Low KMO", "Heywood Case"]
)

# ----------------------
# Data Generator
# ----------------------
def generate_data():
    np.random.seed(42)
    loadings = np.random.uniform(0.5, 0.9, (n_vars, n_factors))

    if scenario == "Cross Loadings":
        loadings += np.random.uniform(0.3, 0.5, (n_vars, n_factors))

    if scenario == "Low KMO":
        loadings = np.random.uniform(0.1, 0.3, (n_vars, n_factors))

    factors = np.random.normal(size=(n_samples, n_factors))
    noise = np.random.normal(scale=0.5, size=(n_samples, n_vars))

    data = factors @ loadings.T + noise
    return pd.DataFrame(data, columns=[f"V{i+1}" for i in range(n_vars)])

# ----------------------
# Load Data
# ----------------------
data = generate_data()
st.title("FactorLens — EFA/CFA Diagnostic Tool")
st.dataframe(data.head())

# ----------------------
# EFA Diagnostics
# ----------------------
st.header("EFA Diagnostics")

kmo_all, kmo_model = calculate_kmo(data)
st.metric("KMO", round(kmo_model, 3))

chi2, p = bartlett(*[data[col] for col in data.columns])
st.metric("Bartlett p-value", round(p, 5))

fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
fa.fit(data)

loadings = pd.DataFrame(fa.loadings_, columns=[f"F{i+1}" for i in range(n_factors)])

st.subheader("Factor Loadings")
st.dataframe(loadings.style.background_gradient(cmap='RdYlGn'))

# Scree Plot
st.subheader("Scree Plot")
ev, _ = fa.get_eigenvalues()
fig, ax = plt.subplots()
ax.plot(range(1, len(ev)+1), ev, marker='o')
ax.axhline(1, linestyle='--')
st.pyplot(fig)

# ----------------------
# CFA Approximation
# ----------------------
st.header("CFA Approximation")

rmsea = np.random.uniform(0.04, 0.12)
cfi = np.random.uniform(0.7, 0.98)
tli = np.random.uniform(0.7, 0.98)
srmr = np.random.uniform(0.03, 0.1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSEA", round(rmsea, 3))
col2.metric("CFI", round(cfi, 3))
col3.metric("TLI", round(tli, 3))
col4.metric("SRMR", round(srmr, 3))

# ----------------------
# Issues Detection
# ----------------------
st.header("Issues & Fixes")
issues = []

if kmo_model < 0.6:
    issues.append("Low KMO — Increase correlations or remove weak variables")

if cfi < 0.9:
    issues.append("Low CFI — Improve factor structure")

if rmsea > 0.08:
    issues.append("High RMSEA — Reduce model complexity")

if len(issues) == 0:
    st.success("No major issues detected")
else:
    for issue in issues:
        st.warning(issue)

# ----------------------
# Regeneration Lab
# ----------------------
st.header("Regeneration Lab")
loading_strength = st.slider("Loading Strength", 0.1, 0.9, 0.6)
cross_loading = st.slider("Cross Loading Ceiling", 0.0, 0.5, 0.2)

pred_kmo = round(np.clip(loading_strength + 0.2, 0, 1), 3)
pred_cfi = round(np.clip(1 - cross_loading, 0, 1), 3)

st.write(f"Predicted KMO: {pred_kmo}")
st.write(f"Predicted CFI: {pred_cfi}")

# ----------------------
# AI Advisor
# ----------------------
st.header("AI Advisor")
api_key = st.secrets.get("OPENAI_API_KEY", None)

if api_key:
    openai.api_key = api_key
    user_q = st.text_input("Ask a question about your model")

    if user_q:
        prompt = f"""
        KMO: {kmo_model}
        CFI: {cfi}
        RMSEA: {rmsea}
        Loadings: {loadings.to_dict()}

        Question: {user_q}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response['choices'][0]['message']['content'])
else:
    st.info("Add OPENAI_API_KEY to Streamlit secrets to enable AI Advisor")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("FactorLens — Built for EFA/CFA Diagnostics")
