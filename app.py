import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

st.set_page_config(page_title="Credit Default Risk Scorer", page_icon="🏦", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@st.cache_resource
def load_artifacts():
    """Load the fitted scoring pipeline once and cache it across reruns."""
    cluster_scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.joblib"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    meta = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    explainer = shap.TreeExplainer(model)
    return cluster_scaler, kmeans, model, meta, explainer


cluster_scaler, kmeans, model, meta, explainer = load_artifacts()
features = meta["features"]
cluster_features = meta["cluster_features"]
THRESHOLD = meta["threshold"]
