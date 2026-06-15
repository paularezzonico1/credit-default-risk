import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import shap

st.set_page_config(page_title="Credit Default Risk Scorer", page_icon="🏦", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
