

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# LOAD MODELS
# =========================
cls_model = joblib.load("Notebooks/best_classifier.pkl")
reg_model = joblib.load("Notebooks/best_regression_model.pkl")
preprocessor = joblib.load("Notebooks/preprocessor.pkl")
cluster_model = joblib.load("Notebooks/dbscan_model.pkl")
cluster_preprocessor = joblib.load("Notebooks/cluster_preprocessor.pkl")

# Load the raw training data for category display and internal frequency encoding.
raw_train_df = pd.read_csv("Data/train_data - train_data.csv")
feature_df = pd.read_csv("artifacts/feature_engineered_data.csv")
page2_model_freq = raw_train_df["page2_clothing_model"].value_counts()

# Helper for sorted category options.
def sort_group_values(values):
    def _key(v):
        try:
            return (0, int(v))
        except Exception:
            return (1, str(v))
    return sorted(values, key=_key)

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Clickstream ML System", layout="wide")

st.title("🛍️ Clickstream Customer Intelligence System")

st.sidebar.title("Choose Task")
task = st.sidebar.radio(
    "Select Module",
    ["📊 Classification", "💰 Regression", "👥 Clustering"]
)

# =========================
# COMMON INPUT SECTION
# =========================
st.subheader("Enter Customer Behavior Data")

month_options = sorted(feature_df["month"].dropna().unique())
page_options = sorted(feature_df["page"].dropna().unique())
page1_options = sorted(feature_df["page1_main_category"].dropna().unique())
location_options = sorted(feature_df["location"].dropna().unique())
model_photo_options = sorted(feature_df["model_photography"].dropna().unique())
day_grouped_options = sorted(feature_df["day_grouped"].dropna().unique())
country_options = sort_group_values(feature_df["country_grouped"].dropna().unique())
colour_options = sort_group_values(feature_df["colour_grouped"].dropna().unique())

month = st.selectbox("Month", month_options, index=0)
order = st.number_input("Order", min_value=0, value=1, step=1)
page = st.selectbox("Page", page_options, index=0)
page1_main_category = st.selectbox("Page 1 Main Category", page1_options, index=0)
page2_clothing_model = st.selectbox(
    "Page 2 Clothing Model",
    options=sorted(raw_train_df["page2_clothing_model"].dropna().unique())
)
country_grouped = st.selectbox("Country Grouped", country_options, index=0)
colour_grouped = st.selectbox("Colour Grouped", colour_options, index=0)
location = st.selectbox("Location", location_options, index=0)
model_photography = st.selectbox("Model Photography", model_photo_options, index=0)
day_grouped = st.selectbox("Day Grouped", day_grouped_options, index=0)

page2_clothing_model_freq = page2_model_freq.get(page2_clothing_model, 0)

input_data = {
    "month": month,
    "order": order,
    "page1_main_category": page1_main_category,
    "location": location,
    "model_photography": model_photography,
    "page": page,
    "page2_clothing_model_freq": page2_clothing_model_freq,
    "country_grouped": country_grouped,
    "colour_grouped": colour_grouped,
    "day_grouped": day_grouped,
}

input_df = pd.DataFrame([input_data])

# ==========================================================
# 1️⃣ CLASSIFICATION
# ==========================================================
if task == "📊 Classification":

    st.markdown("### 🔮 Conversion Prediction")

    if st.button("Predict Conversion"):
        try:
            transformed_input = preprocessor.transform(input_df)
            pred = cls_model.predict(transformed_input)

            if pred[0] == 1:
                st.success("✅ Customer WILL convert (Purchase Likely)")
            else:
                st.error("❌ Customer will NOT convert")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================================
# 2️⃣ REGRESSION
# ==========================================================
elif task == "💰 Regression":

    st.markdown("### 💵 Revenue Prediction")

    if st.button("Predict Revenue"):
        try:
            transformed_input = preprocessor.transform(input_df)
            pred = reg_model.predict(transformed_input)
            st.success(f"💰 Expected Revenue: ${pred[0]:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================================
# 3️⃣ CLUSTERING
# ==========================================================
elif task == "👥 Clustering":

    st.markdown("### 🧩 Customer Segmentation")

    if st.button("Find Customer Segment"):
        try:
            scaled_input = cluster_preprocessor.transform(input_df)
            if hasattr(scaled_input, "toarray"):
                scaled_input = scaled_input.toarray()

            if hasattr(cluster_model, "predict"):
                cluster = cluster_model.predict(scaled_input)
            else:
                # DBSCAN does not implement .predict().
                # Assign a new point by finding the nearest core sample.
                core_points = cluster_model.components_
                distances = np.linalg.norm(core_points - scaled_input, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]
                if nearest_dist <= cluster_model.eps:
                    cluster = [cluster_model.labels_[cluster_model.core_sample_indices_[nearest_idx]]]
                else:
                    cluster = [-1]

            st.success(f"👤 Customer belongs to Segment: {cluster[0]}")
        except Exception as e:
            st.error(f"Clustering failed: {e}")

# =========================
# OPTIONAL FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 ML System: Classification + Regression + Clustering (Clickstream Analytics)")


#mlfow tracking command

##mlflow ui --backend-store-uri file:///e:/Laptop/Clickstream/mlruns##