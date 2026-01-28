import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    "This system uses **K-Means Clustering** to group customers based on "
    "their purchasing behavior and similarities."
)

# --------------------------------------------------
# Load Dataset (SAFE PATH – FIXED)
# --------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "Wholesale customers data.csv"
)

if not os.path.exists(DATA_PATH):
    st.error(
        "❌ Dataset not found.\n\n"
        "Please ensure **'Wholesale customers data.csv'** is in the same folder as **app.py**."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Identify Numerical Columns
# --------------------------------------------------
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# --------------------------------------------------
# Sidebar – Clustering Controls
# --------------------------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1 (Numerical)",
    numerical_cols,
    index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2 (Numerical)",
    numerical_cols,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=4
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    value=42,
    step=1
)

run_clustering = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# Validation
# --------------------------------------------------
if feature_1 == feature_2:
    st.warning("⚠️ Please select two different features.")
    st.stop()

# --------------------------------------------------
# Run Clustering
# --------------------------------------------------
if run_clustering:

    # ----------------------------------------------
    # Data Preparation
    # ----------------------------------------------
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------------------
    # K-Means Model
    # ----------------------------------------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state
    )

    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # ----------------------------------------------
    # Visualization Section
    # ----------------------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(k):
        ax.scatter(
            X_scaled[df["Cluster"] == i, 0],
            X_scaled[df["Cluster"] == i, 1],
            label=f"Cluster {i}"
        )

    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=250,
        c="black",
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Segmentation using K-Means")
    ax.legend()

    st.pyplot(fig)

    # ----------------------------------------------
    # Cluster Summary Section
    # ----------------------------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Customers=("Cluster", "count"),
            Feature_1_Avg=(feature_1, "mean"),
            Feature_2_Avg=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    # ----------------------------------------------
    # Business Interpretation Section
    # ----------------------------------------------
    st.subheader("💡 Business Interpretation")

    avg_f1 = summary["Feature_1_Avg"].mean()
    avg_f2 = summary["Feature_2_Avg"].mean()

    for _, row in summary.iterrows():
        cluster_id = int(row["Cluster"])

        if row["Feature_1_Avg"] > avg_f1 and row["Feature_2_Avg"] > avg_f2:
            description = "High-spending customers across selected categories"
        elif row["Feature_1_Avg"] < avg_f1 and row["Feature_2_Avg"] < avg_f2:
            description = "Budget-conscious customers with lower spending"
        else:
            description = "Moderate spenders with selective purchasing behavior"

        st.markdown(f"🟢 **Cluster {cluster_id}:** {description}")

    # ----------------------------------------------
    # User Guidance / Insight Box
    # ----------------------------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.info("👈 Select features and click **Run Clustering** to generate results.")
