import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1️⃣ App Title & Description
# --------------------------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

# --------------------------------------------------
# Load Dataset (Direct)
# --------------------------------------------------
df = pd.read_csv("Wholesale customers data.csv")

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# --------------------------------------------------
# 2️⃣ Input Section (Sidebar – Mandatory)
# --------------------------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1", numerical_cols, index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2", numerical_cols, index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)", min_value=2, max_value=10, value=4
)

random_state = st.sidebar.number_input(
    "Random State (optional)", value=42, step=1
)

run_btn = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# Validation
# --------------------------------------------------
if feature_1 == feature_2:
    st.warning("Please select two different features.")
    st.stop()

# --------------------------------------------------
# 3️⃣ Clustering Control (Run Button)
# --------------------------------------------------
if run_btn:

    # --------------------------------------------------
    # Data Preparation
    # --------------------------------------------------
    X = df[[feature_1, feature_2]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # K-Means Model
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    # --------------------------------------------------
    # 4️⃣ Visualization Section
    # --------------------------------------------------
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
    ax.set_title("Customer Segments using K-Means")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # 5️⃣ Cluster Summary Section
    # --------------------------------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Feature_1_Avg=(feature_1, "mean"),
            Feature_2_Avg=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    # --------------------------------------------------
    # 6️⃣ Business Interpretation Section
    # --------------------------------------------------
    st.subheader("💡 Business Interpretation")

    for _, row in summary.iterrows():
        cluster_id = int(row["Cluster"])

        if row["Feature_1_Avg"] > summary["Feature_1_Avg"].mean():
            spending_type = "High-spending"
        else:
            spending_type = "Lower-spending"

        st.markdown(
            f"🟢 **Cluster {cluster_id}:** {spending_type} customers with similar "
            f"purchasing patterns in selected categories."
        )

    # --------------------------------------------------
    # 7️⃣ User Guidance / Insight Box
    # --------------------------------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.info("👈 Select features and click **Run Clustering** to begin.")
