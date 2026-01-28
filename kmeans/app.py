import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Wholesale Customer Segmentation", layout="wide")
st.title("📦 Wholesale Customer Segmentation using K-Means")
st.write("Business-driven Unsupervised Machine Learning")

# ----------------------------------
# Task 1: Data Exploration
# ----------------------------------
st.header("🔹 Task 1: Data Exploration")

df = pd.read_csv("Wholesale customers data.csv")

st.dataframe(df.head())
st.write("Dataset Shape:", df.shape)

st.markdown("""
**Purchasing behavior columns identified:**
Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen  
Non-spending columns like Channel and Region are ignored.
""")

# ----------------------------------
# Task 2: Feature Selection
# ----------------------------------
st.header("🔹 Task 2: Feature Selection")

features = [
    'Fresh', 'Milk', 'Grocery',
    'Frozen', 'Detergents_Paper', 'Delicassen'
]

X = df[features]

st.write("Selected Features:", features)
st.info("These features directly represent customer purchasing behavior.")

# ----------------------------------
# Task 3: Data Preparation
# ----------------------------------
st.header("🔹 Task 3: Data Preparation")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Data standardized successfully using StandardScaler.")

# ----------------------------------
# Task 4 & 5: Optimal Cluster Identification (ELBOW ONLY)
# ----------------------------------
st.header("🔹 Task 4 & 5: Optimal Cluster Identification")

wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(K, wcss, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("WCSS")

st.pyplot(fig)

st.markdown("""
**Chosen K = 4**

Reason:
- Clear elbow observed at K = 4  
- Balanced number of customer segments  
- Suitable for business interpretation
""")

# ----------------------------------
# Task 6: Cluster Assignment
# ----------------------------------
st.header("🔹 Task 6: Cluster Assignment")

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.dataframe(df.head())

# ----------------------------------
# Task 7: Cluster Visualization
# ----------------------------------
st.header("🔹 Task 7: Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 6))

for i in range(4):
    ax2.scatter(
        X_scaled[df['Cluster'] == i, 0],
        X_scaled[df['Cluster'] == i, 1],
        label=f'Cluster {i}'
    )

ax2.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

ax2.set_xlabel("Fresh (scaled)")
ax2.set_ylabel("Milk (scaled)")
ax2.set_title("Customer Segments")
ax2.legend()

st.pyplot(fig2)

# ----------------------------------
# Task 8: Cluster Profiling
# ----------------------------------
st.header("🔹 Task 8: Cluster Profiling")

profile = df.groupby('Cluster')[features].mean()
st.dataframe(profile)

st.markdown("""
**Cluster Summary:**
- High Grocery & Detergents → Retail Stores  
- High Fresh → Hotels / Restaurants  
- Moderate spending → Cafés  
- Low spending → Occasional buyers
""")

# ----------------------------------
# Task 9: Business Insights
# ----------------------------------
st.header("🔹 Task 9: Business Insights")

st.markdown("""
**Cluster 0:** Bulk discounts & inventory priority  
**Cluster 1:** Fresh product contracts & fast delivery  
**Cluster 2:** Combo offers & flexible pricing  
**Cluster 3:** Promotions & upselling strategies
""")

# ----------------------------------
# Task 10: Stability & Limitations
# ----------------------------------
st.header("🔹 Task 10: Stability & Limitations")

kmeans_alt = KMeans(n_clusters=4, random_state=99)
alt_labels = kmeans_alt.fit_predict(X_scaled)

changed = np.sum(df['Cluster'] != alt_labels)
st.write("Customers changing clusters with different random state:", changed)

st.warning("""
Limitation:
- K-Means requires predefined K  
- Sensitive to outliers  
- Assumes spherical clusters
""")

st.success("✅ Customer segmentation completed successfully!")
