"""
MECA-H419 Project — Step 2: K-Means Clustering
==============================================
Loads the standardized country profiles, finds the optimal 
number of clusters (K) using the Silhouette Score, groups 
the countries, and analyzes the distribution of the EU27.

Input : outputs/country_profiles_standardized.csv
        outputs/country_profiles_percentages.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------- paths
OUT_DIR   = Path("outputs")
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

STD_FILE = OUT_DIR / "country_profiles_standardized.csv"
PCT_FILE = OUT_DIR / "country_profiles_percentages.csv"

# ---------------------------------------------------------------- 1. Load Data
print("Loading clustering datasets...")
df_std = pd.read_csv(STD_FILE, sep=';', decimal=',', encoding='utf-8-sig', index_col=["EDGAR Country Code", "Country"])
df_pct = pd.read_csv(PCT_FILE, sep=';', decimal=',', encoding='utf-8-sig', index_col=["EDGAR Country Code", "Country"])

# ---------------------------------------------------------------- 2. Find Optimal K
print("\nCalculating Silhouette Scores to find optimal K...")
X = df_std.values

best_k = 2
best_score = -1
scores = []
K_range = range(2, 11)

for k in K_range:
    # random_state=42 is CRUCIAL so everyone in your group gets the exact same clusters!
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    print(f"  K={k}: Silhouette Score = {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"\n=> Optimal number of clusters chosen: K = {best_k}")

# ---------------------------------------------------------------- 3. Final Clustering
print(f"Running final K-Means with K={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_pct["Cluster"] = kmeans_final.fit_predict(X)

# ---------------------------------------------------------------- 4. Identify EU27 Cluster
# List of the 27 EU member states ISO codes
eu27_codes = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", 
    "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", 
    "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
]

# Filter our dataframe to only look at EU27 countries
eu_countries = df_pct[df_pct.index.get_level_values("EDGAR Country Code").isin(eu27_codes)]

# Count how many EU countries ended up in each cluster
cluster_counts = eu_countries["Cluster"].value_counts()

print(f"\n--- EU27 CLUSTER ANALYSIS ---")
print("Distribution of EU27 countries across clusters:")
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} countries")

# ---------------------------------------------------------------- 5. Cluster Interpretation
# Calculate the average sector percentages for each cluster to understand "who" they are
cluster_profiles = df_pct.groupby("Cluster").mean()

# --- NEW CODE: Print exact proportions for each cluster ---
print("\n--- EXACT SECTOR PROPORTIONS BY CLUSTER ---")
# Optional: Focus only on the clusters containing EU countries (0, 5, 7) to keep the terminal output clean
eu_cluster_ids = cluster_counts.index.sort_values().tolist()

for c_id in eu_cluster_ids:
    print(f"\nCluster {c_id} Average Profile:")
    # Get the row for this cluster and convert to percentages for readability
    profile = cluster_profiles.loc[c_id] * 100 
    for sector, pct in profile.items():
        print(f"  - {sector}: {pct:.1f}%")
# ----------------------------------------------------------

print("\nSaving final cluster assignments to CSV...")
df_pct.to_csv(OUT_DIR / "final_country_clusters.csv", sep=';', decimal=',', encoding='utf-8-sig')

# ---------------------------------------------------------------- 6. Visualization
print("Generating visualizations...")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# Plot 1: The Silhouette Scores (Proof of optimal K)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(K_range, scores, marker='o', linestyle='-', color='#2980b9')
ax.axvline(x=best_k, color='#c0392b', linestyle='--', label=f'Optimal K={best_k}')
ax.set_title("Silhouette Score for K-Means Clustering")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_silhouette_scores.png")
plt.close(fig)

# Plot 2: Cluster Profiles (Stacked Bar Chart)
fig, ax = plt.subplots(figsize=(10, 6))
# Transpose so sectors are stacked, and clusters are on the X-axis
cluster_profiles.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
ax.set_title("Average Emission Profile by Cluster")
ax.set_xlabel("Cluster ID")
ax.set_ylabel("Proportion of Total Emissions")
ax.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')

# Note: The manual annotation has been removed as requested.

fig.tight_layout()
fig.savefig(FIG_DIR / "02_cluster_profiles.png")
plt.close(fig)

