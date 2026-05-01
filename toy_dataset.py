"""
Toy validation pipeline for the research question:
'How can we predict future GHG emissions of countries from their historical sector-level emission data?'

The methodology :
- K-Means clustering on country profile (preprocessing)
- PCA for dimensionality reduction
- OLS regression: global vs. per-cluster vs. cluster-as-feature
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score,
    mean_squared_error, r2_score
)

np.random.seed(42)

# ============================================================
# 1. Generate toy dataset
# ============================================================
N_SAMPLES = 200          # ~194 countries in Climate Watch
N_FEATURES = 5           # 5 IPCC sectors
N_CENTERS = 4            # 4 archetypes
SECTOR_NAMES = ['Energy', 'Industrial', 'Agriculture', 'Waste', 'LULUCF']

X_easy, lab_easy = make_blobs(
    n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS,
    cluster_std=1.0, center_box=(-5, 5), random_state=42
)
X_hard, lab_hard = make_blobs(
    n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS,
    cluster_std=2.5, center_box=(-5, 5), random_state=42
)

# ============================================================
# 2. Synthesize regression targets (future emissions at horizon +10y)
# ============================================================
def synthesize_y(X, labels, seed=42):
    rng = np.random.default_rng(seed)
    k = len(np.unique(labels))
    alphas = rng.uniform(20, 100, size=k)
    betas = rng.normal(0, 2.0, size=(k, X.shape[1]))
    y = np.zeros(X.shape[0])
    for i, c in enumerate(labels):
        y[i] = alphas[c] + X[i] @ betas[c]
    y += rng.normal(0, 2.0, size=X.shape[0])
    return y

y_easy = synthesize_y(X_easy, lab_easy)
y_hard = synthesize_y(X_hard, lab_hard)

# ============================================================
# 3. Standardize
# ============================================================
X_easy_s = StandardScaler().fit_transform(X_easy)
X_hard_s = StandardScaler().fit_transform(X_hard)

# ============================================================
# 4. 2D visualization via PCA
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, X_s, lab, title in [
    (axes[0], X_easy_s, lab_easy, 'Easy: cluster_std=1.0'),
    (axes[1], X_hard_s, lab_hard, 'Hard: cluster_std=2.5')
]:
    proj = PCA(n_components=2).fit_transform(X_s)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap='viridis', s=25, edgecolor='k', linewidth=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label='True cluster')
plt.tight_layout()
plt.savefig('01_clusters_2d.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Scan k for K-Means
# ============================================================
def scan_kmeans(X_s, lab_true, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_s)
        lp = km.labels_
        rows.append({
            'k': k,
            'inertia': km.inertia_,
            'silhouette': silhouette_score(X_s, lp),
            'davies_bouldin': davies_bouldin_score(X_s, lp),
            'calinski_harabasz': calinski_harabasz_score(X_s, lp),
            'ARI': adjusted_rand_score(lab_true, lp),
            'AMI': adjusted_mutual_info_score(lab_true, lp)
        })
    return pd.DataFrame(rows)

res_easy = scan_kmeans(X_easy_s, lab_easy)
res_hard = scan_kmeans(X_hard_s, lab_hard)

print('\n=== Metric scan, easy toy (cluster_std=1.0) ===')
print(res_easy.to_string(index=False))
print('\n=== Metric scan, hard toy (cluster_std=2.5) ===')
print(res_hard.to_string(index=False))

# Plot metrics vs k
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
to_plot = ['inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'ARI', 'AMI']
for i, m in enumerate(to_plot):
    ax = axes[i // 3, i % 3]
    ax.plot(res_easy['k'], res_easy[m], 'o-', label='cluster_std=1.0')
    ax.plot(res_hard['k'], res_hard[m], 's-', label='cluster_std=2.5')
    ax.axvline(N_CENTERS, color='r', linestyle='--', alpha=0.5, label=f'planted k={N_CENTERS}')
    ax.set_xlabel('k')
    ax.set_ylabel(m)
    ax.set_title(m)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_metrics_vs_k.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================
# 6. PCA explained variance
# ============================================================
pca_full = PCA().fit(X_easy_s)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, N_FEATURES + 1), pca_full.explained_variance_ratio_, alpha=0.7)
ax[0].set_xlabel('PC')
ax[0].set_ylabel('Explained variance ratio')
ax[0].set_title('Explained variance per PC')
ax[1].plot(range(1, N_FEATURES + 1), np.cumsum(pca_full.explained_variance_ratio_), 'o-')
ax[1].axhline(0.9, color='r', linestyle='--', label='90% threshold')
ax[1].set_xlabel('# PCs kept')
ax[1].set_ylabel('Cumulative variance')
ax[1].set_title('Cumulative explained variance')
ax[1].legend()
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_pca_variance.png', dpi=100, bbox_inches='tight')
plt.close()

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_pc_90 = int(np.argmax(cumvar >= 0.9) + 1)
print(f'\nPCs to retain 90% variance (easy toy): {n_pc_90}')

# ============================================================
# 7. Regression: global vs per-cluster vs cluster-as-feature
# ============================================================
def evaluate_regressions(X_s, y, lab_true, k=N_CENTERS, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_train = int(0.7 * len(y))
    tr, te = idx[:n_train], idx[n_train:]
    Xtr, Xte, ytr, yte = X_s[tr], X_s[te], y[tr], y[te]

    # 1. Global OLS
    yp1 = LinearRegression().fit(Xtr, ytr).predict(Xte)

    # 2. Per-cluster OLS using K-Means assignments
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xtr)
    ltr, lte = km.labels_, km.predict(Xte)
    yp2 = np.zeros_like(yte)
    for c in range(k):
        mtr = ltr == c
        mte = lte == c
        if mtr.sum() < 2:
            yp2[mte] = ytr.mean()
            continue
        m = LinearRegression().fit(Xtr[mtr], ytr[mtr])
        if mte.any():
            yp2[mte] = m.predict(Xte[mte])

    # 3. Cluster-as-feature
    Xtr_aug = np.hstack([Xtr, np.eye(k)[ltr]])
    Xte_aug = np.hstack([Xte, np.eye(k)[lte]])
    yp3 = LinearRegression().fit(Xtr_aug, ytr).predict(Xte_aug)

    return pd.DataFrame([
        {'strategy': 'Global OLS', 'RMSE': np.sqrt(mean_squared_error(yte, yp1)), 'R2': r2_score(yte, yp1)},
        {'strategy': 'Per-cluster OLS', 'RMSE': np.sqrt(mean_squared_error(yte, yp2)), 'R2': r2_score(yte, yp2)},
        {'strategy': 'Cluster-as-feature OLS', 'RMSE': np.sqrt(mean_squared_error(yte, yp3)), 'R2': r2_score(yte, yp3)}
    ]), (yte, yp1, yp2, yp3)

reg_easy, preds_easy = evaluate_regressions(X_easy_s, y_easy, lab_easy)
reg_hard, preds_hard = evaluate_regressions(X_hard_s, y_hard, lab_hard)

print('\n=== Regression comparison, easy toy ===')
print(reg_easy.to_string(index=False))
print('\n=== Regression comparison, hard toy ===')
print(reg_hard.to_string(index=False))

# Parity plots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (preds, title) in enumerate([(preds_easy, 'Easy'), (preds_hard, 'Hard')]):
    yte, yp1, yp2, yp3 = preds
    for col, (yp, name) in enumerate([(yp1, 'Global'), (yp2, 'Per-cluster'), (yp3, 'Cluster-as-feature')]):
        ax = axes[row, col]
        ax.scatter(yte, yp, alpha=0.6, s=20)
        lim = [min(yte.min(), yp.min()), max(yte.max(), yp.max())]
        ax.plot(lim, lim, 'r-', alpha=0.7)
        ax.set_xlabel('True y')
        ax.set_ylabel('Predicted y')
        ax.set_title(f'{title} | {name}')
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_parity.png', dpi=100, bbox_inches='tight')
plt.close()

print('\nFiles saved to current directory:')
print('  01_clusters_2d.png')
print('  02_metrics_vs_k.png')
print('  03_pca_variance.png')
print('  04_parity.png')