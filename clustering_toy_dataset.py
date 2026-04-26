import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
)

# fix all randomness for reproducibility
RNG = 42                                        
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# # Section 1 # # 
# generate a toy dataset with 4 clusters
# the number of clusters is known for this dataset, but we will pretend that it is not
N_TRUE = 4
X, y_true = make_blobs(
    n_samples=600,
    centers=N_TRUE,
    cluster_std=0.80,
    center_box=(-8.0, 8.0),
    random_state=RNG,
)
