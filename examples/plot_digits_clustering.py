"""
Clustering the Digits Dataset
==============================

This example demonstrates how to use :class:`ds2l_som.DS2LSOM` on the
scikit-learn ``load_digits`` dataset.  DS2L-SOM discovers clusters without
being told the number of clusters in advance.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ds2l_som import DS2LSOM

# %%
# Load and preprocess data
# ------------------------
digits = load_digits()
X, y_true = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Fit DS2L-SOM
# ------------
model = DS2LSOM(n_prototypes=200, threshold=10, verbose=True)
labels = model.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}")

# %%
# Visualise with PCA
# ------------------
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

unique_labels = np.unique(labels)
cmap = plt.cm.get_cmap("tab20", len(unique_labels))

for ax, (lab_array, title) in zip(
    axes,
    [(labels, f"DS2L-SOM ({n_clusters} clusters)"), (y_true, "Ground truth (10 classes)")],
):
    for i, lab in enumerate(np.unique(lab_array)):
        mask = lab_array == lab
        color = "lightgray" if lab == -1 else cmap(i)
        label_str = "Noise" if lab == -1 else str(lab)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], s=10, label=label_str)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

plt.tight_layout()
plt.show()
