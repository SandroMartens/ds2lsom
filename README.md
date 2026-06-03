[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# DS2L-SOM

DS2L-SOM is a topological, density-based clustering algorithm. It combines a Self-Organizing Map (SOM) for prototype learning with Gaussian KDE density estimation and gradient ascent to detect clusters — without requiring the number of clusters to be specified in advance.

Based on the papers by Cabanes, Bennani and Fresneau (2008, 2012).

## Usage

```python
from ds2l_som.ds2lsom import DS2LSOM

model = DS2LSOM(n_prototypes=100, threshold=10)
model.fit(X)
labels = model.predict(X)   # sequential integers; -1 = unassigned

# or short:
labels = model.fit_predict(X)
```

DS2LSOM implements the scikit-learn API (`ClusterMixin`, `BaseEstimator`) and is compatible with sklearn pipelines and `GridSearchCV`.

## Parameters

| Parameter      | Default      | Description                                                       |
| -------------- | ------------ | ----------------------------------------------------------------- |
| `n_prototypes` | auto (10·√n) | Maximum number of SOM prototypes                                  |
| `threshold`    | 1            | Minimum shared samples for a prototype edge                       |
| `sigma`        | auto         | Bandwidth for Gaussian KDE density estimation                     |
| `method`       | `"som"`      | Quantizer backend: `"som"` (dbgsom) or `"kmeans"`                 |
| `model_args`   | `None`       | Kwargs passed to the quantizer: `{"init": {...}, "train": {...}}` |
| `verbose`      | `False`      | Print progress                                                    |

Example with custom SOM parameters:

```python
model = DS2LSOM(
    n_prototypes=100,
    threshold=10,
    model_args={"init": {"sigma_end": 1.0, "random_state": 42}},
)
```

## Performance

Evaluated on `load_digits` (1797 samples, 64 features, 10 classes) using pairwise Rand and Jaccard index (as defined in the papers). DS2LSOM does not receive the true number of clusters.

| Algorithm                    | Clusters | Noise | Rand  | Jaccard |
| ---------------------------- | :------: | :---: | :---: | :-----: |
| DS2LSOM                      | 9 (auto) |  37   | 0.912 |  0.461  |
| KMeans _(n=10 given)_        |    10    |   0   | 0.906 |  0.415  |
| Agglomerative _(n=10 given)_ |    10    |   0   | 0.930 |  0.542  |
| HDBSCAN                      | 6 (auto) | 1244  |   —   |    —    |

HDBSCAN metrics are excluded: 69% noise points make the scores not comparable.

## Scalability

Time and memory complexity: **O(n · k)** where k ≈ 10·√n (default heuristic), giving O(n^1.5) overall. The distance matrix (shape k × n) is the main memory bottleneck — practical limit is around **50 000–100 000 samples**.

## Installing

```bash
# Development
git clone https://github.com/SandroMartens/ds2l-som.git
cd ds2l-som
uv sync

# As a dependency in another project
uv add ds2l-som
```

## References

- _A Local Density-based Simultaneous Two-level Algorithm for Topographic Clustering_, Guénaël Cabanes and Younès Bennani, 2008
- _Enriched topological learning for cluster detection and visualization_, Guénaël Cabanes, Younès Bennani and Dominique Fresneau, 2012
