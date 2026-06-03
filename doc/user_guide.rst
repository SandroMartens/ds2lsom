User Guide
==========

Installation
------------

.. code-block:: bash

   # Development install
   git clone https://github.com/SandroMartens/ds2l-som.git
   cd ds2l-som
   pip install -e .

   # As a dependency
   pip install ds2l-som

Quick Start
-----------

.. code-block:: python

   from ds2l_som import DS2LSOM

   model = DS2LSOM(n_prototypes=100, threshold=10)
   labels = model.fit_predict(X)  # sequential integers; -1 = unassigned

Algorithm Overview
------------------

DS2L-SOM operates in three stages:

1. **Prototype learning** — A Self-Organizing Map (or k-means) quantizes the input
   into a set of representative prototypes.

2. **Density enrichment** — Each prototype is enriched with a local density estimate
   (Gaussian KDE) and a local variability estimate based on its assigned samples.
   Prototypes that share at least ``threshold`` samples are connected by an edge.

3. **Gradient-ascent clustering** — Each prototype is iteratively relabelled toward
   its densest neighbor. Micro-clusters that satisfy a pairwise density criterion
   are merged into final clusters.

The algorithm does **not** require the number of clusters as input.

Parameters
----------

+----------------+--------------+------------------------------------------------------+
| Parameter      | Default      | Description                                          |
+================+==============+======================================================+
| n_prototypes   | 10·√n        | Maximum number of SOM prototypes                     |
+----------------+--------------+------------------------------------------------------+
| threshold      | 1            | Minimum shared samples for a prototype edge          |
+----------------+--------------+------------------------------------------------------+
| sigma          | auto         | Bandwidth for Gaussian KDE density estimation        |
+----------------+--------------+------------------------------------------------------+
| method         | ``"som"``    | Quantizer backend: ``"som"`` (dbgsom) or ``"kmeans"``|
+----------------+--------------+------------------------------------------------------+
| model_args     | None         | Kwargs for the quantizer (``"init"`` / ``"train"``)  |
+----------------+--------------+------------------------------------------------------+
| verbose        | False        | Print progress                                       |
+----------------+--------------+------------------------------------------------------+

Tuning tips
~~~~~~~~~~~

- **threshold**: Higher values produce more clusters. Start with the default (1)
  and increase if you get too few clusters.
- **sigma**: Controls the reach of the density kernel. The default heuristic
  (mean nearest-neighbour distance between prototypes) works well in practice.
- **n_prototypes**: More prototypes give finer resolution but increase runtime.
  The default heuristic 10·√n balances quality and speed.

Example with custom SOM parameters:

.. code-block:: python

   model = DS2LSOM(
       n_prototypes=100,
       threshold=10,
       model_args={"init": {"sigma_end": 1.0, "random_state": 42}},
   )
   labels = model.fit_predict(X)

scikit-learn Compatibility
--------------------------

:class:`DS2LSOM` implements :class:`sklearn.base.ClusterMixin` and
:class:`sklearn.base.BaseEstimator`, so it works with sklearn pipelines and
:class:`sklearn.model_selection.GridSearchCV`:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   pipe = Pipeline([
       ("scaler", StandardScaler()),
       ("cluster", DS2LSOM(threshold=5)),
   ])
   pipe.fit(X)

Performance
-----------

Evaluated on ``load_digits`` (1797 samples, 64 features, 10 classes) using the
pairwise Rand and Jaccard index. DS2L-SOM does **not** receive the true cluster count.

+-------------------------------+----------+-------+-------+---------+
| Algorithm                     | Clusters | Noise | Rand  | Jaccard |
+===============================+==========+=======+=======+=========+
| DS2LSOM                       | 9 (auto) | 37    | 0.912 | 0.461   |
+-------------------------------+----------+-------+-------+---------+
| KMeans *(n=10 given)*         | 10       | 0     | 0.906 | 0.415   |
+-------------------------------+----------+-------+-------+---------+
| Agglomerative *(n=10 given)*  | 10       | 0     | 0.930 | 0.542   |
+-------------------------------+----------+-------+-------+---------+

Scalability
-----------

Time and memory complexity: **O(n · k)** where k ≈ 10·√n, giving **O(n^1.5)**
overall. The distance matrix (shape k × n) is the main memory bottleneck —
practical limit is around 50 000–100 000 samples.

References
----------

- Cabanes, G. & Bennani, Y. (2008). *A Local Density-based Simultaneous Two-level
  Algorithm for Topographic Clustering.*
- Cabanes, G., Bennani, Y. & Fresneau, D. (2012). *Enriched topological learning
  for cluster detection and visualization.*
  `10.1016/j.neunet.2012.02.019 <https://doi.org/10.1016/j.neunet.2012.02.019>`_
