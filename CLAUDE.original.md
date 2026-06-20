# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev              # install deps (Python >=3.12, uv-managed)
uv run pytest                    # run all tests
uv run pytest tests/test_ds2lsom.py::test_name   # run single test
uv run pytest --cov --cov-branch --cov-report=xml  # with coverage (matches CI)
uv run ruff check .              # lint
```

CI (`.github/workflows/ci.yml`) runs the test matrix on ubuntu/windows/macos via `uv sync --extra dev` + `uv run pytest --cov`.

## Architecture

Single-algorithm package: `ds2l_som/ds2lsom.py` implements `DS2LSOM`, a scikit-learn-compatible clusterer (`ClusterMixin`, `BaseEstimator`) based on:

- Cabanes & Bennani, 2008 — *A Local Density-based Simultaneous Two-level Algorithm for Topographic Clustering*
- Cabanes, Bennani & Fresneau, 2012 — *Enriched topological learning for cluster detection and visualization* (doi: 10.1016/j.neunet.2012.02.019)

`fit()` pipeline, in order:

1. **`_train_quantizer`** — reduce X to `n_prototypes_` prototypes via `method="som"` (`dbgsom.SomVQ`) or `"kmeans"` (`sklearn.KMeans`). Stores `weights_`.
2. **`_get_dist_matrix`** — prototype × sample distance matrix (`dist_matrix_`).
3. **`_enrich_prototypes`** — attaches density (`_estimate_density`, Gaussian KDE with `sigma` auto-derived in `_calculate_sigma` if not given), local variability (`_estimate_local_variability`), and neighborhood co-assignment counts (`_estimate_neighborhood_values`, i.e. `nbr_values_` = how often prototypes i/j are each other's two closest BMUs).
4. **`_get_edges`** — keeps prototype pairs with `nbr_values_ >= threshold` as graph edges.
5. **`_create_graph`** — builds a `networkx.Graph` of prototypes with `density`/`variability` node attributes.
6. **`_initial_clustering`** — per connected component, repeated gradient-ascent relabeling toward the locally densest neighbor (iterated `diameter(subgraph)` times).
7. **`_final_clustering`** — iteratively merges micro-clusters across an edge when both endpoints' densities exceed a harmonic-mean threshold of their cluster max-densities, until convergence (`_merge_micro_clusters` overwrites the lower-density cluster's label).
8. `predict()` assigns each sample to its nearest prototype's graph label, then remaps sparse labels to sequential `0..n_clusters-1` (`-1` = unassigned, i.e. prototype not in `graph_`).

Everything lives in this one class; there's no separate module boundary between SOM training, density estimation, graph construction, and clustering — when changing one stage, check the others since they share `self.dist_matrix_`, `self.weights_`, `self.graph_` etc. as implicit state threaded through the pipeline.

Key external dependency: `dbgsom.SomVQ` (the actual SOM trainer/vector quantizer) — not implemented in this repo.

## Tests

- `tests/conftest.py` provides `X_blobs` (sklearn `make_blobs`, 3 centers) and `fitted_model` fixtures.
- `tests/test_ds2lsom.py` — algorithm-specific behavior.
- `tests/test_sklearn_compat.py` — scikit-learn estimator API compliance.
