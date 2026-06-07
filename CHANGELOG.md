# Changelog

## [0.3.0] - 2026-06-07

### Breaking Changes
- Replaced MiniSom with `dbgsom` as SOM backend — different `model_args` structure
- Graph changed from directed to undirected — affects prototype connectivity
- Density function rewritten — results may differ numerically from 0.2.x

### New Features
- `random_state` parameter for reproducible results
- `method="kmeans"` as alternative quantizer backend
- Full scikit-learn API compliance: `ClusterMixin`, `BaseEstimator`, pipeline-compatible

### Performance
- `argpartition` replaces `argsort` for neighbor lookup — faster on large datasets

### Fixes
- Fixed sigma formula
- Fixed convergence bug in SOM training
- Variable names clarified

### Infrastructure
- GitHub Actions CI (Ubuntu, Windows, macOS)
- PyPI publish workflow via Trusted Publisher (tag-triggered)
- Sphinx documentation setup
- Test suite with pytest + coverage
- Modern packaging via `pyproject.toml` + hatchling

### Documentation
- Algorithm comparison notebook
- Graph analysis notebook
- Updated README with performance benchmarks and badges

## [0.2.0-beta] - initial beta release
