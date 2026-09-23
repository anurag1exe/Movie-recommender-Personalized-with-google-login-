# Changelog

All notable bugs found and fixed during the development of the Movie Recommender System will be documented in this file.

## [Initial scaffold] - 2026-09-18
- Initialized project tracking.

## [Bug fixes] - 2026-09-18
- **WDAC Policy Block**: Pandas and scikit-learn DLLs blocked by Windows Defender Application Control. **Fix**: Rewrote the entire data loading and modeling pipeline (CF with SGD, CB with TF-IDF, Evaluation) in pure Python.
- **Numpy Compatibility**: `np.asfarray` was removed in numpy 2.0, causing evaluation to fail. **Fix**: Replaced `np.asfarray(scores)` with `np.asarray(scores, dtype=float)` in `evaluation.py`.
