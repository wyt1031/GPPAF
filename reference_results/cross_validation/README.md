# Archived TC-DSFNet five-fold verification artifacts (v1.4.4 implementation)

This directory contains the grouped five-fold TC-DSFNet verification artifacts generated with the **v1.4.4 implementation**. They are retained for traceability and are not presented as outputs of the v1.4.5 implementation, which corrects manuscript Eq. (6) to average all retained trend relationships in one global mean.

Each fold stores the model checkpoint, configuration, grouped split IDs, augmentation-parent IDs, bootstrap trend signs, calibration, training history, per-sample evaluation results and runtime environment. The archived configuration is 100 epochs, seed 2026, M=256 candidates and 1,000 bootstrap resamples. Aggregate Best-of-M values are read from `summary.json`; `scripts/verify_release.py` checks split isolation, complete held-out coverage, aggregate arithmetic, environment consistency and the explicit artifact implementation version.

Fresh five-fold outputs for the v1.4.5 code can be generated with `scripts/cross_validate.py`. The five archived folds record the same CPU stack: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0, Shapely 2.1.2 and scikit-learn 1.8.0. See `requirements-tested.txt`.

Publication Fig. 7 values are stored separately under `reference_results/process_comparison/` and are not inputs to these runs.
