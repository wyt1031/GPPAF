# Process comparison records

`manuscript_reported_figure7.json` contains the Fig. 7 / Section 2.3 reference values for RSM, SVR, RF, cVAE, Affine-cINN and TC-DSFNet. These values are isolated from executable training/benchmark code.

The v1.4.5 release evaluation fixes **M=256** for cVAE, Affine-cINN and TC-DSFNet. The final English manuscript defines a common Best-of-M protocol and uses M as the candidate-count symbol. The JSON stores the release M value beside the manuscript-reported method errors and 24.2% arithmetic for traceability.

Executable comparison code is in `scripts/benchmark_process.py` and `scripts/benchmark_process_cv.py`. The article source-data archive should retain the full six-method per-sample/per-fold outputs used for the publication comparison.
