# Six-method path benchmark example

This directory is a compact executable benchmark produced by `scripts/benchmark_paths.py` using the packaged PA-HGTS trained checkpoint (`reference_results/pa_hgts_training_reference/model.pt`) and eight fixed graph cases. It verifies the complete six-method benchmark path, failure accounting, objective-component export, per-instance IDs/metadata and paired statistics.

The saved run contains 8 aligned instances × 6 methods; PA-HGTS produces feasible paths for all eight cases in this compact execution record. The full 100-case evaluation input is `data/path_validation_100.json`; `docs/FIGURE_REPRODUCTION.md` documents the corresponding full-benchmark command.
