# Article source-data checklist

Keep the Git repository focused on executable research code. For immutable article archiving, retain the numerical records underlying publication figures/tables together with the code release.

Required/strongly recommended records:

1. **Fig. 7** — grouped fold IDs; six-method per-sample/per-fold Best-of-M errors; generative-method interval-coverage outputs; M; seeds; final checkpoints/configs; final aggregate table.
2. **Fig. 10** — 100-instance JSON (`data/path_validation_100.json`); PA-HGTS checkpoint/config; NN/ACO/G-beam/P-beam/R-GLS/PA-HGTS per-instance rows; feasibility flags; objective components; runtime; failure penalty; seeds; Wilcoxon/Holm outputs.
3. **PA-LCCS / HELS-DP** — representative input paths/components and exported smoothed/connected coordinates used in the manuscript figures.
4. **FEA** — model/input description; material; boundary/loading conditions; mesh/element description; exported stress-field values behind the panels and metrics. Large solver-native databases may be archived separately.
5. **3D point clouds** — raw or minimally processed `.ply`, `.pcd` or `.xyz`; registration/cropping metadata; boundary masks; global normalization values; 0.18 outlier threshold; tabulated values behind the point-cloud figures.
6. **Automation workflow** — supplementary operation video plus representative algorithm input/output and robot-program export files.

Recommended archived Fig. 10 file names: `fig10_instances_100.json`, `fig10_pa_hgts_model.pt`, `fig10_protocol.json`, `pa_hgts_benchmark_raw.csv`, `pa_hgts_benchmark_summary.csv`, `pa_hgts_statistical_tests.csv`.

Recommended Fig. 7 file names: `fig7_protocol.json`, `fig7_fold_assignments.json`, `fig7_method_per_sample.csv`, `fig7_method_per_fold.csv`, `fig7_summary.csv`.
