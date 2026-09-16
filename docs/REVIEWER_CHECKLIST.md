# Reviewer checklist

## A. Integrity and environment

- `python -m pytest -q` executes numerical/unit coverage for spline inversion, grouped data isolation, layer supervision, process/path baselines, PA-HGTS training/checkpoint selection, PA-LCCS and HELS-DP.
- `python scripts/verify_release.py` checks source hashes, 105-condition layer metadata, row lineage, Eq. (2), packaged fold environments, the 100-case validation input, smoothing certificates and HELS-DP backtracking.
- `requirements-tested.txt` and the fold `evaluation.json` files identify the software stack used for the bundled CPU results.

## B. TC-DSFNet

- Eq. (2) is `gamma*c + beta`; identity initialization is `gamma=1`, `beta=0`.
- Single-layer/multilayer parameter spaces are four/five dimensional.
- RQS forward/inverse/Jacobian behavior is numerically tested.
- Grouping occurs before augmentation; held-out groups do not enter augmentation, trend screening, calibration or model selection.
- `layers.csv` contains 105 complete five-parameter conditions × 10 layers; `layer_condition_metadata.csv` provides one auditable record per condition; every geometry/X value has source-row lineage.
- Every packaged fold stores splits, augmentation parents, trend signs, calibration, checkpoint, history and per-sample evaluation.
- RSM, SVR, RF, cVAE, Affine-cINN and TC-DSFNet share fold test rows in `benchmark_process_cv.py`.

## C. PA-HGTS

- Heterogeneous node/edge descriptors and relation-aware Transformer blocks are implemented.
- Candidate scoring uses previous/current/candidate state and process-aware increments.
- Hard masks enforce revisit, geometric, nonadjacent-crossing and residual-connectivity constraints.
- Beam scoring contains accumulated-objective and MST lower-bound terms.
- Eq. (35) weights, Eq. (36) `gamma`, Eq. (37) `omega_plus`, local-search depth and ACO budget are externally configurable.
- Failures remain in the success-rate denominator and receive the declared failure penalty.
- `data/path_validation_100.json` supplies the fixed 100-case evaluation set spanning cross-section families, problem sizes and grid phases.
- `reference_results/pa_hgts_training_reference/model.pt` is the compact training-reference checkpoint used by the bundled eight-case execution record. Its 20-instance training input, 8-instance disjoint validation input, copied inputs, manifest, histories, runtime record and selected epoch are packaged in the same directory. The 100-case evaluation set is kept as a separate benchmark input.

## D. PA-LCCS

- `epsilon_L` is explicit.
- Window sizes depend on turn geometry, bead width, segment length, boundary clearance and neighboring windows.
- Quintic Bézier replacement is screened by curvature, deviation, feasible-domain and intersection conditions.
- Rejected replacements retain the original segment and the final path receives a global feasibility check.

## E. HELS-DP

- Closed components generate cut/direction states; open components retain direction states.
- Same-layer labels retain connector geometry for later intersection checks.
- Interlayer edges combine distance, direction and local heat costs under geometric feasibility.
- Predecessors are stored and the final path is produced by backtracking.

## F. Article evidence chain

For long-term article archiving, retain the full Fig. 7 six-method outputs, the 100-case path benchmark input/config/checkpoint and 100×6 results, FEA numerical outputs, point-cloud data, and representative software input/output and robot-program export. See `docs/ARTICLE_SOURCE_DATA.md`.
