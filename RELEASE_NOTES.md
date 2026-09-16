# GPPAF v1.4.5 release notes

This release makes two algorithm-correctness corrections while leaving the remaining GPPAF implementation, datasets and benchmark protocols unchanged.

- **TC-DSFNet Eq. (6):** trend violations are now collected across all retained stable parameter-response relationships and averaged once over the complete set, matching the manuscript normalization by `|C|`. A regression test covers unequal numbers of single- and multilayer constraints.
- **PA-LCCS point handling:** adjacent duplicate removal is separated from the manuscript `epsilon_L` minimum-segment screening threshold. Short but distinct points are no longer removed by deduplication, and the original path endpoints are explicitly preserved. Regression tests cover a short final segment and true adjacent duplicates.
- **Current-code example:** the TC-DSFNet executable training example in `reference_results/examples/` is regenerated with v1.4.5.
- **Archived process artifacts:** the previously packaged five-fold TC-DSFNet checkpoints/results are retained under `reference_results/cross_validation/` with explicit `artifact_implementation_version = 1.4.4` metadata. They are retained for traceability and are not presented as outputs of the v1.4.5 Eq. (6) implementation.
- All PA-HGTS, HELS-DP, process/path input datasets and benchmark protocols are otherwise unchanged.
