# Manuscript figure reproduction map

| Manuscript result | Executable source | Fixed input / record | Evidence location |
|---|---|---|---|
| Fig. 7 TC-DSFNet + five comparisons | `scripts/cross_validate.py`, `scripts/benchmark_process_cv.py` | process CSVs + grouped folds | archived v1.4.4 TC verification artifacts in `reference_results/cross_validation/`; fresh v1.4.5 folds can be regenerated; benchmark driver exports full six-method outputs |
| Fig. 10 PA-HGTS + five comparisons | `scripts/benchmark_paths.py` | `data/path_validation_100.json` + PA-HGTS checkpoint/config supplied to the benchmark driver | six-method per-instance, summary and statistical outputs |
| PA-LCCS | `gppaf/smoothing.py` | representative path + safe region | repository examples/tests + article coordinates |
| HELS-DP | `gppaf/layers.py` | per-layer component paths/states | repository examples/tests + article coordinates |
| FEA | `gppaf/evaluation.py` | exported stress arrays + mesh/area metadata | article source-data archive |
| Point clouds | `gppaf/evaluation.py` | raw/minimally processed clouds + crop/registration metadata | article source-data archive |

## Fig. 7

Manuscript reference values are isolated in `reference_results/process_comparison/manuscript_reported_figure7.json`; they are not imported by training or benchmark code. The five-fold TC-DSFNet artifacts under `reference_results/cross_validation/` are explicitly labelled as v1.4.4 implementation outputs and retained for traceability; they are not presented as outputs of the corrected v1.4.5 Eq. (6) implementation.

```bash
python scripts/cross_validate.py --epochs 100 --candidates 256 --output runs/cross_validation
python scripts/benchmark_process_cv.py --cv-run runs/cross_validation \
  --epochs 100 --candidates 256 --output runs/process_comparison
```

The release evaluation fixes M=256. The six methods' per-fold and per-sample outputs can be archived together with the final figure tables for long-term verification.

## Fig. 10

The fixed benchmark defaults are in `configs/fig10_release_protocol.json`. The packaged 100-case evaluation set can be used directly:

```bash
python scripts/benchmark_paths.py --instances data/path_validation_100.json \
  --checkpoint PA_HGTS_CHECKPOINT.pt \
  --beam 16 --lambda-cost 1.0 --lambda-bound 0.1 \
  --failure-penalty 1000 --local-cost-weight 1.0 \
  --local-rounds 2 --local-top-edges 8 \
  --aco-ants 6 --aco-iterations 5 --aco-evaporation 0.2 \
  --seed 2026 --output runs/fig10
```

Archive the aligned raw, summary and statistical tables together with the 100-case input and the checkpoint/configuration used for the reported run.
