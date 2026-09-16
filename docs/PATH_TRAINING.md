# PA-HGTS training and path-data protocol

The training entry point supports explicit train/validation geometry datasets, deterministic seeds and held-out checkpoint selection. The release includes the training and validation JSON files used to produce the compact training-reference checkpoint at `reference_results/pa_hgts_training_reference/model.pt`, together with copied inputs, dataset manifest, training history and validation history. The smaller `examples/path_training.json` / `examples/path_validation.json` files remain regression fixtures for unit-level checks.

## Packaged checkpoint training

The packaged checkpoint can be regenerated with the exact bundled training/validation inputs:

```bash
python scripts/train_hgts.py \
  --epochs 12 --hidden 32 --seed 2026 --lr 0.001 \
  --instances examples/pa_hgts_training.json \
  --validation-instances examples/pa_hgts_validation.json \
  --failure-penalty 1000 \
  --regret-weight 0.1 --positive-regret-weight 1.0 \
  --output runs/pa_hgts
```

The compact training-reference run uses 20 training instances and 8 disjoint validation instances. Checkpoint selection minimizes validation failure count, then mean penalized objective, with the earliest epoch retained on ties; epoch 8 is selected. The release stores copied inputs, `dataset_manifest.json`, `training_history.json`, `validation_history.json`, `run.json`, the selected `model.pt` and `last_model.pt` under `reference_results/pa_hgts_training_reference/`.

The loader keeps polygon-equivalent geometries and author-supplied `group_id` values in the same split and rejects train/validation overlap. Failed validation paths remain failures and receive the declared finite penalty during checkpoint selection. The regret head uses the independent feasible-neighborhood oracle described in the implementation; `--positive-regret-weight` is Eq. (37) `omega_plus`.

The smaller `examples/path_training.json` / `examples/path_validation.json` files are retained only as regression fixtures used by tests.

## Benchmark data

`data/path_validation_100.json` is the fixed 100-case evaluation set used by the PA-HGTS comparison benchmark. It contains 56–96 nodes per instance, three cross-section families, three scales and 12 grid phases. The same file can be passed directly to `scripts/benchmark_paths.py --instances ...` together with the PA-HGTS checkpoint/configuration used for an evaluation run.

The benchmark never drops failed runs. Each failure remains in the success-rate denominator and receives the configured finite failure penalty in paired objective comparisons.
