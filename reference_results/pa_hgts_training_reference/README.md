# Packaged PA-HGTS checkpoint

`model.pt` is the PA-HGTS training-reference checkpoint bundled for compact executable path-planning checks and training-pipeline inspection.

Training is performed by `scripts/train_hgts.py` with deterministic seed 2026 and disjoint explicit training/validation geometry files:

- training: `examples/pa_hgts_training.json` — 20 instances;
- validation: `examples/pa_hgts_validation.json` — 8 instances;
- optimizer: Adam, learning rate 0.001;
- schedule: 12 epochs;
- architecture: hidden width 32, 2 attention heads, 2 encoder layers;
- regret-loss coefficient: 0.1;
- positive regret class weight: 1.0;
- checkpoint selection: minimum validation failure count, then minimum mean penalized objective, with the earliest epoch retained on ties;
- selected epoch: 8.

The training and validation data, copied inputs, dataset manifest, training history, validation history, runtime environment and final-epoch checkpoint are stored in this directory. `data/path_validation_100.json` is a separate fixed 100-case evaluation input accepted by the benchmark driver.

Re-run the checkpoint training with:

```bash
python scripts/train_hgts.py \
  --epochs 12 --hidden 32 --seed 2026 --lr 0.001 \
  --instances examples/pa_hgts_training.json \
  --validation-instances examples/pa_hgts_validation.json \
  --failure-penalty 1000 \
  --regret-weight 0.1 --positive-regret-weight 1.0 \
  --output runs/pa_hgts
```
