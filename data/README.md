# GPPAF process data

`raw/` contains the four source process tables supplied with the research package; `manifest.json` records row counts and SHA-256 hashes. `processed/` contains the normalized tables read by the training programs.

- `processed/single.csv`: 200 single-layer rows.
- `processed/multi.csv`: 250 multilayer mean-response rows.
- `processed/layers.csv`: 1,050 layerwise rows = 105 process conditions × 10 layers, with all five process variables, layer, width/height and row lineage.
- `processed/layer_condition_metadata.csv`: one record for each of the 105 full five-parameter layerwise conditions.

The raw v12 layerwise table stores factor/layer/X/geometry columns. `layer_settings.json` records the full-parameter map applied for layer supervision and the evidence basis for that map. `processed/layer_settings_provenance.json` stores the exact map used to build the processed table. Every width, height, layer index and varied X value remains unchanged from `raw/layer_partial.csv` and can be verified by source row.

The common single-factor base setting is 80 A, 26 cm/min, 3 Hz and 3 mm. The v12 current/speed/frequency/amplitude layerwise plots use 60 s interpass cooling; the cooling-time series varies the cooling-time factor while retaining the common base setting for the other four variables.

## Rebuild

```bash
python scripts/prepare_data.py --source /path/to/source_csv_directory --output data
```

A different settings JSON may be supplied using `--layer-settings`; provenance is written to `processed/layer_settings_provenance.json`.

Units: current A; travel speed cm/min; frequency Hz; amplitude mm; cooling s; width/height mm.
