# GPPAF model card

## TC-DSFNet

TC-DSFNet uses a shared geometry encoder with domain-specific affine modulation, separate 4D/5D rational-quadratic spline flows, geometry/cycle losses, bootstrap-stable process-trend constraints and layerwise evolution supervision. Eq. (2) is implemented directly as `gamma * c + beta`; adapter initialization sets `gamma=1` and `beta=0`.

The packaged process tables contain 200 single-layer samples, 250 multilayer mean samples and 1,050 layerwise records. Formal evaluation is grouped by process condition so related settings do not cross train/validation/test partitions.

## PA-HGTS

PA-HGTS uses heterogeneous node/edge representations, relation-aware Transformer layers, priority and regret heads, second-order candidate features, hard feasibility masks, beam search and feasibility-preserving 2-opt/Or-opt reconnection. The bundled compact training-reference checkpoint stores its architecture, deterministic seed, epoch/update counts, optimizer settings, train/validation dataset manifest, selected epoch, selection criterion and runtime environment. The fixed 100-case JSON provides the evaluation set for the six-method path benchmark and spans multiple cross-section families, problem sizes and grid phases.

## PA-LCCS

PA-LCCS performs local quintic-Bézier replacement only at screened corners. Adjacent numerical duplicates are removed independently of the manuscript minimum segment threshold `epsilon_L`; short but distinct segments and the original endpoints are retained. It checks curvature, geometric deviation, feasible-region inclusion, regularity and intersections before accepting a replacement.

## HELS-DP

HELS-DP represents alternative component entry/exit/direction states, stores connection geometry in first-level DP labels and performs second-level interlayer DP with predecessor backtracking and feasibility checks.

## Intended use

The package supports manuscript verification, algorithm study, ablation/comparison experiments and research extension. Engineering deployment still requires machine-specific calibration, robot/controller integration and manufacturing safety validation.
