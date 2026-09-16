# Manuscript-to-code correspondence

Source: the 31-page submitted `GPPAF_English_draft_NC_final.pdf`, SHA-256 `01c967c44d28fe4c9c48359c31cf25bb57a7638dfeeb7a4905683c7964c64f12`. The supplied LaTeX formula source was used to resolve characters that do not extract correctly from the PDF; relevant formula pages were also visually inspected. Equation numbers below refer to the submitted manuscript. This map distinguishes equations, implementation decisions and executable tests.

| Manuscript | Implementation | Verification / interpretation |
|---|---|---|
| (1), (5): multilayer means and zero-mean deviations | `process.TCDSFNet.predict_layers`; `data.layer_trends`; `scripts/train_process.load_layers` | all 10 layer embeddings are evaluated before centering; zero-mean test; layer-consistency example |
| (2): shared geometry features and domain modulation | `process.features`, `TCDSFNet.conditions` | `[W,H,W/H,WH]`; scalers fitted only to training data; exact `gamma*c+beta` modulation, with identity initialization `gamma=1, beta=0` |
| (3): 4D/5D conditional spline flow | `splines.rational_quadratic`, `SplineCoupling`, `TCDSFNet.transform` | analytic forward/inverse, 6 layers; roundtrip and finite-difference Jacobian tests; each physical coordinate is transformed |
| (4): bounded logit likelihood and weighted NLL | `TCDSFNet.normalize`, `log_prob`, `losses` | physical-window Jacobian correction included; measured/interpolated weights 1/0.3 |
| (6): bootstrap-stable Jacobian signs | `data.bootstrap_signs`, `TCDSFNet.losses` | 1,000 condition bootstrap resamples; 95% sign consistency; gradients through Jacobian penalties |
| (7): stable width/height layer evolution | `data.layer_trends`, `TCDSFNet.losses` | increasing width/decreasing height constraints only enabled when screened; complete layerwise conditions are loaded by default |
| (8): inverse generation cycle and combined losses | `TCDSFNet.generate`, `TCDSFNet.losses` | every branch has training gradients; configurable ablations in training driver |
| (9): temperature-calibrated bounded candidates | `TCDSFNet.generate`, `process.calibrate` | validation-only temperature grid and common random numbers |
| (10): geometry/density/boundary/layer metrics | `TCDSFNet.candidate_metrics` | actual outputs from model; no stored paper values |
| (11): validation quantiles and weighted ranking | `process.calibrate`, `process.design` | 5th/95th validation-candidate quantiles, clipped scoring and validation-selected weight grid |
| (12)–(16): effective region, spacing, offsets, grid/boundary points | `geometry.build_graph` | Shapely polygon-with-holes erosion, half-bead clearance, grid/offset intersections, deduplication |
| (17)–(18): heterogeneous sparse candidate graph | `geometry.PathGraph`, `build_graph` | I/O/H node types and F/B/C relations; full line-domain test, length threshold and directed K-nearest sparsification |
| (19)–(21): selected edges, length, turning, boundary and spacing | `objectives.Objective.metrics` | evaluated from an actual node permutation; no mixed-integer solver is needed to evaluate these variables |
| (22): path-history heat-accumulation term | `Objective.metrics` | nonadjacent segment minimum distances, segment start times, deposition speed, spatial/time decay |
| (23): scaled composite objective | `Objective.__call__` | validates positive scales/nonnegative normalized weights; benchmark records final common objective |
| (24)–(25), MTZ text | `geometry.path_feasible`, `search.feasible_candidates` | a single ordered no-repeat list enforces one visit and excludes subtours constructively; this is not an MILP implementation |
| (26)–(27): node/edge descriptors | `geometry.build_graph` | 8 node and 7 edge descriptors; dimensions/types checked in tests |
| (28)–(29): type/relation attention | `hgts.RelationLayer` | per-type Q/K/V, per-relation K/V/bias, neighbor softmax, residual/normalization/FFN and edge updates |
| (30): static priority and regret heads | `hgts.PAHGTS.encode` | Softplus positive prior and sigmoid regret, used in decoding/local search |
| (31)–(33): second-order process-aware decoding | `PAHGTS.scores` | previous/current/global/remaining features, accumulated five costs, candidate angle/increments, nonnegative process weights and clipped scores |
| (34): hard mask and residual feasibility | `search.residual_ok`, `feasible_candidates` | no early end node, no repeats/intersections, residual in/out degrees and directed reachability; necessary conditions do not guarantee successful completion |
| (35): beam score and lower bound | `search.beam_search`, `objectives.mst_lower_bound` | log probabilities plus accumulated cost and MST bound; directed edges are relaxed only for lower-bound evaluation |
| (36): feasible local reconnection | `search.local_search`, `neighborhood` | fixed-endpoint 2-opt/Or-opt(1–3), explicit gamma-weighted regret/local-cost priority, acceptance only on non-increasing objective |
| (37) and training paragraph | `search.regret_labels`, `reinforce_step`; `scripts/train_hgts.py` | independent exhaustive feasible local oracle, positive-class-weighted BCE regret loss (omega_plus), REINFORCE cost advantage against frozen greedy baseline; failed episodes recorded |
| (38): adaptive corner windows | `smoothing.smooth_path`, `SmoothConfig` | adjacent duplicate removal is separate from the explicit epsilon_L minimum-segment threshold; endpoints are preserved; turn/curvature thresholds; segment, clearance and adjacent-window caps |
| (39): quintic control points | `smoothing.control_points`, `derivatives` | exact endpoint tangent and zero second derivative tests; G² at accepted straight/curve joins |
| (40), Algorithm 1 | `smooth_path`, `curvature_certificate`, `curve_clear_of`, `deviation_upper` | constrained two-parameter SLSQP; conservative analytic bounds, adaptive export, rejection/retention and global fallback |
| (41): component entry/exit states | `layers.entry_exit_states`, `Traversal` | equal-arclength loop cut positions and two directions; closed-loop length/coverage tested |
| (42): within-layer costs/edges | `layers.connection`, `direction_cost`, `local_heat` | whole segment safety/intersections/maximum length; Gaussian spatial and exponential cooling local heat term |
| (43): geometry-labeled subset DP | `layers.within_layer` | keeps connector geometry in labels, no cost-only deletion of geometrically different labels; explicit resource limit |
| (44): interlayer transition | `layers.interlayer_edge` | 3D length, planar direction term, vertical-projection special case, height slab, domain and tilt constraints |
| (45), Algorithm 2 | `layers.connect_layers` | retained-state DP, predecessor storage, backtracking; checked against exhaustive enumeration |
| §2.3, §3.4 comparisons | `baselines.py`, `scripts/benchmark_process.py`, `scripts/benchmark_paths.py`, `evaluation.paired_comparison` | executable comparison baselines; all failures retained; common candidate count and explicit benchmark configuration |
| §4.1, §4.2.3 metrics | `evaluation.stress_metrics`, `point_cloud_metrics` | global normalization and prescribed thresholds; explicit area weights; raw large experimental files are handled outside the code archive |

## Terminology used consistently

| Name | Meaning |
|---|---|
| TC-DSFNet | Trend-Constrained Dual-Domain Spline Flow Network |
| PA-HGTS | Process-Aware Heterogeneous Graph Transformer Search |
| PA-LCCS | Process-Aware Local Curvature-Continuous Smoothing |
| HELS-DP | Hierarchical Endpoint–Layer-State Dynamic Programming |
| `single` / `multi` | 4D single-layer / 5D multilayer process parameter domain |
| `region` / `safe` | effective material region / feasible bead-centerline region |
| `coverage_fraction` | buffered centerline area intersected with material region, divided by material area |
| `search_failed` | search did not retain a complete feasible path; not a global infeasibility proof |
