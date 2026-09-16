# Engineering-software integration boundary

The research source release exposes the algorithm layer used by the integrated Unity3D workflow. The engineering application provides model import/slicing, visualization, robot kinematics, trajectory planning/collision checking and robot-program export around these algorithm modules.

A reviewer-facing software evidence package should preserve one representative end-to-end hand-off:

1. sliced cross-section / point set + process target;
2. TC-DSFNet selected process parameters;
3. PA-HGTS continuous path coordinates;
4. PA-LCCS smoothed path coordinates;
5. HELS-DP connected multilayer path coordinates;
6. robot trajectory/program export after the engineering application performs kinematic/collision processing.

The supplementary simulation video can demonstrate the UI and workflow. The article source-data archive should include representative machine-readable input/output files so that the numerical research algorithms remain auditable independently of the GUI project.
