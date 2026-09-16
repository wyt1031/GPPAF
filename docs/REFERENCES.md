# Method and software references

- GPPAF submitted manuscript: *GPPAF: Geometry-Driven Process–Path Co-Design and Automation for Wire Arc Additive Manufacturing of Multi-Hole Components*, Xiaoqi Wang, Hao Zhang, Huajun Zhang, Yanling Xu and Mingliang Zhu. The submitted PDF is the implementation specification. No publication status or DOI is inferred.
- Conor Durkan, Artur Bekasov, Iain Murray and George Papamakarios, *Neural Spline Flows*, NeurIPS 2019. [Primary paper](https://arxiv.org/abs/1906.04032). Basis of the analytically invertible rational-quadratic spline kernel. The kernel in this repository was independently implemented.
- Yuming Huang et al., *Learning Based Toolpath Planner on Diverse Graphs for 3D Printing*. [Primary paper](https://arxiv.org/abs/2408.09198), [DOI](https://doi.org/10.1145/3687933). Relevant external method for graph-based toolpath planning and contextual comparison.
- [Nature Portfolio reporting/code availability policy](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards#availability-of-computer-code). The repository follows the corresponding reproducibility practices with runnable code/data, installation instructions, executable examples and recorded outputs.

Dependencies are installed separately and retain their upstream licenses: [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Shapely](https://shapely.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/). Their implementation source is not vendored in this archive.
