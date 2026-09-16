# Reproducible environment

The packaged v1.4.5 reference outputs were verified under the following CPU environment:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- SciPy 1.17.0
- Shapely 2.1.2
- scikit-learn 1.8.0

`requirements-tested.txt` records the corresponding package versions. `requirements.txt` gives the supported reviewer ranges for Python 3.11+. The saved TC-DSFNet fold `evaluation.json` files also record the runtime environment so the numerical outputs can be tied to the software stack that generated them.

Recommended reviewer setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_release.py
python scripts/verify_claims.py
```

For the closest package-level reproduction of the bundled CPU results, use Python 3.13 and `requirements-tested.txt`.
