# Verification from a fresh GitHub checkout

Verified on 2026-11-06 using commit `d41a522caa36bf0625dafaa3502ff48f720a04fc` cloned from GitHub, a new
Python 3.12.12 virtual environment, and the README installation commands.
All 180 competition test images were processed locally on an NVIDIA GeForce RTX 3060.
The complete `notebooks/pipeline.ipynb` ran with `jupyter nbconvert --execute`,
first with the default P3 configuration and then with `FILSEG_CONFIG=P`.
Every code cell completed in each run; no retraining or parameter tuning was performed.

| Configuration | Images | Instance rows | SHA-256 match | Mean / image | Median / image | Peak GPU allocated | Peak host RSS |
|---|---:|---:|---|---:|---:|---:|---:|
| P3 | 180 | 1,110 | Exact | 17.17 s | 17.16 s | 3.59 GB | 2.58 GB |
| P | 180 | 1,129 | Exact | 8.74 s | 8.72 s | 2.29 GB | 2.54 GB |

Both generated CSV files match the final competition submission files byte for byte:

- **P3:** `6ce0bcfc5afdcc3dc0bd88886287e18017d9407de62e0df850ad0488c6eb467a`
- **P:** `f4fce72fe54f088d9f7f3d433ff9ce7eb93ad52d12fe1108942703314cb76021`

Times are network plus post-processing time per image. Peak memory is measured
through CSV generation (including the notebook's initial walkthrough); GPU figures
are PyTorch allocated memory and host figures are process RSS, using decimal GB.
Notebook display and format-check overhead are excluded from per-image timing.

## Checks

- `python fetch_weights.py`: all 37 bundled model files passed SHA-256 and size checks.
- `python -m pytest`: 46 passed, 1 skipped. The optional parity test needs research
  probability maps, which are unnecessary for inference. Four existing
  pycocotools/NumPy deprecation warnings did not affect the results.
- `python -m pip check`: no broken requirements.
- Notebook submission-format validation completed for both configurations.
- The inference package, configuration files and weights matched the previously
  verified release byte for byte (56 files checked with SHA-256).

## Environment and determinism

- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128,
  CUDA 12.8, cuDNN 91900.
- NumPy 2.5.4, OpenCV 5.0.0,
  SciPy 1.18.1, LightGBM 4.6.0.
- Single CPU thread; seeds fixed to 0; TF32 and cuDNN benchmarking disabled;
  deterministic cuDNN enabled. The notebook sets PyTorch/OpenCV threads, seeds and
  CUDA flags; the README command also fixes OpenMP/BLAS threads and the Python hash seed.
- The Jupyter kernel used the new checkout's virtual environment and imported
  `filseg` from that checkout. The model hub was offline during inference.

Notebook outputs are cleared in Git because execution displays competition images
and local paths. Generated CSVs, timing sidecars and executed notebooks remain local.
No competition images or annotation files are distributed in this repository.
Only documentation was added after the verified commit; inference files are unchanged.
