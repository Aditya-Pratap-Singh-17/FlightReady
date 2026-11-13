# Independent validation reproducibility

This experiment evaluates the frozen FlightReady v2 implementation on UAVVaste. The frozen method is defined by `frozen_methodology.json` and its SHA-256 file hashes. Training-only protocol amendments are recorded separately and do not change FlightReady.

## Provenance

- Dataset: UAVVaste, DOI `10.5281/zenodo.8214061`, CC BY 4.0.
- Expected archive MD5: `1575c32c04bdf944047563e4a1786c2a`.
- Official split: 540 train, 116 validation, 116 test images.
- The official test images and labels remain byte-identical across conditions. See `dataset_audit.json`.
- Pretrained checkpoint SHA-256 values are in `dataset_audit.json` and every model result JSON.

## Environment

The executed environment uses Python 3.13.5, CUDA-enabled PyTorch 2.14.1+cu130, Ultralytics 8.4.197, and an NVIDIA GeForce RTX 4070 Laptop GPU. Exact versions are repeated in each model-result JSON.

## Commands

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_uavvaste_independent.py
.\.venv\Scripts\python.exe scripts\audit_uavvaste_dataset.py
.\.venv\Scripts\python.exe scripts\assess_uavvaste_flightready.py
.\.venv\Scripts\python.exe scripts\run_uavvaste_independent_training.py
.\.venv\Scripts\python.exe scripts\analyze_uavvaste_independent.py
.\.venv\Scripts\python.exe scripts\plot_uavvaste_independent.py
```

The training runner is resumable: it skips result JSON files already completed under the final batch/worker protocol. Validation selects checkpoints; the test split is evaluated only after training. The runner's read-only image verifier prevents the framework from rewriting source JPEGs.

## Scope and exclusions

The wrong-class corruption is not applicable because UAVVaste has one valid category. Adding an undeclared class would test schema validation, not semantic misclassification. Reference-free assessment never reads the clean labels, corruption manifest, expected acquisition counts, or test data. Full-oracle and deterministic 10% trusted-audit modes are explicitly separate.

Large datasets and checkpoints are intentionally excluded from the review ZIP. Every reported number is traceable to the included CSV/JSON result tables; original checkpoints remain under `training_runs` in the working experiment directory.
