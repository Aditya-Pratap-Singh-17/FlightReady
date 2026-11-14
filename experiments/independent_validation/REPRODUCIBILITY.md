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

## Completed run ledger

- Planned by the resource-amended protocol: **19** runs.
- Completed planned runs: **19/19**.
- Additional valid observations: **2**, both YOLO11n Gaussian-25 (`seed42` and `seed93`).
- Total final-protocol observations retained: **21**.
- Pilot and pre-final feasibility records remain available for audit but are excluded from the final ledger.

The exact planned tags are stored in `analysis/statistical_analysis.json`. The distinction between planned and supplemental observations is checked independently by `scripts/verify_final_manuscript_results.py` and explained in `model_results/README.md`.

## Recalculate and verify retained results

The following commands are read-only with respect to training and the frozen methodology:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_uavvaste_independent.py
.\.venv\Scripts\python.exe scripts\plot_uavvaste_independent.py
.\.venv\Scripts\python.exe scripts\verify_final_manuscript_results.py
```

The final verifier independently recalculates the 19/21 run accounting, condition means, paired changes, t intervals, Pearson and Spearman statistics, evidence coverage, geometric scorecards, categorical decisions, controlled-defect detection, and conventional baselines from the retained JSON/CSV evidence.

## Frozen-methodology integrity note

The original `frozen_methodology.json` is intentionally unchanged. Three recorded SHA-256 digests match directly. The `backend/readiness_engine.py` entry is an invalid 63-character string with one omitted `e`; inserting that character reproduces the exact CRLF-file digest, while LF normalization is byte-identical to the research-release Git blob. See `../../submission/Frozen_Methodology_Integrity_Audit.md` for the byte-level audit. This is a documented manifest transcription defect, not a methodology change.

## Final review artifacts

- `final_reports/FlightReady_Final_Independent_Validation_Report.md` and `.pdf`
- `../../submission/FlightReady_IEEE_IV_2027_FINAL.pdf`
- `../../submission/FlightReady_IEEE_IV_2027_FINAL.tex`
- `../../submission/Final_Manuscript_Verification.md`
- `../../submission/Frozen_Methodology_Integrity_Audit.md`

## Scope and exclusions

The wrong-class corruption is not applicable because UAVVaste has one valid category. Adding an undeclared class would test schema validation, not semantic misclassification. Reference-free assessment never reads the clean labels, corruption manifest, expected acquisition counts, or test data. Full-oracle and deterministic 10% trusted-audit modes are explicitly separate.

Large datasets and checkpoints are intentionally excluded from Git and review packages. Every reported number is traceable to the compact CSV/JSON result tables; original checkpoints remain under ignored local `training_runs` directories. Public result records use repository-relative paths and retain checkpoint SHA-256 values without publishing retrainable weights.
