# Model-result ledger

This directory retains compact JSON records, not model weights. Every final-protocol record includes the training condition, model, seed, schedule, final test metrics, runtime versions, and SHA-256 identifiers for the initial and best checkpoints. Checkpoint files, datasets, generated variants, and training directories are intentionally excluded from Git.

## Final amended plan: 19/19 complete

The resource amendment defines these planned runs:

- YOLO11n clean: seeds 17, 42, and 93 (3)
- YOLO11n combined-40: seeds 17, 42, and 93 (3)
- YOLO11n Gaussian-25, motion-blur-25, missing-annotations-25, box-displacement-25, coverage-40, and repaired: seed 17 (6)
- YOLO11n Gaussian-10, Gaussian-40, missing-annotations-10, and missing-annotations-40 severity checks: seed 17 (4)
- YOLO11n coverage-40 fixed-update control: seed 17 (1)
- YOLOv8n clean and combined-40 transfer checks: seed 17 (2)

Total: **19 completed planned runs**.

## Supplemental observations

The following final-protocol records satisfy the same Gaussian-25 training configuration but predate the final resource-amended plan. They are retained as **two supplemental observations**, not as additional planned runs:

- `yolo11n__gaussian_25__seed42__fixed_epochs__batch32__workers2.json`
- `yolo11n__gaussian_25__seed93__fixed_epochs__batch32__workers2.json`

## Excluded historical records

Files beginning with `pilot__` and `yolo11n__clean__seed17__fixed_epochs.json` are pilot or pre-final feasibility records. They are preserved for audit history but excluded from the 19 planned runs, 21 final-protocol observations, tables, and statistical tests.

Run `python scripts/verify_final_manuscript_results.py` from the repository root to verify the ledger and independently recalculate the reported findings.
