# Quick mock-data accuracy experiment — November 1, 2026

The real FlightReady import, split, readiness, isolated training worker and independent evaluation pipeline completed successfully in 113.61 seconds on CPU. No model or data downloads were needed.

| Model | mAP@50 | mAP@50:95 | Recall at confidence 0.25 |
|---|---:|---:|---:|
| Unchanged local baseline | 0.00% | 0.00% | 0.00% |
| Fine-tuned on incomplete labels | 2.25% | 0.45% | 0.00% |
| Fine-tuned on corrected labels | 4.50% | 0.90% | 0.00% |

Corrected-label fine-tuning improved mAP@50 by **4.5 percentage points over the baseline** and **2.25 points over incomplete-label fine-tuning**. All three models made no predictions at confidence 0.25: TP=0, FP=0, FN=20, F1=0, and precision undefined. Thus there was a small improvement in the confidence-swept detection metric, but no practical detection improvement at the selected operating threshold.

## What was controlled

- 32 training drawings, 8 validation drawings, 8 reserved internal test drawings, and 20 separately generated final test drawings. The final test data were generated after training. Each drawing contains one synthetic rectangle called `drone`; these are not real drone photographs.
- The incomplete-label arm omitted 17 of the 32 training boxes using the existing controlled-variation implementation. The corrected arm used all 32 known generator boxes. FlightReady did not discover or repair the omitted boxes automatically.
- Both fine-tuning arms used the same local checkpoint, verified identical effective initialization tensors, seed 42, YOLO11n, 35 epochs, batch 8, resolution 64, SGD learning rate 0.02, and the existing adapter's augmentation defaults.
- Both arms had identical image bytes and validation/test labels. Evaluation enforced hash, perceptual-duplicate and declared group disjointness. All three models used the same final test checksum and evaluation settings: confidence floor 0.001 for AP, confidence 0.25 for operating metrics, matching IoU 0.5 and NMS IoU 0.7.
- This was one fixed-budget run. No subsequent training or threshold adjustment was performed using final test results.

## Interpretation and limits

The baseline is an existing **one-epoch synthetic checkpoint**, not a general-purpose pretrained detector. These low results do not establish a useful detector or a real UAV accuracy benefit. More training may help, but was deliberately not attempted in this short experiment. One seed and 20 images do not establish statistical reliability.

Both readiness reports returned `READY` for the declared requirements, despite the omitted training objects. Empty annotation files represent declared negative images, and syntax/presence checks cannot identify missing objects inside an otherwise valid dataset without independent trusted annotation evidence. This experiment demonstrates execution and an association between known label correction and measured AP in this fixture; it does not demonstrate that the readiness assessment itself automatically improves data or model performance.

## Evidence and reproduction

- [Results JSON](../.runtime/quick-accuracy-20261101-run/results.json)
- [Results CSV](../.runtime/quick-accuracy-20261101-run/results.csv)
- [Generated report](../.runtime/quick-accuracy-20261101-run/report.md)
- Full sources, immutable dataset records, split manifests, assessments, model checkpoints, epoch losses, worker logs and per-image predictions are under `.runtime/quick-accuracy-20261101-run/storage`.
- [Reusable experiment script](../backend/scripts/quick_accuracy_experiment.py)

From the repository root, using an existing trusted workbench baseline model:

```powershell
.\.venv\Scripts\python.exe backend/scripts/quick_accuracy_experiment.py --baseline-store .runtime/smoke-30a9d8c5/storage --baseline-model d3be1da7ff6b4e07af9e31cfa52dc416 --epochs 35 --timeout 360
```

Each run creates a new isolated `.runtime/quick-accuracy-*` directory. The original baseline store is retained. Windows sandbox execution may require permission for the loopback sockets used internally by the API test client.
