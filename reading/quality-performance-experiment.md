# Does measured dataset quality predict model performance?

## Result — November 1, 2026

**Partly supported for image clarity in this controlled mock experiment; not validated as a general rule or an overall readiness percentage.** Better-clarity training data outperformed severely blurred training data, but the highest-clarity data did not produce the highest target mAP. Different performance metrics also moved in different directions.

This experiment changes image clarity only. Every annotation is correct and identical across the three target training conditions. It does not repeat the earlier incomplete-label/corrected-label experiment.

## Held-out target-domain results

| Model | Training clarity pass fraction | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| General-dataset baseline | Separate broader dataset | 99.02% | 76.57% | 96.00% | 100.00% | 97.96% |
| Fine-tuned: low clarity | 0% | 99.50% | 65.64% | 96.00% | 100.00% | 97.96% |
| Fine-tuned: mixed clarity | 50% | 99.02% | **86.41%** | 100.00% | 100.00% | 100.00% |
| Fine-tuned: high clarity | 100% | 99.50% | 85.19% | 100.00% | 100.00% | 100.00% |

High-clarity fine-tuning improved target mAP@50:95 by **8.62 percentage points** over the general baseline and **19.54 points** over low-clarity fine-tuning. Mixed clarity scored **1.23 points higher** than high clarity. The relationship was therefore not strictly increasing. Target mAP@50 was already near its ceiling; mAP@50:95 exposed differences in bounding-box localization.

At confidence 0.25 and matching IoU 0.5, the general and low-clarity models each had 24 true positives, 1 false positive and 0 false negatives. Mixed- and high-clarity models each had 24 true positives, 0 false positives and 0 false negatives.

## Held-out broader-domain results

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| General-dataset baseline | 99.50% | 75.73% | 100.00% | 100.00% | 100.00% |
| Fine-tuned: low clarity | 99.50% | 66.30% | 100.00% | 100.00% | 100.00% |
| Fine-tuned: mixed clarity | 99.50% | 75.11% | 92.31% | 100.00% | 96.00% |
| Fine-tuned: high clarity | 99.50% | **76.05%** | 92.31% | 100.00% | 96.00% |

High clarity approximately preserved the baseline's broader-domain mAP@50:95 (+0.31 points), while low-clarity fine-tuning reduced it by 9.43 points. However, mixed/high fine-tuning each introduced two false positives on this test, reducing precision and F1. Thus good target training data did not improve every broader-domain performance metric. A frozen general model is not changed merely by assessing a dataset; these changes occurred after adapting its weights.

## What FlightReady actually measured

The existing **UAV-14** image-clarity metric calculated training-only four-neighbor Laplacian variance. The experiment also reports the fraction of training images with variance >=100. This threshold was fixed before training as an illustrative property-specific rule. It is **not** an established mission threshold or an overall dataset-quality/readiness percentage.

| Target training condition | Mean pipeline Laplacian variance | Images passing illustrative clarity rule | Readiness |
|---|---:|---:|---|
| Low clarity | 8.48 | 0 / 32 (0%) | READY |
| Mixed clarity | 3,542.68 | 16 / 32 (50%) | READY |
| High clarity | 7,058.90 | 32 / 32 (100%) | READY |

All three datasets were READY for the declared detection requirements, yet their downstream scores differed. READY establishes the supplied requirements; it does not rank model performance. FlightReady currently has no validated composite readiness percentage. Sharpness can also measure noise, so this property cannot stand in for all aspects of dataset goodness.

The descriptive Spearman rank correlations between measured clarity and fine-tuned mAP@50:95 were **0.5 on the target test** and **1.0 on the broader test**. These are three dependent synthetic interventions, not three independent datasets; the correlations have no population-validation or statistical-significance claim. Artificial variants remain excluded from FlightReady's natural-dataset correlation analysis.

## Training setup and losses

- Execution completed in **335.9 seconds (5 minutes 36 seconds)** on CPU, without data/model downloads or a subsequent tuning run.
- General model: YOLO11n, trained from scratch on 48 broader synthetic training images for 100 epochs, with 8 validation images and 8 reserved internal test images.
- Each fine-tuned model: 32 paired target training images for 60 epochs, identical initial general-model tensors, seed, labels, counts, optimizer settings and augmentation defaults. Each had the same 8 target validation images and 8 reserved internal test images.
- Shared settings: seed 42, resolution 64, batch 8, SGD learning rate 0.01, deterministic mode, workers 0, AMP disabled.
- Low clarity applies Gaussian blur radius 2.5 to all target training images; mixed clarity applies it to a predetermined half; high clarity leaves all target training images unchanged. Validation/test pixels and annotation bytes stay identical across those conditions.
- Final testing: 24 newly generated target images and 24 newly generated broader-domain images. All training finished before these sets were generated. Every model used the same test data/settings within each domain. Exact hash, perceptual-duplicate and declared recording-group checks passed.
- mAP uses a confidence sweep with confidence floor 0.001 and NMS IoU 0.7. Operating precision/recall/F1 use confidence 0.25 and class-aware matching IoU 0.5.

| Model | Completed epochs | Final train box loss | Final train class loss | Final train DFL | Selected-checkpoint validation mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| General baseline | 100 | 1.39199 | 0.72953 | 1.07295 | 80.03% |
| Low-clarity fine-tuning | 60 | 1.00541 | 0.64837 | 0.98428 | 77.29% |
| Mixed-clarity fine-tuning | 60 | 1.00676 | 0.67971 | 0.99617 | 86.16% |
| High-clarity fine-tuning | 60 | 0.89279 | 0.59563 | 0.97000 | 85.84% |

Losses describe the final optimization epoch; test and validation metrics describe the checkpoint selected using validation data. Training losses across different input distributions are not directly comparable as accuracy. The baseline has 100 training epochs while adapted models have an additional 60 epochs, so baseline-versus-fine-tuning differences combine additional training and domain adaptation. The three fine-tuning conditions have matched adaptation budgets.

## What this establishes

The import, train-only measurement, matched fine-tuning, independent evaluation and evidence export workflow works. Within this fixture, avoiding severe blur helped target localization and preserved more broader-domain localization performance. The test **does not establish that a higher overall FlightReady percentage guarantees higher accuracy**: that score does not exist, target performance was not monotonic with clarity, and broader-domain F1 declined despite high clarity.

The general dataset here is a broader **synthetic generator**, not COCO/VisDrone or a general-purpose pretrained detector. Rectangles called `drone` are not real UAV photographs. One seed, one task, one quality property and 24 final images per domain cannot validate real UAV performance or general causal claims. No final test result was used to alter the training budget, model, threshold or quality criterion.

## Evidence and reproduction

- [Full results JSON](../.runtime/quality-performance-20261101/results.json) contains measured quality, configurations, validation scores, train/validation losses, target/general metrics and provenance IDs.
- [Results CSV](../.runtime/quality-performance-20261101/results.csv)
- [Generated report](../.runtime/quality-performance-20261101/report.md)
- Complete sources, assessments, split manifests, epoch CSVs, checkpoints, worker logs and per-image predictions are under `.runtime/quality-performance-20261101/storage`.
- [Experiment script](../backend/scripts/quality_performance_experiment.py)

```powershell
.\.venv\Scripts\python.exe backend/scripts/quality_performance_experiment.py --general-epochs 100 --finetune-epochs 60 --timeout 420
```

Each invocation creates an isolated `.runtime/quality-performance-*` directory and retains its evidence. No existing model or dataset is replaced.
