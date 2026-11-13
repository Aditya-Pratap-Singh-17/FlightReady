# Experimental evidence

FlightReady retains compact, reviewable evidence and deterministic scripts. Downloaded datasets, derived image collections, model weights, and training checkpoints are intentionally excluded.

## Completed work

### Tabular quality-impact study

- 18 controlled training-data conditions, two model families, and three random seeds: 108 fitted models.
- A fixed clean test set was unchanged across conditions.
- Combined degradation reduced mean R-squared from 0.1635 to 0.0573 for histogram gradient boosting and from 0.1567 to 0.0720 for random forest.
- Oracle restoration reproduced the clean-reference performance.

The historical experimental scalar stayed at 100 with partial evidence coverage even when performance declined. This negative result is preserved rather than presented as score validation.

Evidence: `experiments/data_quality_impact/`.

### VisDrone object-detection study

- Seven controlled YOLO11n training conditions were evaluated against one untouched official test-dev split.
- Clean mAP50-95 was 0.03328.
- Wrong-class labels reduced mAP50-95 by 12.0%; removing 40% of training images reduced it by 15.6%; combined defects reduced it by 15.2%.
- The repaired control reproduced clean aggregate metrics exactly.

The historical v1 score did not align with detector performance (Pearson r = -0.377, p = 0.405; n = 7). These results demonstrate sensitivity to controlled interventions; they do not establish a calibrated performance predictor.

Evidence: `experiments/visdrone_model_validation/`.

### FlightReady v2 reassessment

FlightReady v2 separates requirement gates from score summaries, retains unavailable evidence in coverage, reports bounds, and labels trusted-reference evidence. Its reassessment of preserved tabular and VisDrone variants produced directionally responsive reference-assisted scores. The reported correlations are descriptive design-set results only, because variants share source data and informed the method.

Evidence: `experiments/flightready_v2_validation/` and `docs/FlightReady_Improved_Research_Report.md`.

## Independent-validation status

The UAVVaste protocol, manifests, analysis scripts, and compact results are included in `experiments/independent_validation/`. The present run is incomplete. Do not cite it as completed validation; read `analysis/statistical_analysis.json` and `REPRODUCIBILITY.md` before using its results.

## Reproduce an assessment

1. Record the task, operating conditions, critical classes, and acceptance requirements.
2. Import the dataset and confirm its parsed layout.
3. Confirm the generated goal before assessment.
4. Inspect the decision, evidence coverage, unavailable evidence, and raw findings.
5. Export the certificate.
6. For performance claims, use a predeclared training protocol and an untouched test set.

## Reproduce research scripts

- Tabular study: `scripts/quality_impact_experiment.py`
- VisDrone study: `scripts/visdrone_validation.py`
- v2 reassessment: `scripts/flightready_v2_validation.py`
- UAVVaste protocol: `experiments/independent_validation/REPRODUCIBILITY.md`

Use only datasets and model checkpoints you are permitted to download and process. Dataset licenses remain separate from this repository.

## Interpretation boundary

FlightReady measures available evidence against a user-confirmed task. A `READY` decision means measured, applicable requirements passed under the recorded evidence. It does not establish semantic label truth, collection completeness, expected field performance, safety certification, or regulatory compliance without the necessary external evidence and validation.
