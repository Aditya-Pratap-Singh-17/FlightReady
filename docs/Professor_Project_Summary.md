# FlightReady: UAV Dataset Readiness and Model Validation

**Development duration:** Approximately one month  
**Project type:** Undergraduate research software and preliminary experimental validation

## Research objective

FlightReady investigates whether a context-aware dataset assessment can help researchers identify data problems before training UAV machine-learning models. The project combines dataset ingestion, task-specific quality evidence, readiness certificates, controlled data degradation, and held-out model evaluation in one local research workbench.

## System implemented

- A FastAPI backend with bounded local ingestion and a React/TypeScript interface.
- Support for common tabular, record, text, image, and archive formats, including YOLO, native VisDrone DET, COCO detection, Pascal VOC, and ImageFolder layouts.
- Human-confirmed task requirements and evidence-aware `READY`, `NOT_READY`, or `INSUFFICIENT_EVIDENCE` decisions.
- Quality checks covering integrity, validity, duplication, label evidence, coverage, representativeness, leakage, provenance, and conditional UAV properties.
- Auditable JSON/HTML certificates, analytical views, persistent research records, controlled dataset variants, and model validation jobs.
- Reproducible experiment scripts, manifests, hashes, figures, and result tables.

## Experimental evaluation

The tabular study trained 108 models across 18 data conditions, two architectures, and three seeds. Combined degradation reduced mean R-squared from 0.1635 to 0.0573 for histogram gradient boosting and from 0.1567 to 0.0720 for random forest. Restoring the original data recovered both results. Missingness was detected categorically, while schema-valid feature noise and label corruption were not reliably reflected by readiness outputs. The experimental numeric score was constant at 100.0 with only 50% evidence coverage.

The second study trained seven YOLO11n conditions on real VisDrone imagery using one fixed official test-dev split. Clean mAP50-95 was 0.03328. Wrong-class labels reduced it by 12.0%, removing 40% of training images reduced it by 15.6%, and combined defects reduced it by 15.2%. Restoring clean bytes and labels reproduced clean aggregate performance. FlightReady's numeric score was not aligned with detector performance (Pearson r=-0.377, p=0.405; n=7).

## Conclusion and limitations

The project demonstrates a working readiness-assessment and model-validation pipeline, and the controlled experiments show that dataset defects can materially affect downstream models. The negative result is equally important: the current aggregate experimental score is not a reliable predictor of model accuracy. FlightReady should be presented as an auditable research prototype, not as an aviation-safety certificate or validated performance forecast.

The evaluation used one tabular dataset and one UAV detection dataset; the YOLO study used a single training seed and a short five-epoch, 320-pixel budget. Semantic label correctness and missing acquisition coverage require trusted audits or external manifests.

## Future directions

Recommended work includes multi-dataset external validation, flight/group-aware splits, repeated dataset resampling, blinded label audits, image-corruption measures, transparent score contributions, and prospective calibration against downstream outcomes.

**Repository:** No public GitHub URL is configured in the local repository. Add the final URL here after publication.
