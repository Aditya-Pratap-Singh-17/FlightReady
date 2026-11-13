# FlightReady v2 - One-Page Research Summary

## What was built

FlightReady is a UAV dataset-readiness pipeline. It ingests tabular or image/annotation datasets, checks them against a confirmed task, records metric evidence, and produces an auditable readiness certificate. It is a research assessment tool, not a prediction of model accuracy or an aviation safety certificate.

## What the first experiments exposed

The original work trained 108 tabular models and seven YOLO11n detector conditions. Real performance fell under controlled degradation, but the optional v1 numeric score often did not. The tabular score stayed 100.0 with only 50% evidence coverage. On VisDrone, wrong classes reduced mAP50-95 by 12%, and removing 40% of images reduced it by 15.6%, yet the score stayed unchanged or increased. The main causes were missing semantic/coverage evidence, renormalizing over measured evidence only, and image proxies that could be improved by removing difficult examples.

## What was corrected

- V2 keeps missing evidence in the denominator and never treats it as passed.
- A primary score is withheld unless all applicable evidence is measured.
- Metric weights, normalized values, contributions, score bounds, and gates are exposed.
- Readiness categories come from confirmed gates, not from an average score.
- Trusted references can measure feature, label, class, box, and geometry disagreement.
- Expected-acquisition manifests can detect incomplete collections.
- Exact image duplicates are byte-verified; image changes use deterministic paired samples.

## How it was evaluated

All 18 preserved tabular variants and 13 real VisDrone collections were reassessed. Every saved tabular hash and every VisDrone image count matched its manifest. Historical model results were reused; no retraining was claimed. Six additional VisDrone defect conditions were held out from the original model matrix and used only for assessment sensitivity.

## What the measurements show

Combined tabular defects moved from v1=100.0 to v2=76.10; the repaired data returned to 100.0. Wrong-class VisDrone labels moved from 93.01 to 96.23 with a review warning; incomplete data moved from 93.16 to 88.01 and `NOT_READY`; repaired data returned to 100.0. V2 responded to all six held-out defect families. Correlations with reused performance became positive, but they are design-set, descriptive results and do not prove prediction.

## What remains unresolved

Without a trusted reference or acquisition manifest, reference-free coverage was only 33.3% for tabular and 37.5% for VisDrone, so semantic correctness and completeness remain unknown. Pixel metrics are imperfect proxies. The detector study needs more seeds, an independent dataset, and both fixed-epoch and fixed-update comparisons. The defensible conclusion is that FlightReady is now more transparent and sensitive to supported evidence, while important claims remain explicitly unverified.
