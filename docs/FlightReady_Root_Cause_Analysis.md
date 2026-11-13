# FlightReady Root-Cause Analysis

## Scope and evidence

This analysis preserves the original tabular and VisDrone results as historical evidence. FlightReady had two different outputs: a requirement-based categorical decision and an optional experimental weighted score. They must not be interpreted as the same mechanism.

## Confirmed causes

| Observed behavior | Confirmed cause | Classification | v2 correction |
|---|---|---|---|
| Tabular score stayed 100 with 50% coverage | The experimental score renormalized over only measured components; the two measured components passed while unknown components contributed no penalty. | Misleading aggregation/reporting, not an arithmetic error | Retain unknown weight, expose bounds, and withhold a primary score unless coverage is complete. |
| Missing data produced NOT_READY while score stayed 100 | The decision correctly used confirmed per-feature missingness requirements, but the experimental score did not include those requirement findings. | Inconsistent output scopes | Show gates and score components together; gates determine the decision. |
| Noise and label corruption were missed | Values remained schema-valid. No trusted row/label reference or semantic relationship criterion was supplied. | Missing capability/evidence | Add reference-assisted disagreement checks; without a trusted reference report insufficient evidence. |
| VisDrone incomplete data scored higher | Removing difficult images changed clarity and uniqueness summaries favorably; no expected-image manifest component existed in the score. | Construct-validity/coverage gap | Add expected acquisition count/segment evidence; otherwise completeness is unavailable. |
| Wrong classes did not change the score | Syntax, class presence and distributions cannot establish per-object semantic correctness. | Missing semantic evidence, not a direction bug | Support trusted annotation audits and explicitly label reference-assisted agreement. |
| Combined defects had the highest v1 score | Omitted annotations reduced the population of hard/small objects and image removal/noise shifted proxy distributions; compensatory weighted means rewarded easier remaining evidence. | Aggregation and proxy-confounding problem | Use non-compensatory gates, separate dimensions, reference comparisons, and metric-level contributions. |
| Every VisDrone variant was NOT_READY | A confirmed hard requirement failed across the common dataset; the score was never the decision. | Requirement outcome; not a score bug | Preserve gate trace and explain the common failure independently of quality summaries. |
| Image noise barely affected clarity | Laplacian variance is increased by both detail and noise, so it is not a monotonic noise measure. | Method limitation | Report multiple proxies and paired/reference evidence; do not claim any single proxy proves corruption. |

## Methodology boundary

V2 does not infer semantic label correctness from class histograms, or acquisition completeness from the observed file count alone. Those dimensions remain unavailable without a trusted audit or expected-acquisition manifest. Reference-assisted results in the controlled experiments therefore use more information than the original reference-free run and are labelled accordingly.
