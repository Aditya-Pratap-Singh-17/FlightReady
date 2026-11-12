# Metric definitions and evidence limits

Registry definitions and calculation functions are in `metrics/core.py`. Sample sizes are descriptive denominators; they do not imply independent samples. Optional fields, missing metadata, unsupported modalities, exceptions and measured zeros have distinct states.

ImageFolder classification counts labeled readable images, not boxes; box metrics are not applicable. COCO/VOC detection adds hard manifest-consistency checks for orphan references and duplicate annotation identifiers. COCO crowd flags and VOC difficult/truncated attributes are preserved. Folder-inferred classes require user verification, while unmapped YOLO IDs cannot prove semantic class coverage. Imported nested JSON/XML fields remain structured JSON text and cannot establish scalar type requirements. Text datasets use the declared text/label columns and existing structural metrics; import does not establish semantic text quality.

| Dimension | Metrics / formula | Applicability and limitations |
|---|---|---|
| Integrity | Required schema field membership; missingness = missing / cells or rows; infinity rate = ±inf / numeric cells; unreadable rate; missing target/annotation-file rate | No universal missingness threshold. Image negatives are empty annotation files, not absent files. |
| Validity | Semantic type conformity; nonmissing range/allowed violations; exact declared unit equality; cross-field pair checks; invalid annotation-file fraction; fixed-date timeliness | Ranges/units require user/platform specifications. No guessed maneuver constraints. Nulls handled by separate missingness checks. Bbox invalid rate is a **file-level** rate, not a box-level error estimate. |
| Duplication | Duplicate rows or hashes beyond first occurrence / observations; repeated IDs; feature-identical groups with conflicting labels; optional 64-bit dHash candidate pairs | Repeated observations are not automatically defects. dHash is a heuristic with configurable threshold, not a semantic duplicate oracle. |
| Label quality | Per-class counts/fractions and max/min imbalance; Cohen kappa; trusted-reference mismatch fraction and audit coverage | Counts do not validate labels. Kappa requires a declared second independent annotation column; degenerate kappa is unknown. Fleiss/alpha, box matching across annotators and missing objects inside images are unsupported. |
| Coverage | Required class counts; condition-intersection counts; critical-class/scenario intersections; explicit minimum comparisons | No lighting/altitude inference from pixels. Missing metadata is unknown. Observed presence is not sufficient training sample size. Rare class counts are descriptive, not automatic corruption. |
| Representativeness | Numeric Wasserstein distance; categorical base-2 Jensen–Shannon divergence against selected reference columns | No reference means insufficient evidence; no invented representativeness score. Reference relevance and subgroup sampling are not inferred. |
| Leakage | Exact cross-split content duplicates; multi-split groups; near-duplicate image cross-split candidates; target-copy heuristics; declared preprocessing fit scope | Requires complete split/group metadata. No proof that all leakage is absent. Administrative identifiers do not hide content overlap. Pipeline fit scope is self-reported, not code inspection. |
| Evidence reliability | Provenance field availability; row/group counts; missing evidence; version, hashes, config and dependency records | No independent provenance authentication. No confidence intervals without justified sampling/cluster model. Neighboring frames are not assumed independent. |

## Formulas

- Missing fraction: `sum(isna(column)) / row_count`. Overall: `sum(missing cells) / (rows * columns)`.
- Infinity fraction: `sum(isinf(numeric cells)) / numeric_cell_count`. Missing numeric cells are reported separately; finite-only numeric summaries exclude both NaN and infinities.
- Duplicate fraction: `count(duplicated(keep='first')) / row_count`. For images use exact original file SHA256 instead of image metadata rows.
- Class fraction: `class_count / total_observed_labels_or_objects`. Imbalance: `max(nonzero_counts) / min(nonzero_counts)`. The smallest observed class excludes required classes with zero observations; those are separate hard coverage failures.
- Kappa: `(p_observed - p_expected) / (1 - p_expected)`, over complete annotation pairs. If `p_expected=1`, kappa is undefined, not perfect. Agreement does not establish correctness.
- Verified label mismatch: `mismatched_complete_reference_pairs / complete_reference_pairs`, with `audit_coverage = complete_reference_pairs / all_rows`. This is conditional on the reference's declared trustworthiness and does not automatically estimate population error.
- Scenario count: count of rows/images satisfying all field conditions, with optional exact required class. Detection counts use readable images with syntactically valid annotation files. Unknown metadata cannot establish zero coverage.
- Wasserstein: SciPy 1D first Wasserstein distance on finite numeric values, in feature units. Unit compatibility must be justified independently.
- Jensen–Shannon: `0.5*KL2(P||M) + 0.5*KL2(Q||M)` where `M=(P+Q)/2`, using the union of observed categories; calculated as squared SciPy base-2 Jensen–Shannon distance. No generic acceptance threshold.
- Timeliness: dates between fixed `as_of - max_data_age_days` and `as_of`, divided by all rows. Unparseable/missing dates do not satisfy the check. Fixed `as_of` is required to preserve repeatability.

For native VisDrone DET, target counts include usable score-one category 1–10 boxes; categories 0/11 and score-zero rows are ignored separately. Positive box area and image bounds are verified in pixel coordinates. Occlusion and truncation attributes are preserved without declaring them annotation errors. Invalid target rows affect `bbox_invalid_rate`; malformed geometry in excluded ignored regions is a separate diagnostic. The profile records raw annotation, ignored-region, other-category and invalid-target counts. Valid target rows in a file remain class-presence evidence when another row is invalid; scenario-image coverage remains conservative and excludes that image until its annotation file is valid. No ignored box or missing in-image annotation is counted as a verified target.

## Fully implemented versus conditional

Schema, missingness, infinity, exact duplicates, scalar descriptive profiles, declared type/range/value checks, YOLO bounds, class counts, scenario intersection and requirement status precedence are implemented directly.

Units, telemetry sensor mappings, provenance, timestamp age, leakage splits/groups, independent-annotator kappa, trusted-reference mismatches, reference comparisons and near-duplicate checks are conditional on their documented metadata/configuration. Missing evidence returns a reason, not a substituted score.

Fleiss/alpha, multi-annotator bounding-box matching, automated annotation completeness, confidence intervals for correlated flights, continuous monitoring and model-performance certification are not implemented. These limits are stated in certificates/documentation rather than reporting fabricated values.

## Disk-backed tables

`metrics/disk_metrics.py` calculates supported decision metrics against every database record; the first 100 preview rows are never used as assessment evidence. Results identify `scope=full_dataset`, backend and population count. Schema/units/provenance/pipeline declarations remain metadata checks. Full missingness and numeric summary statistics are cached in the initial profile; exact duplicates and distinct counts are calculated when selected. The initial profile leaves deferred values unknown rather than inventing zero.

Whole-dataset implementations cover usable targets, declared type/range/value constraints, numeric cross-field constraints, class distributions and required-class counts, scenario intersections, exact duplication and duplicate-label conflicts, identifier duplication, exact split/group leakage, target-copy heuristics, two-annotator Cohen kappa, trusted-reference mismatches, ISO-compatible age checks and evidence/group counts. Class-distribution exports are bounded to 10,000 classes; required-class counts still query all records. Target-copy diagnostics do not calculate correlation or mutual information on large tables.

Reference distribution distances are not implemented in the disk backend and return insufficient evidence. No sampled distribution is represented as a full-data result. Non-ISO timestamps cannot be conclusively checked by the SQL timestamp parser and remain unverified. Image-only metrics remain not applicable to these tables. SQL resource failures remain errors and cannot result in a READY certificate.

## Registry output

The registry declares metric ID, dimension, description, supported modalities, required fields/context (when statically expressible), method, output type, unit, applicability, evidence requirements and decision relevance. Runtime parameterized field requirements are resolved by `need_column`. The certificate embeds definitions for every selected metric so calculations remain inspectable.
