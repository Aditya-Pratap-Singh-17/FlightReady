# Reproducible UAV research protocol and validation checklist

Status: implementation and synthetic software validation are available. Real UAV empirical validation is pending. The following is an experiment plan, not a report of completed scientific experiments.

## Questions and preregistered hypotheses

1. Do native/model-input scale distributions describe held-out small-object recall better than image count alone?
2. Are independent audit categories associated with distinct changes in AP/recall/false positives under controlled annotation interventions?
3. Do object-local contrast/detail/background proxies add useful information beyond global thumbnail proxies?
4. Does directly observed joint scenario coverage add information beyond marginal scene/weather/scale counts?
5. How do collection groups/redundancy change uncertainty and generalization under group-separated final tests?
6. Do equal-weight, task-adaptive and gated policy descriptions differ in predictive usefulness on unseen dataset sources?

These are falsifiable hypotheses. Null findings, inversions, unsupported predictors and missing evidence must remain in the results. Do not optimize policy weights on the final held-out dataset sources.

## Prioritized real-data work

1. **Validate evidence first.** Acquire licensed UAV datasets with native images, complete annotation formats, documented class mappings and collection/session metadata. Obtain an independent annotation audit with adjudicated object/class/box/negative/ignored categories and a documented image sampling design. Confirm that deployment references represent the actual operating setting. Measure inter-reviewer disagreement separately from dataset errors.
2. **Run a small feasibility study.** Establish clean train/validation/final-test partitions using collection groups and existing exact/perceptual split checks. Review near-duplicate candidates. Freeze goals, native/input size criteria, preprocessing, sampling budgets, audit/reference fingerprints, software and scoring profiles before viewing final-test outcomes. Verify that interventions move intended measurements without silently damaging unrelated labels.
3. **Run matched paired interventions.** Use the existing Model Validation Lab experiment protocol, matched architecture, initialization hash, epochs/steps, batch, image size, learning rate, validation set, device and final-test identity. Use repeated seeds when resources allow; decide the seed count before inspecting results. No defensible precision/power guarantee follows from any arbitrary seed count.
4. **Replicate across independent sources.** Use independent collection datasets, locations/cameras/periods and representative test domains. Variants of one source and repeated training seeds are not independent datasets. Hold out entire sources when assessing prediction of performance from quality evidence.
5. **Evaluate predictive usefulness and failure cases.** Inspect individual metrics and each of the six dimensions, equal/adaptive baselines and critical gate compliance. Report unavailable dimensions instead of renormalizing away missing evidence. Report unchanged/worse performance as faithfully as improvements.

## Experimental matrix

| Training-only intervention | Controlled parameter | Primary measurement checks | Actual outcomes to examine | Main confounders |
| --- | --- | --- | --- | --- |
| Missing labels | Omission probability and seed | Independent missing-object audit; counts/presence | AP, recall, false positives | Annotation errors are not directly observable without reference audit |
| Wrong class | Reassignment probability; selected class mapping | Wrong-class audit category; confusion | Per-class AP, recall and confusion | Class prevalence and co-occurrence |
| Localization | Box-center perturbation magnitude | Localization audit, box validity, scale changes | AP at stricter IoU, recall | Matching thresholds, boxes below low-IoU cutoff |
| Blur | Gaussian radius | Global/local detail and contrast | AP and size-stratified recall | Native resolution, texture, encoder/preprocessing |
| Noise | Achromatic Gaussian sigma; seed | Residual MAD and detail separately | AP, false positives, recall | Compression, saturation and texture |
| Exposure | Recorded brightness factor | Low/high endpoint fractions; contrast | AP and scenario/size recall | Clipping, already dark/bright source content |
| Small-object reduction | Native-area interval, class and retained fraction | Object/image/class bins and joint scenarios | Ground-truth-area recall, per-class AP | Reduced image/object count; changed scene mixture |
| Hard-negative-candidate reduction | Explicit candidate metadata column and retained fraction | Target-free/unknown/verified/candidate counts | False positives/image on independent negative-rich tests | Candidate selection bias; missing completeness audit |
| Scenario reduction | Explicit metadata condition and retained values | Direct joint scenario counts and unknowns | Relevant held-out scenario performance | Image count, class/source mixture and metadata quality |

Selection is at whole-image level; no retained image silently loses only the selected objects. Small-object and negative selection can change total training size. Create count-matched control subsets with recorded seeds and inspect all changed covariates. Training budget equality in epochs does not necessarily mean identical update count if sizes differ; record steps as well. The existing comparison flags unmatched image counts rather than calling them fair causal comparisons.

For direct script usage, `research.variants.VariantConfig` supports `image_degradation` (`blur_radius`, `brightness_factor`, `noise_sigma`), `annotation_noise`, `label_omission`, `box_perturbation`, `small_object_reduction` (`maximum_native_area`, `retain_fraction`, `classes`), and `negative_coverage` (`column`, `retain_fraction`, `classes`). Existing `environment` and other interventions remain. The workbench `/api/v1/variations` is the canonical public adapter. Every variant requires a new output dataset; original data and evaluation labels remain immutable.

## Outcomes and interpretation

- Primary detection outcomes: actual mAP50, mAP50–95 and per-class AP from the existing validator; class-aware precision/recall/F1 at declared confidence, matching IoU and NMS thresholds; actual false positives per image.
- Size outcome: operating-point recall uses **ground-truth native box area**. Unmatched false positives are separately stratified by predicted box area. Current small-object AP is explicitly unavailable. These are not official COCO mask-area AP metrics.
- Compare paired variant-minus-control changes under identical final-test/protocol hashes and training seeds. The existing paired analysis retains zero/negative changes and related-source provenance.
- Aggregate repeated seeds within each independent source before source-level correlations. Use group-aware uncertainty respecting flights/sessions and independent dataset units; do not treat neighboring frames as IID trials.
- Existing Pearson/Spearman associations and leave-one-source-out OLS are exploratory. Inspect outliers, multiplicity, restricted range, sample size, model mismatch and missing predictors. Do not use correlation as evidence of causation or fit quality-policy weights on reported final-test outcomes.
- Research-only equal/adaptive scalar baselines are comparison predictors, not replacements for the six independent product dimensions. Examine critical gates independently; a high score in unrelated dimensions cannot compensate for blocked requirements.

## Confounders and validation requirements

Record class ontology, annotation format/crowd handling, native and input resolution, preprocessing/augmentation, image/object counts, collection dependence, weather/time metadata quality, source locations/sensors, corruption mixtures, seed and initialization, compute budget, validation-based model selection and test prevalence. A visual proxy may respond to texture or compression rather than the intended degradation. Bounding-box regions mix foreground/background; hard-negative candidates may be mislabeled positives. Deployment reference relevance must be justified independently.

Synthetic tests establish calculation, serialization, missingness and workflow invariants. They cannot establish real UAV annotation correctness, semantic visibility, independence of collections, a universal sample minimum, proxy calibration, downstream accuracy gains, statistical predictive validity, deployment safety or IEEE publication readiness.

## Reproducible checklist

- [ ] Data license and source/version fingerprints recorded; native files preserved.
- [ ] Class mapping and annotation-format/ignored-region semantics reviewed.
- [ ] Independent audit identity, complete references, reviewer protocol, sampling design and adjudication recorded; unverified model predictions excluded as truth.
- [ ] Deployment reference provenance and relevance documented, including missing condition metadata.
- [ ] Confirmed goal/scenarios/minimums and exact preprocessing frozen before final-test inspection.
- [ ] Metric, engine, profile, dependency and local-model versions/fingerprints archived.
- [ ] Pixel/duplicate budgets, sample seeds, sample sizes, coverage and unsupported conditions recorded.
- [ ] Exact/perceptual/group split checks reviewed; recording-group uncertainty acknowledged where unresolved.
- [ ] Variant parameters, affected/removed population and source/output hashes archived.
- [ ] Validation/test image and annotation bytes unchanged across paired variants.
- [ ] Architecture, initialization hash, optimization, preprocessing, budget and evaluation protocol matched; deviations reported.
- [ ] Repeated seeds and independent dataset sources distinguished; no source-overlapping variants in independent-source correlations.
- [ ] Count-matched controls and cross-metric side effects inspected.
- [ ] Actual AP/recall/false-positive outcomes collected; no fabricated outcomes or small-object AP.
- [ ] Equal/adaptive/gate comparisons preregistered; no final-test weight fitting.
- [ ] Uncertainty and multiplicity addressed at defensible independent sampling units.
- [ ] Null/negative results, unmeasured predictors and method failures retained.
- [ ] Cross-source replication completed before broad claims of predictive validity.
- [ ] Full JSON/CSV evidence, manifests, configuration and software commands archived with the research release.

See [measurement definitions/configuration](research-metrics-v2.md) and [actual software validation results](research-metrics-validation-results.md).
