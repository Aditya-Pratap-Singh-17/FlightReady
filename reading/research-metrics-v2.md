# UAV research evidence v2: measurement specification and compatibility

Implementation date: 3 November 2026. These are descriptive measurements and confirmed requirement checks, not calibrated predictors of detection performance. No real UAV training experiment was run as part of this implementation.

## Integration and compatibility

The upload/ingestion → `metrics.run_metric` → existing UAV registry → `MetricResult` / `AssessmentEvidence` → requirement findings → six-dimensional aggregation → `Certificate` → durable record → reports/Model Validation Lab flow is retained. The public UAV-01 through UAV-20 IDs, endpoints and legacy measurement keys remain stable. UAV-08/12/14/15/17/18 declare implementation version `2.0`; their added evidence declares `research-evidence-2.0`. Engine version is `1.5.0` and records metric versions in its assessment fingerprint configuration. New ingestion caches add protected `research_*` pixel fields and `dhash64`; image metadata cannot override computed fields.

The default `adaptive-1.0` scoring configuration, component assignments, engineering weights, within-dimension geometric aggregation, strict weighted coverage, independent gates, and old experimental scalar are unchanged. New proxies are **not automatically scored**, nor made blockers. They lack a defensible universal monotonic quality direction. Explicit requirements may select an exact structured `output_path`; parent/submeasurement unavailability prevents projecting an inspection count as a measured quality value. Scale/negative/scenario evidence belongs conceptually to diversity/operational suitability, local visual proxies to signal quality, independent audits to annotation quality and collection evidence to integrity. This conceptual mapping does not add duplicate scoring components.

Historical certificates, stored profiles and fingerprints are not rewritten. The old count-only `annotation_audit.reviewed_objects/incorrect_objects` agreement component remains the original policy input; the new category-specific reference audit does not silently redefine that denominator. A future scored audit component requires a separately versioned policy and validation of category weights. Existing profile history/current-profile APIs continue comparing frozen policies. New measurements may be unavailable in old cached datasets: reassess for local measurements; reimport to collect new ingestion-wide proxies/signatures. Missing cache fields do not become zeros. Task previews discard context-dependent enhanced UAV results and retain their existing reassessment requirement.

Intentional semantic correction: `scenario_count` uses three-valued conjunctions and retains observed count as a lower bound. An unmet minimum with unresolved matching candidates is insufficient evidence, not a proved failure. A satisfied lower-bound minimum can pass. Object predicates are evaluated on the same annotated object. This behavior is engine-versioned; older saved findings remain unchanged.

Population-scope correction: training subsets retain only independent reference images/candidates belonging to that subset, record the parent audit fingerprint and bind the derived audit to the subset identity. Original metadata is deep-copied and remains unchanged. Count-only audits without image-level scope are unavailable for a new subset; they are not extrapolated to training images. Combined datasets retain source audits in source provenance rather than reinterpreting them as one combined audit. These are intentional evidence-scope exceptions for new assessments, not rewrites of historical results.

## Measurement map

| Stable property / addition | Previous evidence retained | New evidence / source code | Scoring effect |
| --- | --- | --- | --- |
| UAV-08 object scale | Per-class normalized/native box geometry and declared legacy bins | `scale_analysis`: native/model-input distributions, object/image/per-class bins and explicit small-object criteria; `metrics/uav_metrics/research.py` | Descriptive; explicit requirements only |
| UAV-12 clutter | Luminance entropy and historical density semantics | Cached global edges, annotated density/crowding, local background rings, size/entropy cross-tabs and review candidates; `pixels.py` | No universal clutter gate |
| UAV-14 clarity | Resolution, brightness, contrast, thumbnail Laplacian variance, bytes/pixel | Separate global degradation indicators; object-region detail/noise/contrast/exposure and class/size distributions; `pixels.py` | Existing clarity policy unchanged |
| UAV-15 annotation audit | Presence, syntax, manifest diagnostics and older optional counts | `structured_audit`: independent-reference one-to-one matching, eight categories, rates, class/size strata and coverage; `audit.py` | New audit is not substituted into old agreement score |
| UAV-17 deployment | Explicit-reference Wasserstein / categorical JS and optional matching local encoders | Conditional image/object intersections, unknowns, reference fractions/datasets, bounded optional pairwise summaries; `research.py` | Descriptive without meaningful reference; confirmed scenario findings feed existing requirement component |
| UAV-18 independence | Class counts, hashes, legacy duplicates and leakage metrics | Cached dHash/BK-tree candidate groups, declared collection group summaries and cross-split evidence; `independence.py` | No effective sample-size score or invented minimum |
| UAV-14 object visibility proxies | Existing explicitly confirmed minimum box side remains separate | Object/ring luminance difference, normalized contrast, edge separation, texture/detail distributions | No detectability probability |
| UAV-15 negative coverage | Empty label files remain distinct from missing labels | Explicit negatives, other-class-only images, unknown labels, independent negatives, scenario coverage and documented candidate records | No false-positive performance claim |
| Existing `scenario_count` + UAV-17 joint coverage | Explicit task scenarios and requirements | One shared evaluator; class/size/occlusion conjunctions on each object and metadata on its image | Existing confirmed requirements; unknown lower-bound correction |

All additions are CPU-only and use existing NumPy/SciPy/Pillow/Pandas dependencies. No optional model or GPU is required. Optional ONNX integrations remain opt-in and local. No model download, dataset transmission, archive code execution or automatic relabeling/deletion was introduced.

## UAV-08: scale geometry

Inputs: syntactically valid normalized center boxes `(x,y,w,h)`, observed native dimensions `(W,H)` and usable class labels. Reloaded boxes are checked again for finite positive dimensions and containment. Missing class/dimensions or invalid boxes are excluded and counted. Native calculations cover valid observed annotations; they do not count unannotated objects.

For every usable object:

```
width_px = w W                  height_px = h H
area_px2 = w h W H              relative_area = w h
aspect_ratio = (w W)/(h H)      minimum_side_px = min(w W, h H)
```

Model input is unavailable unless `detector_preprocessing` supplies input shape `[Ih,Iw]`, mode and explicit `scaleup`. Ideal letterbox gain is `r=min(Iw/W,Ih/H)`, additionally capped at 1 when scaleup=false. Box sides become `(r w W, r h H)`; padding changes position, not dimensions. Stretch mode uses independent gains `(Iw/W,Ih/H)` and requires scaleup=true. This is the deterministic ideal box transform, not a simulation of training augmentation, integer raster rounding or automatic rectangular batching. Unspecified preprocessing never defaults to square stretching. The geometry follows the explicit distinction in [Ultralytics LetterBox documentation](https://docs.ultralytics.com/reference/data/augment/#ultralytics.data.augment.LetterBox).

Default native-area bins are `[0,1024)`, `[1024,9216)`, `[9216,infinity)`, named small/medium/large. They are transparent descriptive intervals inspired by common pixel-area reporting conventions; they are not a UAV detectability standard, and do not reproduce COCO mask-area evaluation. See the separate official [COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py). `object_size_bins` can replace them with 1–16 ordered disjoint `[lower,upper)` intervals. Gaps appear as `outside_declared_bins`.

Output `scale_analysis`: object count, excluded count, image population, images with valid objects, counts and object percentages by bin, image counts containing each bin, per-class bin counts, native/input distribution summaries and explicit task coverage. Percentages divide by usable **objects**; image counts count each image once per bin and can sum above the number of images. Quantiles are NumPy linear q0/q05/q25/q50/q75/q95/q100 with population mean. Histograms use at most 12 bins, with the last interval including its upper edge.

`small_object_criteria` needs explicit classes, a finite native-area range and optionally `minimum_objects`. Missing annotations/invalid excluded objects retain incomplete-population status; a known observed count can establish an already-met minimum, but missing population cannot prove absence. Class-specific and scene/group diversity must be inspected independently. Geometry never establishes recognizability.

Units: pixels, pixels squared and ratios. Complexity: O(objects + images), plus sorting for quantiles; exact summaries use observed annotation populations, not sampled crops. Integration: `assessment.calculate('UAV-08')`. Tests: native/rectangular input geometry, letterbox scaleup, input omission, invalid/tiny boxes, separate object/image denominators and criterion coverage in `tests/test_research_metrics.py`.

## UAV-12 / UAV-14: global and object-local pixels

Existing global brightness/contrast remain native-image summaries. Existing entropy and Laplacian remain their historical <=256-pixel thumbnail measurements with historical units. New global `research_*` fields reuse that ingestion thumbnail once and normalize gray intensity `a=gray/255`. Local measurements require one additional decode per selected image because ingestion-wide summaries cannot reconstruct target crops. That decode is shared across UAV-12/14 within the assessment's existing vision session. Staged image hashes, native dimensions, 20 MiB/image, single-frame and 20-megapixel limits are checked again. Workbench runtime-only readers support archives, registered directories and virtual combined sources, without serializing read capabilities.

New formulas, using population means/variances:

```
entropy_bits = -sum(p_i log2(p_i))          # 256 gray histogram bins
edge_density = mean(hypot(gradient_x(a), gradient_y(a)) > 0.08)
laplacian = -4a[y,x]+a[y-1,x]+a[y+1,x]+a[y,x-1]+a[y,x+1]
detail_proxy = Var(laplacian)              # interior pixels
residual = a - GaussianFilter(a, sigma=1, reflect)
high_frequency_residual_MAD = median(abs(residual-median(residual)))/0.67448975
local_contrast_RMS = mean(sqrt(max(0, mean_3x3(a^2)-mean_3x3(a)^2)))
underexposed_fraction = mean(a <= 5/255)
saturated_fraction = mean(a >= 250/255)
```

Gradient threshold and endpoint gray cutoffs define indicators; they are not pass/fail criteria. MAD mixes texture, compression and noise and is not a calibrated sensor-noise estimator. Noise can raise Laplacian variance; a higher detail proxy is not automatically better clarity. Local detail/residual/contrast is undefined for crops smaller than three pixels along either side.

A local target uses floor/ceil box edges on an aspect-preserving <=512-pixel grayscale thumbnail by default. Its surrounding window expands box width/height by 2, clips to image borders and excludes the target and other known annotated boxes. A minimum of 16 remaining ring pixels and a complete bounded neighbor scan are required for ring summaries. Global neighbor comparison budget defaults to 100,000; incomplete ring neighborhoods are unavailable. Overlap/crowding counts are separate from semantic clutter. Ring edges are finite-difference gradients of the local window sampled at ring pixels; boundaries can contribute gradients, so this is not a segmentation-based background measurement.

```
luminance_difference = abs(mean(target)-mean(ring))
normalized_local_contrast = difference / max(1/255, mean(target)+mean(ring))
edge_separation = edge_density(target)-edge_density(ring)
```

Ring texture is population gray standard deviation; ring entropy is a histogram over remaining pixels. All denominators are pixels in their respective regions or objects with a defined region, not all dataset objects. Bounding boxes are not masks; unannotated objects can remain in rings. Contrast does not imply semantic visibility, and local entropy does not establish detection difficulty.

Sampling: deterministic SHA-256 bottom-k over `(seed,image path)`, then `(seed,image path,object index)`, default seed 42, at most 128 selected images and 1,024 objects. Configurable hard caps: 512 images, 4,096 objects, thumbnail side 1,024, one million neighbor comparisons. Cache keys include dataset identity, local configuration and size-bin configuration. No random state is borrowed from training. Outputs record selected/inspected images, object population, inspected objects, coverage, seed, exact settings, failures and full-population versus bounded-sample scope. No population confidence interval is fabricated.

Outputs: per-quantity histograms/quantiles, per-class/per-native-size distributions, size versus ring-entropy cross-tabs, and at most 20 ranked review candidates. Cross-tab boundaries are the inspected ring-entropy sample's q25/q75, with ties in the central interval and unavailable rings in an unknown bucket. These relative categories are not calibrated high-clutter labels. Highest ring entropy is a candidate reason, not verified difficulty.

Units: bits, normalized gray intensity, squared normalized intensity, fractions, signed edge-fraction differences, and nearby annotated object counts. Global summaries use all cached readable-image measurements; local summaries are observations on the inspected sample. Complexity: O(images) cached ingestion work; additional local work bounded by image/object/pixel/neighbor budgets. Tests cover uniform/noisy/blurred pixels, exposure endpoints, target exclusion, border/full-image boxes, unavailable rings, deterministic sampling and shared caching.

## UAV-15: independent audit

The old syntax/presence/manifest measurements remain. New semantic evidence requires upload metadata `annotation_audit.images`, `independent=true`, `evidence_type=human_verified|trusted_external`, reviewer, protocol and source ID. Optional `dataset_sha256` must match. Trust is declared, not authenticated. `model_prediction` is ineligible independent reference evidence, regardless of a caller's independent flag. No reference means semantic correctness remains unverified.

Each uniquely named image requires `reference_complete=true` and normalized `reference_objects` with classes. Ignored/uncertain reference objects are explicit; annotations overlapping them above the matching threshold are excluded from error claims. Missing/invalid dataset annotations are syntax/presence failures and **do not** automatically become semantic omissions. Complete reference images with no active/ignored objects can separately carry `negative_verified=true`.

Algorithm: enumerate box/reference IoUs, sort descending with index tie breaks, and match one-to-one. Configurable default `match_iou=.5`, `localization_iou=.1`, requiring `0<localization_iou<match_iou<=1`. Primary matching supports class-aware/class-agnostic mode. A high-IoU unmatched class-agnostic pass identifies wrong classes; a lower-IoU same-class pass identifies localization discrepancies. Remaining annotations overlapping an already matched same-class reference are duplicates; other unmatched annotations are spurious. Unmatched references are missing objects. Very poorly localized boxes below the lower threshold appear as missing + spurious, an explicit matching limitation.

IoU is normalized rectangle intersection area divided by union area. Categories and denominators:

| Category | Numerator | Denominator |
| --- | --- | --- |
| `missing_objects` | Unmatched active references | Active reference objects in semantically assessable audited images |
| `incorrect_classes` | High-IoU matched class discrepancies | Same active reference objects |
| `localization_errors` | Lower-IoU matched same-class boxes | Same active reference objects |
| `duplicate_annotations` | Extra same-class annotations on matched references | Reviewed nonignored dataset annotations |
| `spurious_annotations` | Other unmatched dataset annotations | Same reviewed annotations |
| `incorrect_negative_images` | Valid empty dataset annotations despite active references | Audited images |
| `ignored_uncertain_objects` | Independently ignored/uncertain references | Active + ignored reference objects |
| `invalid_syntax_images` | Present but invalid dataset annotations | Audited images |

Each rate is numerator/denominator; zero denominators give null. Per-class and size strata carry their own object counts and rates. Image-level categories can overlap object errors; summing category rates is invalid. Reports retain images audited, references, annotations reviewed, image/annotation coverage, matching protocol, audit fingerprint, image-level results and declared sampling design. No unreviewed image is counted correct and no subset is extrapolated to the whole dataset.

Budget: at most 512 audited images, 500 reference/annotation objects per image and 200,000 reserved pair-comparison operations; oversized requests fail explicitly. Complexity O(sum A_i R_i) within that fixed cap, plus candidate sorting. Tests cover wrong classes, duplicate one-to-one matching, localization, missing/spurious objects, negative errors, invalid syntax, ignored scope, reference completeness, both matching modes and undefined denominators.

### Negative and hard-negative coverage

`negative_coverage` distinguishes valid empty annotations, missing/invalid labels, valid annotations containing only other classes, independently verified global negatives, and explicitly supplied hard-negative candidates. Explicit-negative proportion divides empty valid annotations by all valid annotated readable images. Target-free per-class counts use that same valid population; missing files never establish absence. Verified-negative population coverage and coverage within the annotated target-free population are separate fractions. An independently verified negative can still have a missing dataset annotation file: those categories deliberately overlap.

Image-metadata scenarios have per-scenario negative counts; object-presence scenarios cannot define an absent-target population and are marked unavailable for that count. Candidate metadata requires an image already known annotation-target-free, a reason and source ID. Model-based candidates additionally require model ID and confidence threshold. Candidates remain unverified; no actual false-positive rate is inferred.

## UAV-17 and shared conditional scenarios

Each confirmed conjunction is evaluated directly; marginal distributions are never multiplied. Image metadata uses explicit column mappings. Reserved object predicates include class, object_size, occlusion, truncation, native area, minimum side and relative area. Class and scale must match the **same** valid object. Weather/time/altitude are never inferred from appearance.

Named `object_size` predicates require explicitly supplied `object_size_bins` in the confirmed goal. Without them, scale predicates remain unknown; descriptive default bins alone do not become task acceptance criteria. Direct numeric native-area/side conditions already specify their own criterion.

Atomic comparison returns true/false/unknown; numeric intervals are inclusive min/max and require finite bounds. In a conjunction, an observed false dominates unknown. An image matches when image conditions and an eligible object match; valid explicit negatives can establish class nonmatch, while missing/invalid labels cannot. Unknown is not absence. Tabular scenarios use recorded rows and the declared prediction target.

For every scenario, output population, known eligible population, matching image count, matching object count, per-class counts, known nonmatching count, missing-metadata count, unknown count, optional minimum, status, provenance and representative references. Unknown and known populations partition images. `matching_image_fraction` is null when unknowns exist; `[matches/N,(matches+unknown)/N]` are identification bounds, **not a confidence interval**. A confirmed minimum passes when observed count already meets it, fails only with a conclusive shortfall, and otherwise remains insufficient evidence.

Optional comparison uses a supplied reference dataset or `scenario_reference_fractions` keyed by scenario name. Differences in fractions are computed only when both image populations are conclusive. Reference relevance and metadata authenticity remain declared. Existing explicit numerical Wasserstein and categorical JS paths are retained. Without a meaningful reference, the output states that deployment representativeness is unverified even when descriptive coverage is available.

Caps: 32 confirmed scenarios, at most 12 predicates each. Optional `scenario_pairwise_limit` is 0–16 (default 0), in deterministic confirmed-scenario order; conflicting predicates are unsupported and require an explicit conjunction. Complexity O((scenarios + selected pairs)*(images + objects)); no automatic combinatorial expansion. Tests cover cross-object false intersections, class conflicts, missing fields, false/unknown conjunction, lower-bound requirements, references and pairwise caps.

## UAV-18: collection groups and duplicate candidates

Full cached SHA-256 evidence gives unique exact hashes, duplicate images (`sum(group_size-1)`), duplicate groups and cross-split duplicate groups. Missing hashes are unavailable. New ingestion stores the existing 64-bit difference hash: 9x8 grayscale resize, 64 comparisons of each pixel with its left neighbor. It is the same perceptual-hash family used by the existing split service.

Near-duplicate analysis reuses the split service's BK-tree with a separate budgeted query method. Defaults: Hamming distance <=4, sorted image order, at most 20,000 images, 200,000 node visits, 128 query results and 100,000 total candidate links. Hard caps: 50,000 images, one million visits, 1,024 results/query and 500,000 candidate links. Exact groups are unioned first; retrieved candidates form transitive components. Result/visit/link truncation explicitly sets incomplete retrieval. Report observed pair counts, cross-split candidates, group-size distributions, bounded examples, signature population/coverage and exact configuration. Zero retrieved pairs under incomplete retrieval is not proof of no near duplicates. Existing strict split safeguards still use the unrestricted exact query and are not weakened.

Group columns are explicit or conventional flight/session/recording/video/camera/location/time-group names. Outputs distinguish missing metadata, group counts, group-size distributions, cross-split group overlap and per-class group coverage. A camera/source group is not automatically an independent flight. Filenames, distinct bytes, declared groups and statistical effective sample size are different concepts. Absent flight/session metadata keeps independence uncertain. No universal sample minimum was added; confirmed count requirements continue defining sufficiency.

Complexity: linear hash/group aggregation; budgeted BK-tree retrieval avoids full all-pairs comparison. Tree storage is O(selected images); quantile summaries sort observed group sizes. dHash collisions and transitive chains require review; group IDs are not authenticated. Older caches without dHash stay unavailable rather than decoding an unbounded archive just for this enhancement. Tests verify exact/candidate groups, cross-split examples, class grouping, absent metadata and budget exhaustion.

## Configuration examples

Confirm these fields as part of the existing goal/context API or the advanced structured-context editor:

```json
{
  "task_type": "object_detection",
  "required_classes": ["drone"],
  "scenarios": [{"name":"small drones in recorded night conditions","conditions":{"lighting":"night","object_size":"small"},"required_class":"drone","minimum_samples":20}],
  "uav_parameters": {
    "detector_preprocessing":{"input_size":[640,640],"mode":"letterbox","scaleup":false},
    "object_size_bins":{"small":[0,1024],"medium":[1024,9216],"large":[9216,null]},
    "small_object_criteria":{"classes":["drone"],"native_area_range":[0,1024],"minimum_objects":40},
    "local_analysis":{"max_images":128,"max_objects":1024,"max_side":512,"seed":42,"ring_expansion":2,"minimum_ring_pixels":16,"max_neighbor_pairs":100000},
    "near_duplicate_analysis":{"distance":4,"max_images":20000,"max_tree_visits":200000,"max_results":128},
    "scenario_pairwise_limit":4,
    "collection_group_columns":["flight_id","video_id","camera_id"]
  }
}
```

The example minimums are user choices for demonstrating configuration, not research-validated UAV sufficiency thresholds.

Independent audit upload metadata example (reference boxes use normalized center coordinates):

```json
{"annotation_audit":{"independent":true,"evidence_type":"human_verified","reviewer":"Your independent reviewers","source_id":"audit-release-1","protocol":"Your documented review and adjudication procedure","sampling_design":"Describe your actual sampling design","matching_mode":"class_agnostic","match_iou":0.5,"localization_iou":0.1,"images":[{"image":"images/example.png","reference_complete":true,"reference_objects":[{"class":"drone","bbox":[0.5,0.5,0.1,0.1]}]}]}}
```

This is a schema example, not supplied audit evidence. Use the existing upload metadata field; no model predictions are promoted to independent truth. Add image-metadata columns through the existing `metadata.csv` convention, not by replacing protected inspection fields.

## Interface, exports and experimental support

The existing guided report and advanced analytics both include a selectable research-evidence view with backend-derived plots, accessible data tables, units, denominators, sample sizes, coverage, provenance, versions and limitations. Empty/unsupported distributions say unavailable and do not draw a zero chart. `quality_measurements.research_evidence` is a presentation view of raw registry results, not a second assessment engine. Raw structured evidence remains in certificate JSON/CSV; HTML/PDF reports include readable new plots-as-tables and measurement summaries. PDF methodology follows existing explicit abbreviation limits; JSON remains the full raw-evidence export.

Controlled interventions extend the existing train-only framework: wrong-class labels reuse `annotation_noise`; Gaussian pixel noise and exposure reuse `image_degradation`; whole-image small-object reduction and explicitly declared target-free hard-negative-candidate reduction extend selection. Existing omission, localization, resolution, class/environment/viewpoint and shortcut interventions remain. Advanced variants can configure retained fractions, classes, area and candidate columns. Guided options expose blur, resolution, omission, localization, wrong-class, noise and underexposure. Candidate difficulty is not assumed.

Interventions retain validation/test image and annotation bytes; record seed, parameters, removed training population, source/output hashes, object counts and software versions. Noise is a seeded Gaussian **achromatic** perturbation (one draw per pixel broadcast to RGB), clipped/rounded to 8-bit. Selection interventions can alter size/class/scene distributions together; count-matched controls and measured covariates are required before causal interpretation. The existing matched architecture/initialization/budget/test protocol and repeated-seed experiment path remain in use. No training is started automatically.

Actual evaluation additionally records operating-point recall by **ground-truth native box area**, plus unmatched false positives separately binned by predicted box area. `recall_small_native`, `recall_medium_native`, `recall_large_native` become research outcome options. Zero denominators are null. Small-object AP is explicitly unavailable; these values do not reproduce official COCO small-object AP. Standard actual mAP50, mAP50–95, per-class AP, precision, recall and false positives are retained.

Research analytics excludes unavailable nested evidence and sampling/configuration constants from candidate predictors. Existing equal/adaptive arithmetic/geometric baselines remain research-only. A conclusive binary `baseline.critical_gates_pass` permits examining gating separately, without replacing six dimensions or masking a failed gate. The dashboard/API explicitly marks empirical validation pending when no actual outcomes exist.

See [research protocol and validation checklist](research-metrics-validation-protocol.md), [implementation log](research-metrics-development-log.md) and [actual test results](research-metrics-validation-results.md).
