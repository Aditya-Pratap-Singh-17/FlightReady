# UAV research assessment extension

## Architecture and compatibility

`metrics.REGISTRY`, `CALCULATIONS`, and `run_metric` remain the public metric
interface. `metrics/uav_metrics/registry.py` registers the stable family IDs `UAV-01`
through `UAV-20`; `GET /api/metrics` exposes the entire machine-readable registry.
The legacy IDs retain their values and interfaces. The new family results reuse
the legacy annotation, class, duplicate, leakage, and reference implementations.

Added/changed source:

| Files | Responsibility |
|---|---|
| `backend/context.py` | Additive measurement status, provenance, scope, units, model metadata, learning views; confirmed UAV parameters and performance objectives |
| `metrics/core.py`, `backend/readiness_engine.py` | Shared registry/runner, exact structured-value selection, contextual selection, explicit acceptance of estimated evidence |
| `metrics/uav_metrics/registry.py`, `assessment.py` | Definitions and deterministic family calculations |
| `metrics/uav_metrics/vision.py` | Optional offline local ONNX encoder/classifier adapter, bounded samples/batches and content/model/configuration cache |
| `metrics/uav_metrics/learning.py` | Six derived analytical views; conditional nuisance associations and evidence-linked recommendations |
| `backend/image_ingestion.py`, `backend/universal_images.py`, `backend/large_ingestion.py` | Additional bounded pixel summaries, protected image metadata, staged-source lifecycle for worker-side crops |
| `backend/context_extractor.py` | Conservative weather/viewpoint/tracking proposals, still requiring confirmation |
| `backend/reports.py`, `frontend/src/main.tsx`, `main.py` | JSON/HTML/UI evidence status and learning views; registry endpoint |
| `backend/research/variants.py`, `experiments.py`, `__main__.py` | Independent opt-in intervention, feature, training, and experiment export workflow |
| `tests/test_uav_metrics.py`, `tests/test_research.py` | Synthetic known-value tests and experiment invariants |

The public schema adds optional/defaulted fields. Existing clients can continue to
read `state`; updated clients should also read `assessment_status`.

| Measurement status | Existing `state` |
|---|---|
| measured / estimated | warning: diagnostic, no acceptance threshold |
| insufficient_evidence / unsupported | insufficient_evidence |
| not_applicable | not_applicable |
| error | error |

Requirement findings keep the original pass/fail semantics. A computed family is
not automatically a passed requirement. A confirmed requirement can select a
structured scalar using `parameters.output_path`; its denominator and unit come
from the selected component when present. Estimated evidence requires explicit
`parameters.accept_estimated_evidence=true`. `READY` means confirmed measurable
requirements passed, without establishing model accuracy, generalization, sample
adequacy, or deployment safety. Missing optional diagnostics do not veto READY.

All families are listed for image/UAV/telemetry assessments, with applicability
and missing evidence reported individually. Ordinary tabular classification keeps
its existing diagnostic selection unless UAV parameters are explicitly provided.
New UAV summaries return `unsupported` for disk tables: their 100-row preview is
never misrepresented as full evidence. Existing disk metrics still query all rows.
Raw arbitrary upload metadata is not copied into reproducibility output. Source
descriptions are represented by presence and a hash; private source locations and
credentials are not needed to establish provenance availability. Deliberately
provided task text, category names and relevant declared units remain report data.

## Property map, algorithms, units and inputs

For finite numeric observations, summaries return n, minimum, 25th/50th/75th
percentiles, maximum and arithmetic mean (NumPy's default linear quantiles).
Missing and invalid numeric counts are separate. Category fractions use the
nonmissing category count; population and missing counts are explicit. No
histogram acceptance boundaries are supplied by the application.

| Property / actual metric ID | Implemented definition / input / limits |
|---|---|
| Lighting / **UAV-01** | Per-image mean gray luminance and population gray standard deviation, in 8-bit gray levels; category frequencies from declared lighting and optional classifier. Brightness cannot establish weather or capture time. |
| Weather / **UAV-02** | Counts/fractions of nonmissing recorded `weather`; optional configured classifier category counts with uncalibrated softmax/entropy summaries. No brightness-based weather inference. |
| Time / **UAV-03** | Counts of declared `time_of_day`, otherwise recorded timestamp UTC hours. Parse errors and absent records reported. Naive timestamps are interpreted as UTC and must be verified; UTC hour is not solar time. |
| Seasons / **UAV-04** | Declared `season` categories; otherwise timestamp month with explicit north/south hemisphere. North DJF/MAM/JJA/SON maps to winter/spring/summer/autumn; south shifts two seasons. This is a declared meteorological-calendar assumption. No visual season estimator is shipped. |
| Scenes / **UAV-05** | Declared `scene` categories; optional scene classifier, or global embedding dispersion `mean(1 - normalized_embedding dot normalized_centroid)`. No inference that global image variation is task-useful background variation. |
| Viewpoint / **UAV-06** | Declared `viewpoint` category counts, or finite `camera_pitch` distributions with explicit angular units; calibration presence reported. Precise visual angle inference is unsupported. |
| Altitude / **UAV-07** | Finite declared altitude quantiles and explicit half-open bins; units required and datum reported, including unspecified datum. No scale-to-altitude inference. |
| Scale / **UAV-08** | Normalized `(cx,cy,w,h)` on decoded image `(W,H)`: width `wW` pixels, height `hH`, area `whWH` pixel², aspect `wW/(hH)`, relative occupancy `wh`. Per-class summaries and explicit user bins. Only retained valid annotations on readable images. |
| Appearance / **UAV-09** | Optional annotated crop embeddings; per-class mean cosine distance to normalized centroid and nearest within-class cosine similarity as a redundancy proxy. Crops round outwards to pixel boundaries. Full sample and model scope recorded; high dispersion does not prove usefulness. |
| Occlusion / **UAV-10** | Per-class annotated occlusion code counts and missingness. Native VisDrone: 0 none, 1 partial, 2 heavy. Missing is unknown. No occlusion estimator is shipped. |
| Truncation / **UAV-11** | Per-class annotated truncation plus a separate border-contact proxy. Contact when minimum distance of a box edge to the image border <=1e-9 normalized units (geometry tolerance). Contact is not true truncation. |
| Clutter / **UAV-12** | Gray thumbnail histogram entropy `-sum(p log2 p)` bits plus a reference to UAV-13 density. Semantic background clutter remains unverified; object pixels are not masked. |
| Density / **UAV-13** | Counts per fully annotated readable image (including explicit negatives), counts per megapixel and per-class counts/image. Pair IoU `intersection/union` and center distance/image diagonal. Compare an explicit ordered prefix up to `max_spatial_pairs`; report total possible pairs. Only user-supplied sparse/dense bins. |
| Clarity / **UAV-14** | Resolution; luminance/contrast; population variance of four-neighbor Laplacian on thumbnail interior; encoded bytes/image pixel. Aspect-preserving gray thumbnail <=256px longest edge, Pillow default resize. Bytes/pixel is not a compression-quality estimate. No universal blur threshold. |
| Annotations / **UAV-15** | Reused `bbox_invalid_rate`, `missing_label_rate`, manifest/policy evidence, and optional independently declared reviewed/incorrect-object counts. Audit error fraction = incorrect/reviewed in the reviewed population. Syntax cannot establish semantics or exhaustive visible-object labeling. |
| Discrimination / **UAV-16** | Optional crop embeddings: nearest cross-class cosine similarity for each sampled object and top review candidates. Class labels come from annotations, not predictions. Candidate confusion is neither semantic error nor proven training benefit. |
| Representativeness / **UAV-17** | Reused reference `reference_comparison` for selected numeric/categorical columns; W1 in matching feature units and base-2 JS divergence in bits. Explicit category-weight references normalized to probabilities; missing categories identified. Optional Euclidean distance between means of normalized image embeddings. Equal centroids do not imply equal distributions; no predicted accuracy degradation. |
| Independence / **UAV-18** | Reused class counts, SHA256 duplicate fraction (repetitions beyond first/n images), configured 64-bit dHash candidates, content split overlap, group overlap, and evidence summary. Counts known flight/session groups and missing IDs. Sample adequacy remains unverified. |
| Sensors / **UAV-19** | Declared RGB/thermal/GPS/IMU/multisensor field presence/missingness, declared units versus confirmed expected units; calibration/alignment presence only. Paired timestamp absolute differences in seconds; within-tolerance counts only when a tolerance is declared. Other sensor-specific physics unsupported. |
| Temporal / **UAV-20** | Within declared sequence in recorded order: duplicate timestamp count (beyond first), negative adjacent time intervals, interval quantiles, adjacent frame-index jumps >1/nonincreases, duplicate `(frame,track)` records. Missing timestamps are not bridged. Static images are not applicable unless explicitly declared temporal. Identity correctness, calibrated physical motion, frame-level video decoding remain unsupported. |

The original registry also exposes all underlying diagnostic IDs; these family
results do not replace them. All metric records include method, sample size,
dataset scope/hash, provenance, limitations, implementation version and optional
estimation metadata. The registry declares applicability, tasks/modalities,
required inputs, unit, calculation method and implementation status. Some
components within an otherwise measured family remain unavailable or estimated;
read component status and provenance rather than only the top-level status.

## Dataset and contextual input conventions

Existing YOLO, VisDrone DET, COCO and Pascal VOC ingestion/validation are retained.
YOLO boxes are normalized center/size; VisDrone/VOC/COCO coordinates are normalized
by their existing parsers after validation. ImageFolder and unlabelled images are
distinct from detection datasets. No visible-object ground truth is invented.

For images, include a `metadata.csv` next to the selected image collection root,
keyed by exact complete relative archive `image` path. For standard YOLO and
unwrapped COCO this is archive-root `metadata.csv`; wrapped YOLO/VisDrone uses its
dataset root. COCO/VOC collections also accept an explicit `image_root` prefix
when selecting a layout. Metadata is <=8 MiB and cannot replace computed fields.
Missing rows remain missing metadata. Example:

```csv
image,weather,lighting,scene,viewpoint,timestamp,altitude_meters,flight,split
images/train/a.png,clear,daytime,road,nadir,2026-01-01T12:00:00Z,50,flight-a,train
images/test/b.png,rainy,daytime,urban,oblique,2026-01-02T12:00:00Z,80,flight-b,test
```

Upload-level metadata JSON carries declarations such as:

```json
{
  "dataset_version": "collection-v1",
  "units": {"altitude_meters": "m", "camera_pitch": "deg"},
  "altitude_datum": "height above ground, declared by collector",
  "collection_kind": "static_images",
  "sensor_types": {"rgb": ["image"]}
}
```

Use ISO8601 timestamp strings with explicit offsets, not unspecified numeric epoch
values. `collection_kind=video` or `telemetry` enables temporal checks on exported
frame/telemetry records, not raw video decoding. For independent annotation audits,
upload `annotation_audit` with `independent`, `reviewer`, `protocol`,
`reviewed_objects`, and `incorrect_objects`. This is a declared independent audit,
not independently authenticated audit provenance.

Confirm optional `Goal.uav_parameters` in the existing Goal JSON editor:

```json
{
  "columns": {"altitude":"altitude_meters", "timestamp":"timestamp", "sequence":"flight"},
  "hemisphere":"north",
  "scale_bins":{"relative_occupancy":{"declared_small":[0,0.01],"other":[0.01,null]}},
  "altitude_bins":{"declared_envelope":[20,100]},
  "density_bins":{"declared_sparse":[0,5],"other":[5,null]},
  "reference_columns":["altitude_meters","weather"],
  "reference_distributions":{"weather":{"clear":0.5,"rainy":0.5}},
  "nuisance_columns":["scene","lighting","source"],
  "invariance_columns":["lighting","weather","viewpoint"],
  "max_spatial_pairs":10000,
  "compare_reference_embeddings":false
}
```

**The numbers above are illustrative user criteria, not recommended scientific
thresholds.** Numeric bins are `[lower,upper)`, with `null` for open endpoints;
overlapping bins are allowed and do not imply fractions that sum to one.
`reference_columns` require a separately uploaded reference dataset with matching
numeric units. Reference distribution authenticity/relevance remains declared.
`target_environment` and scenarios retain their existing binding requirement
semantics; the distribution comparisons are diagnostic unless explicitly bound
to a confirmed requirement. Qualitative performance objectives are recorded but
never established by data metrics alone.

## Six derived views

The certificate `learning_views` references the same property IDs, without a
composite score: concept coverage, invariance support, shortcut vulnerability,
discriminative information, independence/redundancy, and deployment-domain shift.

Shortcut analysis builds an image-level contingency table between each target
class's binary annotated presence and each selected nuisance category. Explicit
negatives count as absence; missing/invalid annotations do not. It reports
`MI / sqrt(H(nuisance)*H(class_presence))` and the complete contingency table;
constant variables yield undefined (`null`) normalized MI. At most 100 target
classes and 100 nuisance categories are supported; declare coarser categories or
select fewer classes otherwise. This describes association, not learned model use.

Invariance support counts distinct annotated images per class/observed condition.
Marginal condition diversity does not prove identity-preserving variation.
Generated transformations are not assumed label/visibility preserving; independent
review is required. Contrastive negative mining, physical invariance guarantees,
and target-model representation attribution are not established by these views.

## Optional local vision

No vision dependency is required for deterministic assessment. The adapter accepts
locally obtained, trusted ONNX exports of an image encoder (e.g. an appropriately
pooled DINOv2/CLIP encoder) or a separately justified scene/weather/lighting
classifier. No pretrained models or ONNX exports are bundled or validated here.
Native Hugging Face downloads and external image APIs are not implemented.

Install either `onnxruntime` for CPU or the compatible CUDA-enabled runtime in your
own configured environment. Follow the [official ONNX Runtime API](https://onnxruntime.ai/reading/api/python/api_summary.html)
for supported providers. Place a server-owned JSON file outside uploads and set
`UAV_VISION_CONFIG` to its path. The upload cannot select a model or enable inference.

```json
{
  "enabled":true, "device":"cpu", "batch_size":8,
  "max_images":256, "max_objects":512, "seed":17,
  "cache_dir":".runtime/vision-cache",
  "models": {
    "encoder": {
      "path":"C:/DataReadiness/local-models/encoder.onnx",
      "sha256":"REPLACE_WITH_ACTUAL_FILE_SHA256",
      "name":"your-encoder", "version":"your-export-version",
      "image_size":224, "mean":[0,0,0], "std":[1,1,1], "output_index":0
    }
  }
}
```

Configure the actual model's preprocessing; zeros/ones above only illustrate the
contract. One RGB NCHW float32 input and finite NxD output, D<=4096, are required.
For classifiers, use `weather`, `scene` or `lighting` instead of `encoder` and add
an ordered `labels` list matching output logits. Scene/classifier exports requiring
text inputs, special tokens or model-specific pooling need an independently
prepared compatible export; they are not automatically adapted. Classifier
softmax values and predictive entropy are **uncalibrated**, not probabilities of
correctness or statistical confidence intervals.

Missing metrics/config/runtime/model -> insufficient evidence. Invalid hashes/shapes or
execution failures -> error, logged locally without model paths in result reasons.
CUDA is requested only explicitly and falls back to CPU when unavailable. Batches
are 1..32, resize 8..1024, each sample <=2048 images/objects. Uniform sampling uses
the configured seed. Cache keys include original content hash, box, model hash,
model version, preprocessing, runtime/provider configuration and sampling config.
Model paths are excluded from public visual provenance. Caches hold derived
features locally; manage their retention deliberately and never publish them as
raw independent ground truth. Calls run inside the existing assessment executor,
not the HTTP request thread; cancellation is checked between inference batches.

## Resources and complexity

Original archive, upload, decompression, image pixel, member and object-count limits
remain active. Image inspection decodes one image at a time, <=20 megapixels;
new entropy/Laplacian intermediates are <=256². Uploaded image sources remain in
private staged storage for optional later crops and are deleted with dataset
lifecycle cleanup. This adds disk retention up to the configured upload limit;
there are at most ten uploaded datasets per local session.

Deterministic summaries are O(images + objects), apart from per-class summaries
and quantile sorts. Geometry retains O(objects) numeric values, bounded by existing
MAX_ANNOTATED_OBJECTS (default 1,000,000); large annotation sets may require
hundreds of MiB in addition to the existing ingestion tables/objects. Crowding is
limited to 100,000 requested pairs (default 10,000), with ordered prefix scope.
Existing near-duplicate dHash comparisons retain their configured image-count
cap (default 2,000); above it they are explicitly skipped, not approximated.

Visual memory is O(batch*3*size² + sampled_objects*D + model), in addition to one
decoded image; encoder/model memory depends on the supplied export. Nearest
cross-class search is exact **within the bounded sample**: O(S²D) time and O(SD)
memory, S<=2048, not an all-dataset exhaustive similarity search. Cache disk usage
grows with distinct content/configuration combinations; no automatic cache
retention policy is imposed. No multi-GiB archive or real GPU stress benchmark
was invented or newly run for this extension.

## Run application and API

From PowerShell in the repository, use the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Use the existing React development server or build with `npm.cmd run build` in
`frontend`; see the main README. Keep the service bound to loopback.

`backend/scripts/uav_api_example.py` demonstrates upload -> interpret -> explicit human
review -> confirmed context -> queued assessment -> JSON/HTML export. It pauses
after interpretation to produce a reviewable context file. Run again only after
editing and explicitly confirming it. The example never enables remote vision
or a language model by itself. The backend may use its existing configured text
interpreter; uploaded data and metadata are never sent to that provider.

```powershell
.\.venv\Scripts\python.exe backend/scripts/uav_api_example.py prepare --dataset demo_data/detection_valid.zip
# Review uav-api-example/context.json and explicitly confirm it.
.\.venv\Scripts\python.exe backend/scripts/uav_api_example.py assess
```

The equivalent confirmed manual assessment request is:

```json
{
  "dataset_id":"REPLACE_WITH_UPLOAD_ID",
  "context":{
    "original_text":"Detect annotated cars in UAV images",
    "confirmed":true,
    "goal":{
      "task_type":"object_detection", "input_modalities":["image"],
      "required_classes":["car"], "split_column":"split", "group_column":"flight",
      "requirements":[{
        "requirement_id":"USER-SIZE",
        "description":"Illustrative confirmed median occupancy criterion",
        "metric_id":"UAV-08", "operator":"gte", "expected_value":0.01,
        "parameters":{"output_path":["per_class","car","relative_occupancy","median"]}
      }]
    }
  }
}
```

POST it to `/api/assessments`, poll `/api/assessments/{id}/progress`, then GET
`/api/assessments/{id}/certificate` or `/export?format=json|html`. The illustrative
threshold is not a scientific default. A fully computed synthetic example report
is under `reading/examples/uav-synthetic-report.json` and `.html`; it is software
demonstration evidence only, not a UAV benchmark or evidence of accuracy.

## Controlled experiments

Research commands never run as part of readiness assessment. Canonical root YOLO
ZIPs with `classes.txt`, `images/...`, `labels/...` and complete split assignments
are currently supported. Explicitly empty annotations remain negatives. Every
image must have a label file. Include `val` and `test`, flight/session/source IDs
where available, and keep groups independent between splits. If IDs are genuinely
absent, configuration must explicitly acknowledge unverified group independence.
Exact image overlap across splits is rejected. Perceptual/semantic overlap still
requires independent review.

Example variant configuration (the numbers are experimental choices):

```json
{"seed":17,"intervention":"annotation_noise","parameters":{"probability":0.1},
 "target_train_images":100,"group_column":"flight"}
```

```powershell
.\.venv\Scripts\python.exe -m research variant parent.zip experiments/noise-seed17 variant.json
.\.venv\Scripts\python.exe -m research assess experiments/noise-seed17 confirmed-context.json
.\.venv\Scripts\python.exe -m research export experiment-table.csv experiments/noise-seed17
```

Each new directory contains `dataset.zip`, `manifest.json`, and (after assessment)
`readiness-features.json`. Original datasets are never overwritten. The manifest
records parent/variant SHA256, intervention, seed, every train copy/change/removal,
evaluation image/annotation hashes, group-independence evidence and sample counts.
ZIP timestamps are fixed so identical input/configuration produces identical
variant bytes. Failed jobs retain a partial directory without a successful
manifest; inspect it and choose a new output directory before retrying.

Supported interventions:

| Intervention | Parameters and operation |
|---|---|
| control | Uniform whole-image training selection to explicit matched count |
| annotation_noise | `probability` in [0,1]; replace selected training class IDs with a different declared class, preserving geometry |
| object_scale | `relative_area_range: [low,high]`; retain images containing a matching annotated object |
| redundancy | `unique_images`; sample unique training sources, fill matched count with copied image/annotation pairs |
| class_representation | `classes`; retain images containing selected classes, preserving other co-occurring object labels |
| environment / viewpoint | `column`, `allowed_values`; restrict complete declared condition metadata |
| shortcut | `column`, `class_condition_map`; restrict images whose target-class/nuisance associations match the declared mapping |
| image_degradation | `blur_radius`, `brightness_factor`; bounded Gaussian blur/brightness intervention on training images only |

Scale/class/condition restrictions operate on **whole images** to avoid creating
false incomplete annotations. They can change co-occurring distributions, which
must be measured as confounders; they do not isolate one causal property by fiat.
Shortcut interventions use selection, not fabricated backgrounds. Compare multiple
seeds and matched sizes/budgets, and inspect collateral distribution changes.

Optional detector runner: install and pin your chosen compatible `ultralytics`
package separately, acquire a trusted local `.pt` checkpoint, record its actual
SHA256 and version, and configure:

```json
{
 "model_family":"ultralytics-yolo", "model_version":"YOUR_MODEL_VERSION",
 "package_version":"YOUR_INSTALLED_VERSION",
 "checkpoint":"C:/DataReadiness/local-models/detector.pt",
 "checkpoint_sha256":"ACTUAL_SHA256", "seed":17,
 "epochs":50, "batch":8, "image_size":640, "device":"cpu",
 "workers":0, "optimizer":"SGD", "learning_rate":0.01,
 "condition_subsets":{}
}
```

Only an explicit command launches training:

```powershell
.\.venv\Scripts\python.exe -m research train experiments/noise-seed17 training.json --execute
```

This stages only validated image/annotation files into a new training directory;
data YAML is generated locally without download directives. Missing local weights
or package-version/hash mismatch stops the run. AMP/plots/cloud sync are disabled,
offline intent is set, seeds/deterministic algorithms and training budget are
explicit. The runner uses the optional library's documented
[training configuration](https://docs.ultralytics.com/usage/cfg/) and
[validation metrics](https://docs.ultralytics.com/modes/val/). Operator-provided
dependencies/checkpoints still need their own local trust and license review.
No detector was downloaded or trained for this implementation.

`experiment.json` stores real validation/test mAP50, mAP50-95, precision, recall,
measured per-class AP indices, optional named unchanged test subsets, checkpoint
hashes and environment/package details. Absent-class overall-AP placeholders are
not exported as per-class ground truth. CSV export includes training-only readiness
features, intervention/model/training configuration, seeds, sample counts and
condition results. Before training, performance fields remain null/empty.

## Independent scientific validation still required

`reading/vision-evaluation-protocol.md` specifies an independent annotated evaluation
procedure for optional estimates. No prediction is its own ground truth. Need to
validate metadata reliability, annotation semantics/completeness, encoder/classifier
domain validity and uncertainty, transformation visibility/label preservation,
sample independence, reference deployment relevance, and real downstream detector
effects. Association, diversity and readiness are not guarantees of generalization.
