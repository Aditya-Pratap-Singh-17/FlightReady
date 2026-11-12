# FlightReady: connected readiness and model validation

Implemented in the existing repository. The existing parsers, metric registry,
quality policies, saved records, training queue and research tools are retained.
The preimplementation audit is in `flightready-audit.md`.

The model importer has since been repaired and expanded. See
[Model Lab import repair](model-import-fix.md) for repository-ID/URL imports,
local model folders, prediction previews before readiness, and the actual
pretrained run on the user's saved VisDrone dataset.

## Run locally

Use the project's existing environment and `WORKBENCH_ROOT` to retain the same
saved projects. Restart the backend after updating the code.

```powershell
Set-Location C:\DataReadiness
# Existing assessment/report dependencies:
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements.txt
# Actual YOLO training and its existing evaluator:
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements-training.txt
# Optional safe YOLOS inference and COCO evaluator:
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements-inference.txt
Set-Location frontend
npm.cmd install
npm.cmd run build
Set-Location ..
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Bind locally: this remains a single-user research
application, without a multi-user authorization layer. No model download occurs
on import, registry listing, compatibility preview or ordinary training.

## Three-stage workflow

1. **Dataset & goal**: import a supported dataset, describe the task, inspect the
   interpreter's provenance and review the interpretation. Compatibility is
   checked immediately using the existing class, modality, label and scenario
   measurements. Confirm the goal to save it as an immutable context record.
2. **Readiness report**: assess either the complete dataset or a training
   partition. The canonical executive report provides one conservative verdict,
   coverage, critical blockers, three priority findings and recommendations.
   Dimension cards and the evidence tabs expose all original diagnostics,
   UAV-01 through UAV-20, normalization, weights, populations and limitations.
   Research detail adds method identities and incomplete-index bounds.
3. **Model Validation Lab**: choose a controlled training comparison or an
   inference-only benchmark. The dataset and assessment IDs carry forward.
   Actual outcomes appear as a separately linked report supplement. The frozen
   analytical report and its hash never change when a model finishes.

### The VisDrone distinction

Native VisDrone DET classes describe ground objects viewed from a drone. They do
not include drone targets. The goal "detect drones at night" produces a confirmed
label mismatch and, separately, unknown nighttime evidence if lighting metadata
is missing. Darkness/brightness proxies do not establish capture time.

Official native wrapper roles such as `VisDrone2019-DET-val` are preserved as
partition evidence. Validation/test downloads cannot be repartitioned into
training data. Label conversion supports native VisDrone and VOC, in addition to
existing YOLO and COCO formats. Evaluation uses ordinary canonical boxes; it
does **not** reproduce official VisDrone ignore regions or COCO crowd semantics.
Published leaderboard numbers are therefore not assumed comparable.

For a training-only source, use **Workspace tools > Model Training > Partition
purpose > Training only**, then supply a **separate validation dataset**. A final
test dataset is supplied separately in the experiment protocol. This explicit
mode cannot repurpose declared validation/test images. Default split behavior
and all exact/near-duplicate and flight/session safeguards remain in force.

## Mathematical contract and versioning

Raw metric definitions and implementation versions remain in the registry:
`GET /api/metrics` and `GET /api/v1/quality/registry`. Each assessment freezes
selected metric IDs, definitions, evidence, raw values, denominators, processing
configuration, dependency versions and a dataset/context fingerprint.

The existing `geometric-gates-1.0` policy remains unchanged. Its six dimensions
are annotation quality; diversity and required coverage; signal and image
quality; distribution and sample adequacy; task and operational suitability;
and integrity and reliability. Applicability excludes legitimate inapplicable
criteria. Positive applicable metric weights normalize to one within each
dimension. A dimension index is:

```text
D = 100 * product((score_m / 100) ^ weight_m)
```

Actual zero scores remain zero. Unknown evidence retains its assigned weight.
The primary dimension score is withheld unless all positive applicable criterion
weight is measured. Technical validity does not establish semantic label
correctness: independent annotation audit evidence remains a separate criterion.
Clarity thresholds, warning cutoffs and default task/scenario priorities are
documented engineering choices, not empirically learned optimal criteria.

The new `final-readiness-1.0` snapshot adds conservative bounds on that existing
geometric index. If any known positive-weight component is zero, both bounds
are zero. Otherwise the upper bound sets unknown component scores to 100. The
lower bound is zero when positive weight is unknown; with complete evidence,
the bounds coincide with the observed index. These are identification bounds on
a defined mathematical index, **not** statistical confidence intervals.

Evidence coverage is `sum(dimension importance * measured criterion weight)`.
Coverage is distinct from measurement quality, independent provenance and
statistical confidence. Evidence strength remains explicitly uncalibrated;
inspect the method and provenance of each measurement rather than inferring
confidence from coverage.

There is deliberately **no overall numerical readiness index** under this
policy. The canonical report labels it "Not scored" and explains why. The
existing experimental weighted score remains available in the appendix, with
its provisional/partial labels; it is never the final verdict or a training
performance prediction. Its missing default config was restored from the
already documented eight-component preset, under the explicit version
`experimental-engineering-1.0-restored`. Historical stored profiles stay intact.

Hard requirements are evaluated independently of weights. The canonical report
returns `NOT_READY` for any confirmed mandatory failure, `INDETERMINATE` when
mandatory evidence, task support, configured dimension coverage or acceptance
conditions remain unresolved, and `READY_FOR_DEFINED_USE` only when all these
conditions support that specific use. The original certificate's narrower
`overall_status` remains unchanged and is labeled as technical/confirmed
requirement status. A successful technical check cannot override a task blocker.

The new assessment engine version is `1.6.0`. New certificates add
`task_compatibility` and `final_report`; old certificates validate with these
fields absent. Old reports retain their original conclusions. Saved alternative
aggregations remain distinct artifacts and do not reuse the source canonical
verdict under a changed policy.

## Canonical report and exports

The same frozen `final_report` drives the frontend, HTML and PDF:

- A: verdict, dataset version/hash, intended use, compatibility and coverage;
- B: six dimension indices or explicit unscored states;
- C: confirmed failures and missing evidence;
- D: ranked actions, ordered by mandatory failure, mandatory evidence gaps and
  configured task importance; effort is qualitative and not fabricated;
- E: validation status at assessment time;
- F: complete methods, provenance, metrics, policies and limitations.

`GET /api/v1/reports/{assessment_id}?format=json|html|pdf|csv` preserves the
existing JSON record envelope and CSV metrics. HTML/PDF retain the same report
version, assessment fingerprint and report hash. PDF leads with a one-page
executive summary, followed by the evidence appendix. Structured appendix values
may be abbreviated in PDF with explicit notice; JSON retains the complete data.

`GET /api/v1/assessments/{assessment_id}/validation` is a **live supplement** with
linked inference benchmarks, model evaluations, experiments, actual same-seed
performance differences and scientific interpretations. The immutable exports
retain their original validation state. Export each experiment or benchmark to
capture later empirical outcomes and reproducibility metadata.

## Safe model acquisition and inference

The initial remote registry allowlists `hustvl/yolos-tiny`, pinned to commit
`95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de`. It declares COCO labels, Apache-2.0,
approximate download size, supported operations, CPU/CUDA requirements and source.
Only `config.json`, `preprocessor_config.json` and `model.safetensors` are fetched
using Hugging Face's supported snapshot API. Download requires explicit consent
and license acceptance. **Load cached revision without network** uses
`local_files_only=True`; it still requires license review.

The adapters use concrete built-in Transformers YOLOS, DETR and RT-DETR classes, local-only model
loading, `use_safetensors=True`, and `trust_remote_code=False`. Repositories for
these built-in detector families can be inspected and imported by ID or URL.
Remote Python code, pickled downloads and custom architectures are not accepted.
Architecture and label schema are validated; each file hash is stored
and checked before execution. Download failures leave no active acquired model.

Goal-validation benchmarking requires an exact dataset/assessment match and
task support. Annotation-only benchmarking can instead evaluate an explicitly
selected label scope without asserting that the project goal is validated.
Selected labels require detection ground truth and one-to-one source-to-model
mapping. Prediction previews need readable images, not an assessment or matching
labels, and produce no detection-quality scores. A mapping cannot convert a COCO
bird/airplane detector into a drone-target detector. A benchmark source without a recorded evaluation role requires explicit
acknowledgement of whole-dataset benchmarking. A known split selects only its
held-out test images. Pretrained development-set overlap remains unverified and
is reported as a limitation, not passed independence.

Metrics use established `pycocotools` bounding-box evaluation: IoU thresholds
0.50:0.05:0.95, 101 recall positions, maxDets=100, per-class AP and native-area
small/medium/large AP. Operating-point precision, recall, TP/FP/FN use the existing
class-aware confidence-ordered matcher. Undefined denominators remain null.
Only the declared mapped classes are scored; unrelated pretrained categories
are explicitly out of scope. Visual sample review shows ground-truth solid
boxes, prediction dashed boxes, false positives and missed labels.

Preprocessing size, confidence/matching thresholds, safe model revision/hashes,
dataset/test hashes, actual predictions, dependencies, adapter source hash and
pipeline latency are persisted. Pipeline latency includes preprocessing and
postprocessing and is labeled separately from model-only inference timing.

Endpoints: `GET /model-registry`, `POST /model-registry/acquire`,
`GET /acquired-models`, `POST /inference-benchmarks`,
`GET /inference-benchmarks?readiness_id=...`, and
`GET /inference-benchmarks/{id}/predictions` (under `/api/v1`). Acquisition and
inference use the existing durable serialized subprocess queue, logs,
cancellation and restart recovery.

## Controlled experiments and scientific interpretation

Existing YOLO11n/s training remains real and local, with CPU fallback, CUDA
detection, recorded initialization hashes, optimizer/budget/preprocessing/seeds,
checkpoints, failures and resume. Controlled interventions modify training data
only. Models share checkpoint-selection validation data and an independent
final test. The existing experiment API supports repeated seeds and matched
sample-count protocols; removal of examples is not presumed beneficial.

The new scientific interpretation joins same-seed baseline/candidate outcomes
only when test identity and evaluation settings match. It reports actual
baseline/candidate values, absolute changes and relative changes (undefined when
the baseline is zero), checks training protocol/sample-count differences, and
withholds numerical dimension changes for unknown scores or different policies.
Experiment HTML/PDF and the live assessment supplement include this conclusion.
Existing research analytics retains independent-source checks, seed aggregation,
uncertainty where supported, exploratory correlations and sensitivity analysis.

## Verified behavior and limits

The verification record is in `flightready-verification.md`.

This implementation supports detection execution first. Classification,
regression, anomaly, tracking and telemetry retain their existing readiness
analysis; they have no new model-training adapters. YOLOS acquisition/inference
does not add YOLOS fine-tuning; train/fine-tune through the supported YOLO adapter
and trusted local checkpoints. No hosted arbitrary-code model execution is added.

After explicit user consent, YOLOS Tiny weights were downloaded and real
inference was run on the saved 548-image VisDrone validation dataset. The
annotation benchmark explicitly scores bicycle, car, truck and bus; excluded
classes, ignore-region limitations and the unvalidated drone/night goal are
recorded. See `model-import-fix.md` for the actual results. Local tiny YOLOS,
DETR and RT-DETR fixtures also verify each built-in adapter. CUDA was not tested. Independent audit
authenticity and verified nighttime subset performance require additional actual
evidence. No readiness-to-accuracy calibration, statistically significant benefit
or causal conclusion is established by these software tests.

Primary API/model references:
[Hugging Face downloads](https://huggingface.co/docs/huggingface_hub/guides/download),
[pinned YOLOS revision](https://huggingface.co/hustvl/yolos-tiny/commit/95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de),
[COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
