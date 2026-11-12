"""Curated explanations for the source-derived guide, not an application module."""

FILE_ROLES = {
'backend/context.py':'Shared Pydantic contracts: goals, requirements, evidence, findings and certificates. The base Model forbids extra fields and disallows nonfinite values in validated numeric fields.',
'backend/context_extractor.py':'Deterministic text/tag extraction and optional Gemini or OpenAI-compatible REST extraction. Local validation and human confirmation remain final.',
'backend/ingestion.py':'Dataset dataclass, ingestion exceptions, strict CSV decoding and the smaller in-memory parsing path.',
'backend/large_ingestion.py':'Bounded upload staging, CSV/Parquet/record batching, whole-table DuckDB profiles and large-data dispatch.',
'backend/record_formats.py':'Stream-oriented adapters for delimited text, plain text, JSON/JSONL, XML, spreadsheets and Arrow records.',
'backend/dataset_archives.py':'Read ZIP/TAR/compressed containers, discover candidate datasets and require explicit choices for ambiguous contents.',
'backend/image_ingestion.py':'Validate archive members, decode image evidence, parse YOLO/VisDrone annotations and merge declared image metadata.',
'backend/universal_images.py':'COCO/VOC/ImageFolder/YAML-YOLO/unlabelled image adapters; maps source geometry into canonical normalized boxes.',
'metrics/dataProfiler.py':'Dataprofiler/DataProfiler and DataProfile: basic observed statistics, missingness, duplication and descriptive ranges.',
'metrics/featureAnalyzer.py':'FeaturAnalyzer/FeatureAnalyzer and FeatureAnalysisResult: constant/dominant/redundant feature and target-copy heuristics.',
'metrics/labelAnalyzer.py':'LabelAnalyzer and LabelAnalysisResult: label counts, missing labels, imbalance and optional paired-label agreement.',
'metrics/IsoIecChecker.py':'Retained profile-only checker. Several properties deliberately return None; it is not the active readiness dispatcher.',
'metrics/core.py':'One registry and run_metric dispatcher for requirement checks, diagnostics, disk checks and registered UAV properties.',
'metrics/disk_metrics.py':'Whole-data SQL implementations of supported tabular metrics. Does not compute statistics from the 100-row preview.',
'backend/readiness_engine.py':'Compiles Goal into requirements, executes checks, decides requirement readiness and assembles the Certificate.',
'metrics/quality_profiles.py':'Validated adaptive policy schema, file loading and deterministic configuration hashes.',
'metrics/quality_aggregation.py':'Component normalization, applicability, evidence coverage, within-dimension geometry, independent gates and adaptive status.',
'metrics/experimental_scoring.py':'Older optional scalar summary. Computes an observed-component arithmetic mean, coverage and possible range; it does not decide readiness.',
'backend/reports.py':'Escaped deterministic HTML for a saved Certificate, including adaptive profile charts.',
'main.py':'FastAPI startup, body-limit middleware, original transient assessment API, router inclusion and production static hosting.',
'backend/workbench/api.py':'Persistent dataset/job/model/report API; ErrorRoute, cached services, training/evaluation submission, research joins and exports.',
'backend/workbench/contracts.py':'Pydantic request/response models for import options, splits, readiness, training, testing and variations.',
'backend/workbench/datasets.py':'Persist/reload parsed datasets and source references, register directories, expose verified image bytes and derive training-only subsets.',
'sql/store.py':'Generic SQLite JSON records through SQLAlchemy; managed artifact directories and serialized updates.',
'backend/workbench/jobs.py':'Serialized persistent queue, subprocess launch, cancellation, hard-stop recovery and graceful close.',
'backend/workbench/worker.py':'Subprocess dispatcher for readiness, training, evaluation and experiments; writes result/failure state.',
'backend/workbench/workflow_api.py':'Guided projects: saved goal conversations, confirmation pointers, assessment scope and controlled-comparison preparation/protocol.',
'backend/workbench/splits.py':'Grouping, duplicate/perceptual signatures, reproducible train/validation/test partitions and independence assertions.',
'backend/workbench/quality_api.py':'Frozen-profile reaggregation previews, saved aggregation history, task comparison and hypothetical sensitivity.',
'backend/workbench/lab_api.py':'Named experiment registration/start, trusted checkpoint import/resume and separately labelled external published benchmarks.',
'backend/workbench/experiments.py':'Validates frozen multi-mode/multi-seed protocols and serially executes child jobs in the parent worker.',
'backend/workbench/detector.py':'Real YOLO/PyTorch setup, source materialization, matched initialization, train/resume, independent evaluation and error matching.',
'backend/workbench/combined.py':'Reassesses the complete union of combined training subsets with cached source evidence and virtual source paths.',
'backend/workbench/variations.py':'Converts managed training subsets to the standalone variation format, records artificial provenance and preserves validation/test data.',
'backend/workbench/analytics.py':'Joins saved outcomes with readiness; comparability grouping, seed aggregation, correlation, held-out prediction and paired interventions.',
'backend/workbench/reporting.py':'Builds report documents from persisted evidence and renders application HTML/PDF with charts and wrapped tables.',
'backend/workbench/synthetic.py':'Explicit synthetic image/box fixture for software checks. It is not real UAV validation data.',
'metrics/uav_metrics/registry.py':'Stable UAV-01 through UAV-20 metadata and closure-based registration into the shared metric registry.',
'metrics/uav_metrics/assessment.py':'Observed/declared/estimated UAV measurements with explicit populations, denominators, provenance and unsupported states.',
'metrics/uav_metrics/vision.py':'Optional trusted local ONNX adapters, bounded visual sampling, cached sessions, embeddings and classifier evidence.',
'metrics/uav_metrics/learning.py':'Six analytical views plus conditional class/nuisance associations and evidence-based recommendations.',
'backend/research/__main__.py':'Optional command-line variant, assess, train and export workflow. Training requires the explicit --execute switch.',
'backend/research/variants.py':'Standalone immutable controlled-data variants with checksums, partition preservation and intervention manifests.',
'backend/research/experiments.py':'Standalone training configuration, variant assessment, optional execution, library metric extraction and experiment exports.',
'frontend/src/main.tsx':'Browser entry: mounts Workbench normally or the original App for legacy=1 / assessment query parameters.',
'frontend/src/Workbench.tsx':'React Query provider, Shell, primary/advanced navigation and reusable dataset, training, comparison, analytics and settings tools.',
'frontend/src/GuidedWorkflow.tsx':'Guided project restore, upload, goal chat/review, confirmation, analysis progress and readable readiness report.',
'frontend/src/GuidedValidation.tsx':'Sequential intervention preparation, measured change, protocol review, actual training monitor and experiment-scoped outcomes.',
'frontend/src/api.ts':'Persistent API fetch helper, JSON request construction, response DTO aliases and display formatting.',
'frontend/src/useAssessment.ts':'Original UI hook: restore an assessment ID, poll progress and fetch completed certificates using separate abortable effects.',
'frontend/src/QualityAnalytics.tsx':'Six-dimensional evidence cards/charts, frozen reweighting, gates, distributions, version history and sensitivity.',
'frontend/src/ExperimentalScore.tsx':'Older arithmetic scalar inspection and read-only weight previews with evidence-coverage warnings.',
'frontend/src/ExperimentBuilder.tsx':'Advanced experiment selection, frozen scoring versions, protocol review/start, import and published benchmarks.',
'frontend/src/BenchmarkCharts.tsx':'Actual per-class outcomes, precision-recall curves and confusion matrix visualizations from Evaluation DTOs.',
'frontend/src/workflow.css':'Responsive guided workflow, progress, report, chat and lab themes.',
'frontend/src/workbench.css':'Shared workbench layout, forms, tables, charts and accessibility focus states.',
'frontend/src/styles.css':'Original assessment application styling and common base styles.',
'metrics/config/adaptive_quality.json':'Six quality dimensions, component definitions, applicability, task/scenario ratios, normalization provenance and gate policies.',
'metrics/config/experimental_score.json':'Independent versioned configuration for the older provisional arithmetic scalar.',
}

CHAPTERS = [
('ch01','1. Understanding the project in five minutes'),
('ch02','2. Repository structure and responsibilities'),
('ch03','3. How the application starts'),
('ch04','4. Frontend architecture and user actions'),
('ch05','5. Backend API and interface map'),
('ch06','6. Dataset ingestion and storage'),
('ch07','7. Natural language becomes a confirmed goal'),
('ch08','8. Metrics: measurements before decisions'),
('ch09','9. Adaptive normalization, weighting and gates'),
('ch10','10. Model Validation Lab and real training'),
('ch11','11. Research analytics and statistical limits'),
('ch12','12. Database, filesystem and data lineage'),
('ch13','13. Errors, cancellation and recovery'),
('ch14','14. Three complete worked examples'),
('ch15','15. Function, class and data-model reference'),
('ch16','16. Debugging and safe extension map'),
('ch17','17. Glossary, findings and verification'),
]

TEXT = {}
TEXT['ch01'] = r'''
## What the platform does

FlightReady is a local research application for asking whether a dataset contains the evidence needed for a particular task, and then testing how measured data differences relate to model performance. A dataset is not evaluated in isolation: the user supplies a goal, such as detecting drones in airport imagery, and confirms the interpretation before measurement starts.

The normal browser workflow has three stages. First, import data and confirm the goal. Second, run the readiness analysis and inspect a report. Third, for supported detection datasets, run a controlled training comparison and inspect independent-test results. The advanced workspace exposes the same underlying tools with more configuration. The original assessment interface remains reachable through `?legacy=1` or an `assessment` query parameter.

The inputs can be tables, documents represented as records, images and supported annotated image archives. Readiness supports more modalities and tasks than detector training: the current model adapter trains YOLO11n or YOLO11s object detectors on eligible YOLO/COCO sources. It does not train classification, regression, telemetry or tracking models merely because those task policies exist.

## Three outputs that must stay distinct

| Output | Actual meaning | Owning implementation |
|---|---|---|
| `Certificate.overall_status` | Whether the confirmed requirements pass, fail or lack evidence | `readiness_engine.decide` |
| `Certificate.quality_profile` | Six independent quality dimensions, coverage, gates, policy and adaptive status | `quality_aggregation.aggregate_quality` |
| `Certificate.experimental_score` | An older optional provisional arithmetic scalar over observed components | `experimental_scoring.calculate` / `aggregate` |

Passing a minimal requirement set can coexist with unknown quality dimensions. Unknown means that evidence or an explicit criterion is missing; it is not a zero measurement. None of these outputs is predicted model accuracy. Actual accuracy-related metrics come from held-out model evaluation.

<details>
<summary>Analogy: a research laboratory</summary>

The browser is reception. FastAPI is the coordinator. Ingestion registers and inspects the samples. The context interpreter writes a draft research brief that the user approves. Metrics are instruments; each measures one property. The decision engine compares instrument readings with the approved brief. Adaptive aggregation summarizes defined criteria while retaining gaps. The training worker runs the experiment. SQLite is the records archive, while large datasets and instruments' output files stay in managed cabinets on disk. Reporting prepares a readable account of the stored evidence.

This analogy does not imply a background scientist invents conclusions: context interpretation may use an external language model, but the measured report narrative and comparisons are generated from saved evidence by ordinary application code.

</details>

## How to read the diagrams

Rounded/action nodes describe verified functions or operations; boxes describe components or data; cylinders indicate persistent storage; diamonds are decisions. Solid arrows describe the labeled runtime control/data transfer. Dashed arrows in the dependency diagrams mean imports, not direct execution. Sequence diagrams show time from top to bottom. Class triangles mean inheritance, diamonds mean contained fields and labeled dependency arrows mean use/conversion. A subprocess boundary is always named. A polling request is a fresh HTTP read, not a server push.

The guide is a source snapshot. Line references refer to the inspected files and may move after edits. The generated inventory records source hashes. It does not inspect credentials, uploaded personal datasets or checkpoints. The documentation verification does not rerun expensive training or test live providers; earlier application test results are described separately from what was checked for this book.
'''

TEXT['ch02'] = r'''
## Why responsibilities are separated

The architecture separates evidence from policy and transport. Ingestion understands bytes and source formats. Metrics understand measurements. `Goal` and the requirement engine decide which checks are binding. Adaptive profiles define how supported measurements become criterion scores. API handlers accept validated requests and save work; they do not contain the PyTorch training loop. This lets the same assessment engine run from the web worker or standalone research tools.

The project has grown through several layers, so historical names and interfaces remain. `Dataprofiler` and `FeaturAnalyzer` are preserved public names with correctly spelled aliases. The original `ISOIECchecker` is a retained helper with partially implemented profile-only checks; no active web readiness path invokes it. The standalone `research` package is reachable from its CLI, not dead code just because it is outside the main API.

The inventory below describes repository-owned files. Virtual environments, node_modules, caches, uploads, runtime databases, pretrained checkpoints and generated frontend bundles are excluded from the architecture inventory. Test code is inventoried as evidence, not presented as application behavior. The documentation generation tools under `reading/tools` are separate from the analyzed application.

## Working directory map

The following tree is generated from actual source, configuration and test paths. It is intentionally a source tree, not a dump of downloaded dependencies or research artifacts.
'''

TEXT['ch03'] = r'''
## Backend command and import-time behavior

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Uvicorn imports `main.py` and locates `app`. The module optionally calls `load_dotenv()` unless `PYTHON_DOTENV_DISABLED=1`, declares the legacy in-memory dictionaries, creates the bounded legacy assessment executor, defines middleware/routes and registers the persistent routers. It mounts `frontend/dist` at `/` only if that directory exists at import time. Defining a route is not executing an assessment.

FastAPI then enters `lifespan`. `workbench.api.services()` is a cached singleton factory. `Store()` resolves `WORKBENCH_ROOT` (default `.runtime/workbench`), creates the directory and SQLite table, enables WAL/foreign-key pragmas and returns alongside `JobManager`. `recover()` audits records left running, verifies a retained worker PID before terminating that owned process, marks interrupted work and starts the queue. Shutdown closes the manager, waits for the legacy executor and closes transient datasets. The queue is local; there is no Redis, Celery, broker or distributed scheduler.

## Browser startup

`frontend/index.html` loads `src/main.tsx` during development; a built asset entry does the equivalent in production. The final `createRoot` branch mounts `Workbench` by default. `Workbench` creates one React Query client. `Shell` loads dataset selection and theme preferences, selects a destination from the URL hash, fetches datasets/jobs/compute and renders the guided component or an advanced tool. The guided component restores its selected project ID and requests the saved workflow from the backend.

`GET /api/health` is a light health check. `GET /api/v1/compute` calls the cached detector compute inspection and can take longer on first access because it imports the ML runtime. A physically present graphics device is not equivalent to `torch.cuda.is_available()`.

## Development frontend

```powershell
Set-Location C:\DataReadiness\frontend
npm.cmd run dev
```

Vite binds to loopback and proxies `/api` to `http://127.0.0.1:8000`. The frontend uses relative API URLs, so it works through the Vite proxy or the built FastAPI host. `npm.cmd run build` runs `tsc --noEmit` and Vite. The TypeScript aliases describe expected JSON shapes at development time; backend Pydantic models enforce their own HTTP boundary at runtime. No frontend development server is required to open the documentation HTML export.

<details>
<summary>Analogy: opening the laboratory for the day</summary>

Importing the module puts labels on doors and defines procedures. The lifespan hook unlocks the records room and checks whether yesterday's instrument run stopped unexpectedly. A visitor opening the browser asks reception for the current records. That does not itself start a new experiment.

</details>
'''

TEXT['ch04'] = r'''
## Component map

`Workbench -> QueryClientProvider -> Shell` is the normal root. `Shell` offers three primary destinations: Dataset & goal, Readiness report and Model Validation Lab. A collapsed Workspace tools section exposes Overview, Datasets, Dataset Evaluation, Advanced Lab, Model Training, Experiments, Model Comparison, Research Analytics, Reports and Settings. Hash changes update navigation; navigation scrolls the document to the top.

`GuidedWorkflow` owns the selected project ID, mutation pending/error state and React Query reads for project lists/current project. It renders `DatasetUpload`, `GoalChat`, `ReadinessStep` or `GuidedValidation`. The top workflow navigation disables later stages until the saved goal and completed report exist. A direct sidebar/hash jump still renders a prerequisite explanation rather than an unlocked action.

`DatasetUpload.upload` sends an `XMLHttpRequest` because upload byte progress is needed. ZIP/file import uses multipart `file` and `metadata`; folder upload uses multiple `files` plus JSON `paths`. A local directory is a JSON POST through `api`. On success, a workflow record is created for the returned dataset ID. Transfer at 100% means bytes arrived; the server can still be parsing. The UI labels this state instead of pretending the analysis is complete.

`GoalChat.send` posts a message and the per-request remote-provider flag. The returned workflow updates the query cache. The interpreted context becomes an editable draft; the user can correct task, classes/target and description, answer questions and inspect all structured fields. A separate confirmation request saves the accepted context. Typing in a chatbox does not change the pipeline until the interpretation/confirmation actions complete. Unsubmitted text lives only in React state.

`ReadinessStep.run` sends assessment scope and split options. Before completion, it shows actual job stage text and optional worker details. After completion, `ReadinessSummary` shows requirement status and six dimensions; `ReportActions` prioritizes unmet requirements and recorded evidence gaps. Tabs reveal `QualityAnalytics` or the technical measurements/audit trail. An unknown dimension remains visibly unscored even when basic requirement checks pass.

`GuidedValidation` derives its phase from saved preparation, readiness and experiment records. The phases are data-change selection, changed-data measurement, training configuration, frozen-protocol review, actual execution and results. `ValidationResults` joins the experiment's evaluation/model IDs, displays original-versus-controlled metrics and passes DTOs to `BenchmarkCharts`. The research tab requests `/research?experiment_id=...`; its CSV export targets the same experiment. Advanced research tools remain an explicit opt-in.

## Requests, cache and local state

`api<T>(path, options)` prefixes `/api/v1`, fetches JSON and throws the backend `detail` on non-OK responses. `json(body, method)` serializes a request body and supplies the content-type header. There is no generated client SDK, GraphQL layer or frontend secret store. `Shell.act` and guided `perform` wrap async actions in busy/error handling and invalidate React Query after successful mutations. The guided uploader creates a new project only after an import returned an ID.

Server records are authoritative for saved goals, job state and results. Component state holds draft inputs, active tabs, selected comparison results and transient upload progress. `localStorage` keeps selected workflow, selected dataset and theme. `sessionStorage` keeps advanced Lab section selection and the original UI's last assessment ID. Existing synchronous storage reads can throw if browser storage is prohibited; the legacy hook guards its storage/history operations, while the guided Shell path has fewer such guards. That is a documented limitation, not a fix made for this guide.

Jobs are polled every two seconds by Shell. Guided project state uses three-second polling; validation/experiment reads use roughly 2.5 seconds. Individual logs/curves poll while running. The original `useAssessment` hook uses a separate 600 ms status loop, a slower retry after read errors and a separate completed-certificate effect so a cancelled polling request cannot discard a report. There are no sockets or streamed token responses.

## Visualizations and errors

Recharts receives plain arrays from saved DTOs: dimension score/weight arrays, actual histogram bins, loss CSV rows, per-class counts, PR curve coordinates and research points. React does not calculate a new dataset score by drawing a bar. `QualityAnalytics` submits recalculation requests; it does not rerun ingestion. `BenchmarkCharts` separates AP curves from operating-threshold error counts. Forms disable invalid/busy actions, `ErrorMessage` or local alerts surface request failures and empty states tell the reader which evidence is unavailable.

<details>
<summary>Analogy: a receptionist's notepad and the official archive</summary>

React state is the notepad while someone is speaking. React Query is a convenient copy of recently retrieved records. SQLite is the official archive. Saving a goal creates an archived record; switching tabs merely changes which page of the notepad is visible. Polling means asking the coordinator whether the instrument has finished.

</details>
'''

TEXT['ch05'] = r'''
## Two HTTP families

`main.py` owns the original `/api` routes. Their datasets, interpretations and jobs are process-memory dictionaries, with owned staged files for large inputs. Restarting the backend loses those session IDs. Legacy jobs use `ThreadPoolExecutor(max_workers=2)` and cooperative cancellation through a threading Event.

The persistent workbench is split among four included routers plus the guided router: `workbench.api`, `quality_api`, `lab_api` and `workflow_api`, with `/api/v1` or `/api/v1/quality` prefixes. Their records live in Store. Submitting a persistent job creates a database record and starts an isolated process through JobManager. The static frontend mount comes after all these routers, so it does not intercept their API paths.

## HTTP boundary and error conversion

FastAPI parses multipart bodies or JSON and constructs the declared Pydantic request model. The shared `Model` forbids unknown fields and disallows nonfinite numbers in validated numeric fields. Generic `Any`/dictionary values need their additional domain checks and JSON serialization safeguards. Individual schemas add bounds, enum choices and model validators. This validates shape and some constraints; domain checks such as dataset hashes and train/test independence happen in service code.

`BodyLimitMiddleware` counts streamed request bytes. `ErrorRoute` wraps the persistent handler: missing record keys become 404, and value/filesystem/archive failures become 422. Schema failures also use FastAPI's 422 response. The original routes explicitly convert their own session/import errors and use 409 for unavailable/in-use resources, 429 for capacity and 413 for the middleware's body limit. A worker failure is usually a failed job record returned by a successful GET, not an HTTP 500 at the browser that originally queued it.

The request path is generally `component callback -> api/json or XHR -> FastAPI schema -> handler -> domain service -> Store/files -> JSON response -> React Query/state`. Export links use a direct browser GET and download the returned media type. No browser connects directly to SQLite or loads a checkpoint into the React process.

## Complete declared route map

The following inventory is generated from actual route decorators. Framework-provided `/docs`, `/redoc` and `/openapi.json` are not repository-declared handlers. Path aliases are listed separately. A `response_model` describes FastAPI serialization; handlers returning a Response may instead choose JSON, HTML, CSV, PDF or an image/checkpoint media type. Relevant consumer and service columns are a curated routing map, not an automatic claim that every syntactic callee executes on every branch.
'''

TEXT['ch06'] = r'''
## From bytes to a Dataset

Both APIs ultimately use existing ingestion adapters. The workbench calls `datasets.import_file` with a file handle. It delegates to `load_dataset_stream`, then persists a normalized Dataset and a managed source. Registering a local directory uses `DirectoryArchive`, a read-only archive-like adapter that rejects links/junctions and paths outside the resolved root. It initially copies no images. The parser still decodes/profiles the source; later training materializes verified bytes into managed cache files.

`Dataset` is an in-process dataclass with `name`, `sha256`, `modality`, `frame`, `profile`, metadata, canonical `objects`, issues, hashes, near-duplicate evidence, optional `disk_path`, ownership and source path. It is not itself a database row. For ordinary in-memory data, `frame` is the full pandas table. When `disk_path` is set, `frame` is only a bounded display preview, and full-data checks must query DuckDB.

`Dataprofiler.profile()` produces a `DataProfile`; converting it to a dict gives observed row/column counts, missingness, duplicate rows, finite numeric summaries, selected categorical counts and date diagnostics. It keeps missing values distinct from infinity and does not silently coerce invalid measurements into scientific truth. Image loaders extend the profile with format, class counts, declared classes, issues and source interpretation.

## Table and record dispatch

`load_dataset_stream` stages incoming bytes using bounded reads and configured upload limits. Smaller CSV/Parquet inputs can take the in-memory path. Large tables use batches and a private DuckDB table, with an owned spill directory, configurable memory/threads and internally generated SQL column identifiers. `full_profile` queries the entire table. A 100-row frame is for display, not a statistical sample used to replace whole-data calculations.

`record_formats.prepare_records` selects an adapter. CSV/TSV/tab/pipe-like data uses declared or detected delimiters and strict structure. Text can be one line or one document per record. JSON/JSONL rejects invalid constants, duplicate keys and ambiguous structures; nested values are represented consistently with parsing notes. XML uses defused parsing with entity/DTD restrictions and a chosen record path. XLSX reads cells without executing formulas, requires an unambiguous sheet and nonempty unique headers. Arrow/Feather use record batches. Unsupported or ambiguous sources report a selection requirement or ingestion error.

Encoding is evidence too. BOMs identify UTF encodings, otherwise strict UTF-8 is preferred; a recorded Windows-1252 fallback can handle legacy spreadsheet text. An explicit declared encoding is not silently overridden. Invalid bytes, binary controls and oversized logical records fail rather than being discarded. A successfully decoded string does not by itself prove that the language/encoding interpretation is semantically correct.

## Containers and images

`load_container` validates members before choosing a dataset. ZIP/TAR/compressed inputs are inspected without executing bundled code. Member count, expanded size, compression ratios, unsafe names and links are checked. Nested archive depth is bounded. Multiple plausible tables/image layouts require an explicit choice rather than automatically mixing train and test files. The original UI exposes `/datasets/{id}/select`; the workbench import surfaces an error and accepts parser metadata on a new import. This difference is important when troubleshooting a downloaded repository archive.

For labeled images, `image_ingestion.inspect_image` decodes one image, caches dimensions and proxy pixel measurements and hashes original bytes. `parse_yolo` checks class indices and normalized box coordinates. `parse_visdrone` preserves the distinction between ignored regions and target boxes. `universal_images` reads COCO, Pascal VOC, ImageFolder, YAML/names YOLO and unlabelled inputs, resolves image references and converts supported target geometry to normalized `(x_center, y_center, width, height)` boxes. Source-specific crowd/ignore semantics remain documented; canonical conversion does not make final evaluation an official COCO evaluator.

An empty annotation file is an explicit annotated negative. A missing annotation file is unknown label availability. Valid syntax does not establish that every visible object is labelled or the class is correct. ImageFolder names are inferred classification labels and require review. Single unlabelled images can be inspected without inventing labels. These data can support readiness analysis even when detector training is ineligible.

## Persistence and mutability

`persist` writes `frame.json` (pandas table orientation), `evidence.json` and, for disk tables, `table.duckdb`. The dataset record contains source paths/kind, profile, SHA256/version, eligibility/reasons and provenance. `load` reconstructs Dataset from these files. `image_bytes` accepts only a member in the saved manifest and verifies its hash. Training verifies bytes again. A registered external directory is mutable outside the app; a changed source must be imported as another snapshot.

`subset` filters an image Dataset to the exact selected training names, filters objects/hashes/issues, recomputes its profile and records training-only scope and identity. `combined_readiness` constructs a virtual union of both source training subsets, namespaces image paths and reassesses the whole combined population. It retains only compatible common metadata and does not treat one source's independent audit as an audit of the union.

<details>
<summary>Analogy: inventory cards and specimen boxes</summary>

The Dataset object is a working inventory card. A pandas frame is the table written on that card; a disk-backed Dataset instead points to a large ledger and carries a small preview. Store persists the card's serializable evidence and the ledger's location. It does not cram every specimen into a SQLite blob. Hashes are tamper-evidence labels: they reveal changed bytes, but do not authenticate who originally collected them.

</details>
'''

TEXT['ch07'] = r'''
## The contract is Goal inside DatasetContext

`Goal` describes the task, target, required/optional/excluded inputs, class requirements, environment, scenarios, units, ranges, missingness constraints, provenance needs, partition/group fields, UAV parameters, performance objectives and quality policy. `DatasetContext` wraps it with `original_text`, selected tags, field-source labels, questions, provider and the `confirmed` flag. Required/optional/excluded inputs cannot overlap, duplicate requirement IDs fail and numerical ranges/fractions are checked by validators.

In the guided workflow, `project_goal` appends the latest user message to the project's saved messages and joins the user texts, bounded to 12,000 characters. It calls `interpret(text, {}, allow_remote=...)`. The advanced and original forms can additionally supply tag categories. No dataset images, full tables or managed source files are passed as provider input. The saved context points back to the wording, not to an imagined perfect outcome.

## Deterministic interpretation and optional providers

`interpret` first rejects a limited set of instruction-override patterns; the code explicitly states that such a filter cannot prove the absence of prompt injection. It builds a conservative fallback from task/tag vocabulary, environment terms, explicit ranges and known targets. Conflicting tasks or tags become questions. Missing target/class information becomes questions. It does not invent a minimum training sample count, altitude range or desired recall value.

Provider selection reads configuration presence from the backend environment. A configured Gemini key/provider selects `GeminiProvider`; explicit or compatible historical model settings can select `OpenAICompatibleProvider`. `allow_remote=False` avoids provider use. `GeminiProvider.extract` sends one REST request with the key in a header, task text/tags serialized as data, zero temperature and JSON output. It validates finish state and parses the response as DatasetContext. The compatible provider sends a schema-oriented chat request. These adapters have no dataset tools or executable capabilities. Their type contract alone does not make every inferred claim correct.

The provider output is reset to the original user text/tags and unconfirmed state. Numerical values in selected requirement-related fields must occur in the user's wording, including percentage equivalents; deterministic conflict questions are retained. Any provider exception or invalid output yields the identified local fallback. The numerical grounding pass covers the listed fields in `interpret`; it is not a universal proof against all possible semantic mistakes in free-form dictionaries. Human review still matters.

## Confirmation and reuse

`confirm_project` requires the latest original text, a concrete task, no unresolved questions and `confirmed=True`. It creates another context record and updates `confirmed_context_id`. The exact confirmed context is later reconstructed with `DatasetContext.model_validate`, placed in the readiness job configuration and embedded in the resulting Certificate. A user revision creates a new context and clears the current assessment/validation pointers. Historical records are preserved. The guided action lock and `require_idle` prevent revising the current study while its analysis or validation compute is active.

The goal is persisted in the existing SQLite JSON records, not in a separate magically discovered goal file. The original transient API instead keeps an interpretation in its session dictionary. Model-performance objectives remain recorded context unless independent training/test evidence establishes them; saying “high accuracy” in chat does not calculate accuracy.

<details>
<summary>Analogy: approving a research brief</summary>

The interpreter is a draft writer. It proposes the question and required conditions. The user is the person who signs the brief. The lab instruments measure against that signed version, even if the user later writes a different draft. Saving the signed brief with the results makes it possible to ask which finish line the experiment actually used.

</details>
'''

TEXT['ch08'] = r'''
## What a metric returns

A `MetricResult` carries a metric ID, dimension, state, value, reason and `AssessmentEvidence`, with optional measurement status, unit, scope, provenance, estimation method and limitations. Evidence includes the calculation method, sample size, bounded references and details. A plain diagnostic `state='pass'` often means the calculation produced observed evidence without an acceptance threshold. It is not a claim that the dataset meets an unstated requirement. `evaluate` is the separate operation that compares a measured value with an explicit Requirement.

`@metric(...)` stores a `MetricDefinition` in `REGISTRY` and its function in `CALCULATIONS`. UAV property registration creates a closure for each stable ID. `run_metric` is the public dispatcher: it handles unknown IDs, modality applicability, UAV structured output-path extraction, full-data disk dispatch and ordinary calculations. MissingEvidence becomes insufficient evidence. Calculation exceptions become an error result rather than a fabricated number. UAV estimated output paths retain their estimated status.

## The exact active order

`assess` enters a visual-session context and calls `_assess`. The engine verifies confirmation/questions, compiles Goal requirements and adds relevant image/manifest checks. It evaluates requirements in list order. Then it executes the `DIAGNOSTICS` list below in its declared order. For image datasets, object-detection/tracking/telemetry tasks or explicit UAV parameters, it appends `PROPERTY_IDS`, namely UAV-01 through UAV-20 in registry order. This is the actual check order, not a list of all registered functions sorted by filename.

After measurement, `learning_views` assembles six interpretations supported by property IDs. Modality conflicts add explicit findings. `decide` makes the requirement decision. The engine calculates the provisional scalar, builds the certificate/reproducibility record, collects cached pixel distributions and calculates the adaptive profile. Requirement checks and diagnostics can invoke the same metric more than once; they are not a single globally memoized metric pass. Optional image model work has its own bounded session behavior.

## Supported evidence by modality

Tables support missingness, finite-value/type/range checks, class/label evidence when a target is declared, duplication, selected leakage/provenance checks and corresponding-reference comparisons. Text documents enter as tabular records such as a `text` column; there is no general-purpose language-model semantic text-quality judge. Image inputs add decode evidence, labels/boxes, image hashes, geometry and cached pixel proxies. Telemetry is represented by declared tabular columns and metadata, not by a proprietary flight-log driver.

UAV properties include recorded scene/weather/time/season/viewpoint/altitude, annotated object geometry and density, clarity proxies, audit evidence, deployment-reference differences, collection independence, sensor mappings and temporal ordering. Optional local ONNX models can provide estimated scene/appearance/embedding evidence only when explicitly configured. No default external model download is used to fabricate missing metadata. Metadata presence is not authentication, sensor calibration presence is not validity and static-image appearance is not recorded capture time.

Disk-backed tables take the SQL path for implemented metrics. `uav_metrics.assessment.calculate` explicitly reports unsupported for its disk-backed metadata/temporal summaries rather than evaluating the small frame preview. Some whole-data reference-distance checks are likewise unavailable in `disk_metrics`. The exact registry and status descriptions below expose those distinctions.

## The older helper analyzers

`Dataprofiler` supplies general observed profile statistics. `FeatureAnalyzer.analyze` detects constant/dominant features, numeric correlations and a feature identical to the target; it records that mutual information requires a declared estimator configuration and does not compute it. `LabelAnalyzer.analyze` computes label proportions, missingness, largest/smallest observed-class ratio and optional raw agreement/Cohen's kappa on paired nonmissing labels. `ISOIECchecker` can calculate profile-only completeness/uniqueness/conformity, but its accuracy, rich consistency and provenance checks deliberately lack enough evidence and return None. It is not the active engine just because its filename sounds authoritative.

## Measurement mathematics in concrete terms

The denominator is part of a result. A column's missing fraction is missing cells divided by rows; a nonfinite rate inspects the selected numeric population; unreadable-image rate divides failed decodes by image count. Duplicate rate counts repeated occurrences after the first, divided by records/images. For 10 observations with two redundant repeats, duplicate rate is 0.2 and criterion uniqueness is 80/100. `bbox_invalid_rate` divides annotated images containing malformed target annotations by images that have annotation files; it is not malformed boxes divided by all boxes. For eight annotated images with two malformed files, that rate is 0.25 even if the archive contains other unlabelled images.

`scenario_count` intersects all declared conditions and optional class presence. Separate daytime and urban counts do not establish that any urban-daytime example exists. If the matching count is zero and relevant metadata is unknown, the check cannot conclude true absence. The current sample-sufficiency metric deliberately asks for a confirmed minimum instead of deriving one universal sample count from a formula.

For a normalized box `(x,y,w,h)` in an image of width W and height H, the geometry module derives width `w*W`, height `h*H`, area `w*W*h*H`, aspect `(w*W)/(h*H)` and relative occupancy `w*h`. A width of 0.1 in a 1,000-pixel image is 100 pixels; the same normalized width in a 100-pixel image is 10 pixels. Density divides object count by image megapixels. Pairwise crowding uses IoU and center distance divided by the image diagonal, within a declared bounded pair population.

The image inspector uses grayscale brightness mean and standard deviation for contrast. For a bounded thumbnail, its four-neighbor Laplacian is `L = up + down + left + right - 4*center`; the clarity proxy is the variance of L over valid interior pixels. Luminance entropy is `-sum(p*log2(p))` over nonzero histogram bins. A high entropy can come from texture/noise, and a high Laplacian variance can come from sharp noise, so these proxies are not a semantic object-visibility model. Encoded bytes per pixel does not prove compression quality.

For numeric reference columns with matching declared units, `reference_comparison` delegates to SciPy's one-dimensional Wasserstein distance. Conceptually it is the area between empirical cumulative distributions, in the column's original unit. Moving every value from 10 meters to 15 meters gives distance 5 meters; distances across unrelated units are not averaged into one readiness score. For categorical distributions P and Q, the code squares SciPy's base-2 Jensen-Shannon distance to obtain divergence in bits:

```text
M = (P+Q)/2
JSD(P,Q) = 0.5*KL_base2(P||M) + 0.5*KL_base2(Q||M)
KL_base2(P||M) = sum of P_k*log2(P_k/M_k) over P_k > 0
```

Identical distributions have divergence 0; two disjoint deterministic categories have divergence 1 bit. Reference relevance remains user-declared: a low distance to an irrelevant collection does not prove deployment representativeness.

Paired-label raw agreement is the matching fraction after missing pairs are excluded. Cohen's kappa is delegated to scikit-learn and conceptually adjusts observed agreement p_o for marginal-chance agreement p_e: `(p_o-p_e)/(1-p_e)`. Constant/undefined cases remain unavailable. Separately, `uav_metrics.learning.associations` computes image-level binary class-presence versus nuisance-category mutual information from a contingency table, normalized as `MI/sqrt(H(nuisance)*H(class_presence))`. Constant variables return null; missing/invalid annotations are excluded instead of being treated as absent classes. This descriptive association identifies review candidates, not proof that a trained model learned a shortcut.

The optional visual adapter normalizes nonzero embedding vectors before cosine dot products, reports bounded nearest-neighbor/dispersion summaries and labels model-derived categories as estimates. Its softmax probabilities and predictive entropy are uncalibrated; they do not establish classifier correctness. Temporal checks use differences in recorded row order within sequence, without bridging missing timestamps. They count backwards intervals, repeated times and frame gaps; they do not infer physical motion or identity correctness from static images.

<details>
<summary>Analogy: an instrument reading versus a pass/fail rule</summary>

A thermometer returns 12 degrees and states how it measured that value. Whether 12 is acceptable depends on the signed experiment brief. Likewise, `row_count=100` is an observation; “at least 50 rows” is a criterion. A missing thermometer reading is not zero degrees. A failed instrument and an out-of-range reading also require different follow-up.

</details>
'''

TEXT['ch09'] = r'''
## The six dimensions

The configured dimensions are Annotation quality; Diversity and required coverage; Signal and image quality; Distribution and sample adequacy; Task and operational suitability; and Data integrity and reliability. Their exact component assignments are generated from `metrics/config/adaptive_quality.json` below. Task policies, scenario multipliers and manually confirmed overrides define priorities. These are engineering hypotheses; the code does not fit them to the user's model test results or claim they are empirically calibrated.

`QualityProfile` validates configuration versions, finite positive base ratios, component IDs, task coverage and scenario compatibility. `_weights` separately validates runtime overrides: known IDs only, finite nonboolean numbers in [0,1000]. Changing a weight requires explicit override confirmation. `load_profile` supplies the default file; a saved assessment carries its full frozen profile. Reaggregation normally reuses that profile. `use_current_profile=True` explicitly requests the current policy.

## From raw values to comparable criteria

Let x be an observed fraction in [0,1]. For a good fraction, `s = 100*x`; for a defect fraction, `s = 100*(1-x)`. A zero-count criterion returns 100 for an observed count of zero and 0 for an observed positive integer count. That all-or-nothing transformation is criterion compliance, not a continuous estimate of defect severity. Nonfinite values, booleans presented as numbers and invalid fractions do not become scores.

Other component measurements are structured. Required-category coverage is `100 * observed_requested_categories / requested_categories`, only after the categories are explicitly declared. Requirement coverage is the fraction of conclusive passing findings; any unknown finding makes that component unknown. Annotation-audit agreement uses the independently reviewed error fraction. Clarity is the fraction of readable images meeting the stored criterion, using cached thumbnail Laplacian variance. Object visibility uses an explicitly confirmed minimum annotated object side in pixels. These are data/criterion measurements, not semantic detectability or predicted accuracy.

Uniform-class balance is calculated only when `goal.uav_parameters.class_balance_target` explicitly equals `uniform` and at least two classes are declared. For class proportions p_k and K declared/required classes, the score is:

```text
H(p) = -sum_k p_k * ln(p_k), taking 0*ln(0) as 0
balance = 100 * H(p) / ln(K)
```

Counts of an unobserved required class still affect K. An 80/20 two-class split has H approximately 0.5004 and normalized balance approximately 72.2/100; a 50/50 split has 100. Uniformity is only a requested reference distribution; actual deployment prevalence need not be uniform.

## Effective weights and coverage

Within a dimension, applicable positive-weight components have raw importance:

```text
a_i = task_component_weight_i * scenario_multiplier_i * confirmed_override_i
w_i = a_i / sum_j(a_j)
coverage = sum of a_i for measured components / sum of a_i for eligible components
```

The sums retain eligible components whose evidence is unknown. Not-applicable components and zero-importance components do not influence the score denominator. If a dimension has two equal-priority eligible components and only one is measured, coverage is 0.5. The measured component can be 100, but the primary dimension score is null. It is not silently renormalized into a perfect dimension score.

Across dimensions, a separate weight is `task_dimension_ratio * scenario_multiplier * confirmed_override`, normalized over applicable dimensions. These dimension weights explain importance and order next actions. They do not produce a primary all-dimensions percentage. Changing dimension importance therefore does not change a dimension's internal score. Component priority overrides can change its internal score if the evidence is complete.

## Weighted geometric mean and masking

For fully covered eligible component scores s_i in [0,100], `geometric` computes:

```text
D = 100 * exp(sum_i w_i * ln(s_i/100))
```

If any positive-weight component has a genuine score of zero, D is exactly zero. The implementation checks this before taking logarithms; it does not substitute a small epsilon. With scores 90 and 40 and weights 0.75 and 0.25, D is approximately 73.5. An arithmetic mean would be 77.5. Geometry penalizes imbalance, but it cannot by itself prevent every weak component from being obscured. Therefore the code also keeps the dimension in warning if any contributing criterion is below `warning_below`, even when D is above it. The default 70 threshold is an uncalibrated review policy.

## Independent gates and two readiness decisions

Critical component gates are evaluated separately from weighting. An observed defect on a block gate can block the adaptive status even if the corresponding score importance is zero. Confirmed hard requirements become block gates; preferred requirements become warning gates. A task/modality incompatibility adds a block failure. A task preview clears context-dependent measurements/findings and adds an evidence gate requiring a fresh task-specific assessment.

Adaptive status is `blocked` for a failed block gate, otherwise `insufficient_evidence` for missing positively weighted dimensions/block evidence, otherwise `conditional` for remaining gate or dimension warnings, otherwise `ready`. This is separate from `Certificate.overall_status`.

The requirement decision follows this actual precedence: a hard failure gives NOT_READY; otherwise a finding/metric error gives ASSESSMENT_ERROR; otherwise an unspecified task or unpassed hard evidence gives INSUFFICIENT_EVIDENCE; object detection without required classes is insufficient; unmet preferred criteria give CONDITIONALLY_READY; otherwise READY. A hard failure remains established even when another check is unknown.

## The retained arithmetic scalar and sensitivity

`experimental_scoring.aggregate` is different. It computes `sum(a_i*s_i) / sum(observed a_i)`, reports observed/applicable coverage and a possible score range that assigns missing components zero or 100 at its ends. It can therefore display a partial observed scalar. It never overrides requirement readiness or the strict-coverage primary dimensions. Its configuration and profile hash are independent of the adaptive quality profile.

`quality_api.sensitivity` perturbs dimension priorities by 0.5 and 2 and reports resulting importance rankings. Its metric-deterioration scenarios multiply measured scores by 0.5 or 0 and recompute a hypothetical geometric score. Those simulations are explicitly not dataset measurements, training outcomes or persisted experimental evidence. Changing a selected task likewise makes a preview, not a new collection of labels.

<details>
<summary>Analogy: six laboratory reports and mandatory checks</summary>

Each quality dimension is a report folder with its own tests. The task determines how much attention each test receives. Missing pages remain missing, so the folder does not get a complete score. A mandatory failed check remains flagged even if the committee assigns its numerical score no importance. The folders stay separate; a pristine file-integrity folder cannot repair wrong labels in the annotation folder.

</details>
'''

TEXT['ch10'] = r'''
## Guided preparation versus the advanced builder

The guided lab starts from a completed training-partition readiness assessment. `prepare_validation` creates a recorded training-only variant with blur, resolution reduction, label omission or box perturbation; creates its split with the existing grouped splitter; and queues readiness under the same confirmed context. Validation/test image and annotation bytes are retained. A `workflow_validation` record links the variant, split, readiness job, intervention and later experiment.

After the changed-data assessment finishes, `validation_protocol` constructs an `ExperimentRequest` with original data as candidate and controlled data as reference. Both configurations share original validation data and the same original held-out test partition. The route calls the existing `create_experiment` domain validation and stores its experiment ID. The browser reviews this immutable protocol before `start_experiment` queues actual work. The name reference here means comparison arm; an artificial comparison dataset is still labelled as related artificial provenance in analytics.

The advanced `ExperimentBuilder` exposes candidate/reference/combined/controlled modes, multiple seeds, shared or per-source validation, class mappings, frozen aggregation versions, matched or natural sizes and an independently selected test. Reference/combined modes require reference configuration. A controlled mode requires artificial provenance. Matched size requires a reference dataset, retains whole recording groups and reassesses the selected training subsets. Combined mode intentionally increases sample count and cannot share that matched-size protocol option.

## Partitions and preflight validation

`create_split` loads hashes and cached/new perceptual signatures, unions exact duplicates, near images and known recording groups, then preserves complete declared partitions or uses a seeded allocation of whole clusters. Partial declared assignments, overlap across declared partitions, fewer than three independent units or an empty partition fail. Without group metadata the user must acknowledge unverified recording independence.

`validate_training` checks training eligibility, exact dataset/split/readiness association, a completed detection-task readiness report, any selected aggregation, nonempty annotated train/validation populations, class schemas and trusted local checkpoint paths. It does not require a READY verdict; a diagnostic research experiment may intentionally train on data with known issues. The relevant invariants are evidence identity and training feasibility, not a hidden readiness pass gate.

`validate_experiment` matches model configurations, validates class mappings and asserts that final-test data are independent of every development manifest. `assert_independent` checks exact image hashes, likely near duplicates and comparable recording groups; missing comparable groups require acknowledgement. These checks cannot prove collection independence if metadata is wrong or near-duplicate heuristics miss a relation. Preflight runs again when an immutable experiment starts.

## Persistent queue and worker execution

`JobManager.enqueue` writes a queued job and starts a coordinator thread. Its loop selects the oldest queued work, creates `jobs/{id}`, launches `python -m backend.workbench.worker {id}` with the same Python executable and local storage root, and redirects stdout/stderr to `worker.log`. One process runs at a time for that manager. This is process isolation for compute and predictable local device scheduling, not a distributed queue or a guarantee across multiple independently started backend instances.

`worker.run` dispatches by `task`: readiness reconstructs Dataset/Context and optional train subset; training calls `detector.train`; evaluation calls `evaluate_models`; experiment calls `execute_experiment`. A parent experiment creates child job records and invokes `run` serially within the same process. It checks each child's terminal state and stops if a child failed. All requested training completes before the final evaluation phase. The browser polls persisted records and epoch callbacks instead of holding one training HTTP request open.

## Real model preparation and training

`device` examines actual PyTorch CUDA availability; `auto` chooses available CUDA or CPU. The selected device is frozen. `gpu_usage` reports actual device memory allocation/reservation, not utilization percentage. `setup_detector` configures local Ultralytics storage and disables integrations/synchronization, with no automatic pretrained download. The supported architectures are YOLO11n and YOLO11s. A trusted `.pt` import is a local user-controlled operation; PyTorch checkpoint files require trust.

`materialize` verifies source bytes, maps classes one-to-one, creates cached image/label inputs and uses hard links where safe for managed bytes. External mutable directory files are copied into verified cache rather than hard-linked. `write_yaml` creates dataset YAML and train/validation lists. Final test images are not supplied to training/checkpoint selection.

The code fixes the initial target-model seed to 0, constructs the requested class head, saves a local initialization checkpoint, hashes the tensor state and verifies that the trainer begins with that exact state. The requested run seed then governs training stochasticity. This separates matched initialization from repeated augmentation/data-order seeds. The `pretrained=True` argument in the library call refers to the explicitly constructed/reloaded model state; it does not by itself imply downloaded COCO weights.

The actual train call uses the requested epochs/batch/image size/learning rate/device, SGD, deterministic mode, no mixed precision, zero loader workers, no cache, no early-stop patience and checkpoint saving. The `on_fit_epoch_end` callback records completed epochs, train losses, validation values and GPU memory, with a guard for a repeated library callback after final validation. The external Ultralytics/PyTorch loop performs gradients and optimization; those internals are not repository-owned Python functions.

## Checkpoints, resume, combined scope and imports

`finish_training` requires `run/weights/best.pt`, validates that checkpoint on validation data and creates a model record with hashes, effective configuration, manifests, code hash, environment, training counts and the exact quality profile. Combined training calls `combined_readiness` for the entire training union and links the model to the resulting virtual combined dataset/report. It does not attach only the candidate's readiness profile to a larger combined population.

`lab_api.resume` accepts only stopped training jobs with usable retained checkpoint/configuration. It queues a new training job with a resume pointer. `resume_training` loads the actual epoch checkpoint and calls library resume so optimizer/epoch state is retained. There is no generic resume-from-midstep implementation for a partially completed named experiment; partial child artifacts remain, and a new protocol is the supported retry path.

Imported checkpoints retain user-declared provenance and are excluded from controlled predictive correlations. External published benchmarks are stored in their own record kind with a comparability warning. They are not inserted as measured local evaluation results and do not establish a fair model ranking.

## Evaluation: operating errors and average precision

`test_manifest` chooses only the selected test partition, or all images in a supplied external test dataset. Its current identity includes image bytes, canonical annotations/classes, class mapping and ground-truth protocol, plus a separate image-only checksum. This prevents two differently labelled test sets from being called identical because their images match. The test is asserted independent from each model's retained development manifests.

`evaluate_models` uses Ultralytics validation at the recorded low AP confidence floor for the PR/AP sweep. It also runs per-image predictions at the declared operating confidence. Repository `detection_errors` greedily matches confidence-ordered predictions to unused ground truth of the same class at the chosen IoU; unmatched predictions are FP, unused truth is FN. With TP, FP and FN:

```text
precision = TP/(TP+FP)      recall = TP/(TP+FN)
F1 = 2*TP/(2*TP+FP+FN)
IoU = intersection_area / union_area
```

A zero denominator produces None, not an invented perfect or zero score. For TP=8, FP=2, FN=4, precision=0.8, recall approximately 0.667 and F1 approximately 0.727. FP per image uses images as denominator; it is not a true-negative-based false-positive rate. Latency is mean model inference time and excludes loading/preprocessing/NMS.

Average precision is delegated to the installed Ultralytics metric implementation. Inspection of that installed dependency shows a precision envelope and 101-point interpolated trapezoidal integration over recall. mAP50 averages class AP at IoU 0.50; mAP50_95 averages across the library's 0.50 through 0.95 thresholds. The repository adapter retrieves these values through `research.experiments._metrics`; it does not implement an alternative AP formula. Library versions are therefore part of reproducibility.

`confusion_counts` uses confidence-ordered spatial matching ignoring class to reveal class confusion, with rows predicted and columns true plus background. It is deliberately distinct from the class-aware TP/FP/FN matching. Canonical boxes treat retained parsed objects as ordinary boxes; the adapter explicitly does not reproduce official COCO crowd-ignore semantics, so published COCO values are not automatically comparable.
'''

TEXT['ch11'] = r'''
## How data quality joins model outcomes

`research` walks newest evaluation records, loads the corresponding model, training dataset and readiness record, and optionally filters selected model IDs, dataset, scenario, test or configuration. It keeps the latest matching observation per model/protocol. `numeric_metrics` extracts finite nonboolean numeric leaves from measured diagnostics and available dimension scores. It also exposes clearly named research baselines: equal/adaptive arithmetic/geometric dimension combinations only where every required dimension is complete, and training image count. These predictors do not become the application readiness verdict.

The configuration signature includes architecture, matched initialization hash, budget, device, classes, actual test identity/settings, full context, experimental-score profile/observed components and adaptive policy/measurement support. This is deliberately strict: unlike task descriptions or different evidence coverage can prevent pooling. A saved experiment's guided research request passes `experiment_id`; the API restricts rows to that experiment's model IDs. The advanced workspace's default can inspect all stored runs.

## Independent unit and repeated seeds

Natural, non-combined, provenance-verified source rows are grouped by source image identity and comparable configuration. Repeated seed outcomes for that source become one dataset-level observation. If readiness differs across repetitions on the same source/configuration, that source is excluded instead of pretending its x-value is fixed. Sources sharing exact image bytes or declared comparable recording groups are excluded from independent-source correlations. Artificial variants, combined datasets and imported histories remain descriptive or paired evidence, not additional independent datasets.

For n independent training seeds on one source, mean performance is the arithmetic mean; sample standard deviation uses n-1. The reported 95% interval is a t interval:

```text
mean = sum(y_i)/n
sd = sqrt(sum((y_i-mean)^2)/(n-1))
interval = mean +/- t(0.975, n-1) * sd/sqrt(n), when n > 1
```

For seed outcomes 0.4, 0.5 and 0.6, mean=0.5, sd=0.1 and the small-n t interval is approximately [0.252,0.748]. It describes seed variation on that fixed data/protocol, not uncertainty across future deployment domains or independent test sets. With one seed, the interval is unavailable.

## Correlation and predictive checks

Pearson measures linear association; Spearman applies rank association and therefore asks whether values tend to increase together without requiring a linear scale. `research` requires at least three independent comparable sources and variation in both x and y before reporting either coefficient. Pearson's interval is requested for at least four sources. Spearman's interval uses 1,000 independent-source bootstrap draws with seed 42 when n is at least six; constant resamples are excluded, and at least 900 usable draws are required. These are exploratory checks without multiplicity correction.

```text
Pearson r = sum((x_i-x_mean)*(y_i-y_mean)) /
            sqrt(sum((x_i-x_mean)^2) * sum((y_i-y_mean)^2))
Spearman rho = Pearson correlation of the ranked observations
```

For x=(1,2,3) and y=(2,4,6), both coefficients are 1, but three illustrative points do not establish general predictive validity. A constant x or y leaves the coefficient undefined. Source independence, fixed policy and matched protocol are prerequisites, not cosmetic chart filters.

`predictive_comparison` performs leave-one-independent-dataset-out univariate ordinary least squares only with at least six matched sources and sufficient predictor variation. Each fold centers/scales x using its training sources, fits intercept/slope on those sources, predicts the held-out source and compares error with predicting the training-fold mean. The code reports MAE and RMSE. It never fits scoring weights using the held-out performance and does not train a multivariate quality predictor under another name.

```text
MAE = mean(abs(predicted_i - observed_i))
RMSE = sqrt(mean((predicted_i - observed_i)^2))
```

Paired intervention analysis matches a variant to its parent by seed, test identity/settings and training conditions. It reports size matching, performance deltas and compatible quality deltas. If quality policies differ, the affected dimension/baseline deltas are withheld. `ranking_stability` compares the same multiple models over multiple tests under matched evaluation settings, retaining ties so arbitrary model ID order cannot manufacture a reversal.

`comparison` audits protocol differences, imported provenance, legacy annotation-unaware evaluations and training sizes. A side-by-side table can be informative while still being a descriptive, confounded comparison. Empty research plots can therefore be correct: the application refuses to manufacture observations when there are too few independent matched sources.

<details>
<summary>Analogy: repeating an experiment is not recruiting another population</summary>

Training five times on one collection measures how the instrument varies on that collection. It does not turn one collection into five independent towns, flights or cameras. To claim a score predicts performance across datasets, the laboratory needs genuinely different comparable collections. A deliberately blurred copy is valuable for a paired intervention, but it is still related to the original collection.

</details>
'''

TEXT['ch12'] = r'''
## The actual database model

There is one declared SQLAlchemy application table, `research_records`, represented by `Record`. Its columns are `id` (primary key), `kind` (indexed), `created_at` and `document` (JSON text). There are no declared foreign-key columns between datasets, jobs and models. IDs inside JSON express logical relationships enforced by application lookups/validation. Enabling SQLite foreign keys does not turn those JSON fields into enforced relational constraints.

`Store.create` serializes with `allow_nan=False`; `get` reconstructs a response wrapper `{id, kind, created_at, payload}`; `list` orders newest first; `update` serializes read-modify-write with an RLock and SQLite `BEGIN IMMEDIATE`; `directory` restricts managed kind/id components to alphanumeric values. Connections use WAL and a timeout. This supports local concurrent HTTP/worker access, but the queue is designed for one local backend instance rather than multiple independent schedulers.

Record kinds used by the workbench include dataset, split, context, workflow, workflow_validation, job, model, evaluation, comparison, aggregation, experiment, external_benchmark and analysis. They share one table. A workflow points to context/readiness/validation IDs; a model points to training job, dataset and readiness IDs; evaluation points to model/test data; a named experiment lists child/model/evaluation IDs.

## Artifact cabinets

The root defaults to `.runtime/workbench` and can be configured. `datasets/{id}` contains managed source files, `frame.json`, `evidence.json` and optional `table.duckdb`. Registered directory sources retain an external path. `jobs/{id}` contains the worker log, frozen configuration, YAML/list inputs, initialization and training run artifacts. The run includes `results.csv`, `weights/best.pt` and retained epoch/last checkpoints as supplied by the training library. Test evaluation stores per-model prediction JSON and references its path from the evaluation record. `cache`, `imagecache` and `variations` hold materialized inputs, shared image bytes and controlled-data manifests. Source code writes each artifact path; the inventory is not an assertion that all possible artifacts exist before a run.

The in-process Dataset can be discarded and reconstructed from evidence. A disk-backed dataset points to DuckDB for full-data calculations. Transient staging owners are cleaned on normal completion/error, and persistent imports copy required files before closing them. Worker processes naturally release their loaded objects when they exit. JSON records preserve paths and evidence, not live Python references or GPU tensors.

## Cleanup and retention differ by interface

The original API's lazy `cleanup` uses a one-hour TTL and protects active datasets/jobs. Explicit deletion or shutdown closes owned Dataset staging, including DuckDB/temp files. Abrupt termination can leave temporary artifacts; cleanup must follow ownership and an inactive process check.

The workbench DELETE dataset endpoint archives it after checking active job use. It intentionally retains artifacts and research provenance rather than deleting the files used by older reports/models. There is no automatic age-based pruning of all durable research artifacts. Renaming changes the visible dataset record name; source hash/version and earlier job evidence remain their own saved data. External registered files are not made immutable by a name or checksum; materialization detects changes to verified image bytes.

Dataset/assessment/configuration hashes describe the byte evidence and selected processing context, not authenticated collection identity. They help replay a particular computation. GPU kernels, library versions, checkpoint files, source changes and declared metadata also affect reproducibility. The code hashes source in `finish_training` even when Git has no commit, and retains both actual source hashing and an optional revision string.

<details>
<summary>Analogy: a catalog entry is not the object on the shelf</summary>

The SQLite row is a catalog card with IDs, evidence and shelf addresses. The image archive, DuckDB ledger, model checkpoint and prediction file are separate objects on disk. Archiving a dataset hides its catalog entry from normal selection while preserving the shelf objects that existing research references. A local directory registration points to a shelf somebody else can still change.

</details>
'''

TEXT['ch13'] = r'''
## Classify the failure before retrying

An invalid HTTP request, unknown evidence, failed compute and cancelled work are different states. Pydantic/request validation catches malformed schemas. Ingestion rejects invalid formats/unsafe structures and can request a layout choice. `MissingEvidence` means the calculation lacks its required observation. A metric exception is an error result. A worker exception marks the persistent job failed and retains a traceback in `worker.log`. The guided frontend exposes the stage/error and lets supported operations be retried without inventing results.

If a remote context response is unavailable, incomplete, invalid JSON, schema-invalid or numerically ungrounded, `interpret` uses the identified deterministic fallback. This is not a model answering successfully; review the provider label and questions. The API key is read by the backend provider and is not returned to the browser. File access permissions for a local environment file are outside this documentation's scope; environment storage is not inherently an access barrier to a process that can read it.

For metric uncertainty, the engine records why evidence is missing, withholds applicable primary dimensions and can return insufficient evidence for a binding requirement. Changing a weight cannot supply observations. A missing checkpoint is a training/import/resume failure. An incompatible label map or test overlap fails domain preflight. A model with no positive predictions can legitimately have undefined precision and zero recall; this is different from an HTTP read error.

## Persistent cancellation and restart

Persistent `cancel` marks queued/running work cancelled and terminates the owning subprocess. Cancelling an inline child resolves to the parent experiment; the parent and still-active child records are marked cancelled so no orphan appears to keep running. Previously completed artifacts stay available. There is no promise that a force-terminated library operation produces a valid final checkpoint.

At startup, `recover` marks running jobs interrupted and updates running experiment status, checking retained PID identity before terminating an owned leftover worker. It can continue the existing queue. Training-only checkpoint resume creates a new job and reuses actual optimizer/epoch state if prerequisites are met. A stopped experiment has partial evidence, not a universal restart-at-any-step mechanism.

The legacy API uses cooperative cancellation: the Event is checked when the assessment progress callback runs. It does not preempt every inner operation immediately. The original UI hook treats a missing session assessment ID as unavailable after backend restart/expiry. Saved persistent workbench IDs have a different lifetime.

## GPU/CPU and reporting failures

`device('0')` rejects unavailable CUDA; `auto` chooses based on the installed PyTorch runtime. A CPU-only build cannot use a physically present GPU. Worker errors containing “out of memory” get a smaller batch/image suggestion; the code does not automatically lower training settings and pretend the protocol was unchanged. A new explicit configuration is required.

Export handlers require compatible record types/states and reuse saved measurements. JSON is the full evidence format for application reports; their readable PDF can explicitly abbreviate large structures. In this visual textbook task, the documentation PDF uses a separate complete-content exporter, so application-report abbreviation is not used to shorten the book. A documentation rendering issue is unrelated to whether a model trained successfully.
'''

TEXT['ch14'] = r'''
## Example 1: inspect a drone dataset against a declared use

This is an illustrative walkthrough using the real schemas and code paths, not a claim that a real user's dataset has these values. Imagine a small YOLO archive with classes.txt, image files, label files and metadata.csv. It contains 10 images with declared train/val/test assignments of 6/2/2, a `drone` class and recorded flight IDs. The software fixture generator can create such a synthetic source; synthetic imagery checks the software path and does not validate real UAV performance.

1. Open the normal frontend. `main.tsx` mounts Workbench, Shell queries the workbench and GuidedWorkflow renders the upload surface.
2. `DatasetUpload.upload` posts multipart bytes to `/api/v1/datasets/import`. `api.upload -> datasets.import_file -> load_dataset_stream` routes to the image loader. Inspection fills a Dataset frame, canonical objects, hashes and profile.
3. `persist` writes parsed evidence and Store creates a dataset record. The browser receives RecordResponse. GuidedWorkflow creates a workflow with that dataset ID.
4. Enter “Detect drones in images.” `GoalChat.send -> project_goal -> interpret` returns an unconfirmed DatasetContext. The configured provider can propose a draft; local fallback recognizes this simple task. Review the actual `drone` class name and required conditions instead of assuming aliases match.
5. Confirm the goal. `confirm_project` saves a new context record. The Finish line is the goal in that exact record, not a numerical desired accuracy silently inferred from the wording.
6. Continue to analysis with training scope. `assess_project` reuses a selected split or calls `create_split`; complete compatible declared partitions stay 6/2/2. It calls `api.readiness`, which persists the context snapshot and queues a job.
7. `JobManager._run` starts the subprocess. `worker.run` reconstructs the dataset and filters the six training images with `subset`, so test observations do not inflate training quality.
8. `assess -> _assess` compiles class/readability/annotation requirements, runs requirement checks and the declared diagnostics/UAV properties, then generates findings and the requirement verdict.
9. Cached pixel measurements feed `collect_measurements`; `aggregate_quality` produces supported normalized criteria and six dimensions. If no independent label audit, target scene/weather categories or object-size criterion was supplied, the relevant components remain unknown. A minimal requirement verdict can still be READY while dimension scores remain null.
10. The completed job stores Certificate. Shell polling finds completion. ReadinessSummary displays the goal, requirement result, evidence gaps, dimension overview and downloadable report. QualityAnalytics uses the saved report for drill-down/reweighting; it does not reread all images merely because a card was clicked.

The interpretation request body below is illustrative and matches MessageRequest; substitute the actual workflow ID in the route:

```json
{
  "message": "Detect drones in images.",
  "enable_remote": false
}
```

This minimal context is valid Pydantic input because omitted fields have defaults. It is an illustrative confirmed manual context, not unreviewed model output:

```json
{
  "original_text": "Detect drones in images.",
  "goal": {
    "task_type": "object_detection",
    "task_description": "Detect drones in images.",
    "required_classes": ["drone"],
    "input_modalities": ["image"]
  },
  "clarification_questions": [],
  "provider": "deterministic",
  "confirmed": true
}
```

To make a real nighttime requirement binding, explicitly state and confirm it and supply recorded lighting evidence. Do not label a dark thumbnail as proof of nighttime capture. To score object-size compliance, confirm the pixel-side criterion through the supported context field and supply annotated geometry. These changes refine what is measured; they do not guarantee a detector's recall.

## Example 2: train original and controlled-data models

Continue from the training-partition report. Choose blur at severity 0.25. This is an intervention magnitude; the variation adapter maps it to a blur radius, not a promised 25% accuracy drop.

1. `GuidedValidation.prepare` posts to `prepare_validation`. `variation` canonicalizes the original split and calls `research.variants.create_variant`, preserving validation/test bytes and marking artificial provenance.
2. The route creates the variant split and queues the changed training-subset readiness job under the same goal. It saves workflow_validation with dataset/split/readiness IDs.
3. After measurement finishes, inspect the actual criterion/dimension changes. A fully unscored dimension remains unknown, while measured component criteria can still show a change. The UI withholds score deltas if policies differ.
4. Select a training budget. The illustrative request below asks for one short CPU run per model, appropriate only to a small software exercise; it is not an accuracy-study recommendation.
5. `validation_protocol` creates candidate and reference TrainRequest objects sharing original validation and final-test partitions. `create_experiment -> validate_experiment` rejects overlap, class mismatch or inconsistent readiness identities before saving the configured record.
6. Review the frozen protocol, then POST `/experiments/{id}/start`. The handler revalidates, updates status and enqueues an experiment job.
7. The parent worker's `execute_experiment` runs original-data and modified-data training children, one mode/seed at a time. `train` verifies inputs, materializes caches, writes YAML, establishes the same target weights and invokes real YOLO training.
8. Training tensors/model/optimizer occupy process memory and the selected compute device. Image/label caches, configuration, logs and checkpoints are on disk. JSON job/model metadata is in SQLite. React stores only the current view and retrieved progress, not training tensors.
9. Epoch callbacks and `results.csv` feed progress/loss curves. `finish_training` validates best.pt and creates each model record with hashes/manifests/profile.
10. Final evaluation starts only after both models finish. If training fails, the parent retains completed child evidence and stops; it does not fabricate the missing second model.

```json
{
  "epochs": 1,
  "image_size": 64,
  "batch": 2,
  "seeds": [42],
  "device": "cpu"
}
```

The advanced direct-training equivalent uses the actual IDs and TrainRequest. These are illustrative placeholder IDs, not working IDs from a personal dataset:

```json
{
  "dataset_id": "DATASET_ID",
  "split_id": "SPLIT_ID",
  "readiness_id": "TRAINING_READINESS_JOB_ID",
  "architecture": "yolo11n",
  "initialization": "scratch",
  "model_type": "candidate",
  "epochs": 10,
  "batch": 4,
  "image_size": 512,
  "learning_rate": 0.001,
  "seed": 42,
  "device": "auto"
}
```

The presence of IDs is not enough: validate_training checks that the saved report belongs to this exact dataset and split. A report of the entire dataset is not accepted as training-only evidence for a chosen split.

## Example 3: compare two models and export the research evidence

1. In a named experiment, model IDs are retained in mode/seed execution order. In the advanced comparison page, the user selects real model records. Model A/B are comparison labels, not universal quality ranks.
2. Select a genuinely independent external test dataset or the held-out test partition. The advanced TestRequest below uses actual-model placeholders and default-compatible thresholds.
3. `api.model_evaluation` obtains the test manifest, validates one-to-one classes and checks every development manifest for overlap before queueing an evaluation job.
4. `evaluate_models` creates the canonical test inputs. Ultralytics validation supplies AP/PR curves; operating predictions supply repository TP/FP/FN and confusion evidence. Predictions are saved as JSON; each evaluation record references its model, test identity, protocol/settings, metrics, curves and environment.
5. `ComparisonPage` can align the same image only when the test protocol/ground truth/settings match. It draws truth, predictions and missed boxes. `BenchmarkCharts` uses saved per-class values and confusion counts.
6. `analytics.comparison` audits differences rather than silently claiming a controlled ranking. Different seeds, initialization, budgets, sizes, test settings or imported provenance are visible confounders.
7. `analytics.research` joins the model's readiness and dataset with the actual evaluation. Seed aggregation and independence/comparability rules determine whether coefficients can be reported. For one original/blur pair, expect a paired observation and insufficient evidence for an independent-source correlation.
8. The guided quality/performance tab requests only its experiment's model IDs through experiment_id. Advanced analytics can look at the whole collection. Exporting that guided tab's CSV targets `/reports/{experiment_id}`.
9. `api.report -> reporting.document` gathers selected evaluations, training configurations and readiness profiles. `html_document` or `pdf_document` renders deterministic stored-evidence reports; JSON keeps full structures. No external language model writes an unsupported empirical conclusion.
10. The browser downloads the response. If the second model never evaluated successfully, the report states incomplete outcomes rather than drawing a fabricated comparison.

```json
{
  "model_ids": ["CANDIDATE_MODEL_ID", "REFERENCE_MODEL_ID"],
  "dataset_id": "INDEPENDENT_TEST_DATASET_ID",
  "confidence": 0.25,
  "iou": 0.7,
  "match_iou": 0.5,
  "device": "auto",
  "acknowledge_missing_groups": false
}
```

Suppose the measured operating outcomes were TP=8, FP=2, FN=4 for A and TP=7, FP=3, FN=5 for B on the same labelled objects. Those illustrative values imply A precision 0.8 versus B 0.7 and A recall 8/12 versus B 7/12. They do not determine either model's AP, which requires the confidence sweep, and do not prove that a quality score predicts performance across independent collections.
'''

TEXT['ch15'] = r'''
## How to use the reference

The detailed cards below cover the functions that move information across major boundaries. Each names verified callers, inputs, behavior, output/consumer, side effects and failure paths. The following compact inventory covers the remaining source symbols, including test helpers, nested callbacks and TypeScript type aliases. It is generated with Python AST and the installed TypeScript compiler parser. Its source links resolve to the actual inspected file and line.

Static-call lists are syntactic references, not a complete dynamic call graph. Registry dispatch, callbacks, monkeypatching, runtime dictionary access and external library execution cannot be fully resolved by a simple AST. For nested Python functions, the collected parent subtree can include calls inside its nested definitions; the curated workflow diagrams show the actual mediated execution path. A missing return annotation is labelled as such rather than an invented type. Test functions assert behavior; they are not application handlers.

Data-model field references include actual annotations/default expressions and distinguish dataclasses, Pydantic, SQLAlchemy and TypeScript compile-time types. The Pydantic base's runtime validation does not automatically apply to arbitrary internal dictionaries or TypeScript aliases. JSON record payloads are validated where request/domain models are used; Store itself is a generic JSON store.
'''

TEXT['ch16'] = r'''
## Start with the boundary that failed

Use the workflow ID, job ID, dataset/split IDs and saved context before guessing at a chart or retraining. A current browser tab can hold stale draft selection while an older immutable report correctly reflects its own context. Inspect the record's exact scope, profile hash and test protocol. The table below maps symptoms to actual paths and the nearest tests; those are reading pointers, not instructions to run a GPU study for every UI issue.

## Where to extend the project

To add a metric, implement and register it in `metrics/core.py` (or the appropriate UAV calculation), declare applicability/method/evidence and return MetricResult. Add it to DIAGNOSTICS or an explicit Requirement if it should execute; registration alone does not make it run or become binding. For a disk-backed table, implement a full-data path or explicitly report unsupported; never substitute Dataset.frame's preview. If it should contribute to quality, add a component definition and a defensible normalization/applicability rule in the adaptive configuration, with version/provenance and tests for unknown/zero/gate behavior.

To add a source format, adapt bytes/records into the existing Dataset representation through the bounded ingestion path. Preserve parsing notes, source/hash ownership, label semantics, limits and cleanup. A new parser does not automatically qualify the dataset for detector training; training_eligibility, canonical annotation conversion and class mapping must support it.

To add a training task, a task weight profile is insufficient. Add a real request/configuration, validation, materialization, worker branch or adapter, result/provenance model, evaluation protocol and UI workflow. Keep final-test data outside checkpoint selection. Do not map tracking labels onto a detection model and call identity tracking validated.

To change the UI flow, start with Shell and the guided components; the advanced components remain reusable. Keep server-saved state authoritative and expose loading/error/prerequisite states. To extend storage, understand that all record kinds share research_records; an ID in JSON is not a relational foreign key. To change report wording, use saved evidence and preserve the distinction between requirement results, quality evidence and measured model performance.

To alter scoring semantics, change the appropriate profile/version and algorithm deliberately. Preserve frozen assessments and make policy migrations explicit. Scientific validation requires independent evidence; editing ratios or warning thresholds does not calibrate them. The research tools can investigate a policy, but their exploratory correlation output is not an automatic proof that it is valid.

<details>
<summary>Analogy: follow the specimen label before replacing an instrument</summary>

If two reports disagree, first verify that they refer to the same specimen, signed question, procedure and test population. A useful debugging ID chain is workflow -> confirmed context -> readiness job -> training manifest -> model -> evaluation -> research report. Replacing an instrument before checking that chain can hide the actual source of disagreement.

</details>
'''

TEXT['ch17'] = r'''
## Implemented, retained, optional and incomplete

The guided/persistent web workflow, bounded ingestion, registry measurements, requirement decisions, strict-coverage adaptive dimensions, recorded variants, real detector training/evaluation, saved protocols, scoped research joins and evidence exports are implemented paths. The original App and `/api` workflow are retained and reachable, with a different transient lifecycle. The standalone research CLI is reachable through `python -m backend.research` and deliberately requires `--execute` for actual training.

`ISOIECchecker` is retained but no application call path into it was found in the source inventory/import/call search. Its profile-only accuracy/consistency/traceability calculations are partly unimplemented and return None. `FeatureAnalysisResult.target_mutual_information` exists, but the analyzer records that a declared estimator configuration is required; it is not populated by a hidden MI model. Those facts are different from an optional implementation that is present but unconfigured.

Local ONNX vision is optional and requires trusted configured model paths, explicit preprocessing/task settings and the ONNX runtime. No downloaded default model was verified for this documentation run. Gemini and the compatible provider are implemented network adapters; live responses depend on credentials/provider availability and were not called for this task. Tests mock that boundary. A schema-valid model interpretation still requires human confirmation.

Readiness for classification, regression, anomaly detection, telemetry and tracking does not imply corresponding downstream training adapters. Disk-backed UAV summaries and some whole-table reference computations remain explicitly unsupported. General feature mutual information, physical sensor calibration validity, semantic label correctness, exhaustive annotation completeness and real target-model generalization require additional evidence/implementations. Canonical evaluation is not the official COCO crowd protocol. Named experiments do not have automatic arbitrary-step resume. Storage is local/single-user, with no added authentication or distributed scheduling guarantees.

## Architectural observations, not silent fixes

The two API families have different deletion, expiry, status and report lifecycles. An older document that describes only in-memory sessions or only the newest workbench is incomplete for the current codebase. Ambiguous selection has a richer original-UI wizard than the persistent importer. The generic JSON database has application-level ID relations, not enforced foreign keys. Module/task dispatch and callbacks make fully automatic call-graph claims unsafe. Project/source hashes identify bytes and configuration but do not authenticate declared collection metadata.

Primary dimensions intentionally become unknown when any positively weighted required criterion lacks evidence. Consequently a well-formed small labelled image dataset can pass the confirmed basic requirements while all six dimensions remain unscored. This is a consequence of evidence/criterion coverage, not a chart failure. Dimension weights adjust priorities but do not change within-dimension scores. The older partial arithmetic scalar is retained behind its own label. A new task preview invalidates context-dependent evidence and cannot substitute for reassessment.

The guided CSV export uses the current experiment; the general research export covers the workspace. The application PDF formatter explicitly abbreviates long structured values and points to JSON. This book's complete documentation PDF is a separate exporter with all paragraphs, references and diagram images. Source-only review cannot establish actual CUDA throughput, external model compatibility or predictive validity on independent real UAV datasets.

## Reading and regeneration

Read chapters 1-4 for the user/application map, 6-9 for data and scoring, 10-11 for real ML/statistics and 12-13 for persistence/failure behavior. Follow the three examples in chapter 14, then use the searchable reference and debugging map when working on a specific file. The HTML export works as a local file with embedded SVGs, syntax colors, search, sidebar navigation and diagram zoom. The PDF contains the same guide content with rendered diagrams, bookmarks, clickable contents and source links; interactive explanations are expanded in print.

The tools in `reading/tools` refresh the source inventories, assemble this Markdown, render its Mermaid blocks with a locally vendored renderer and export the same parsed Markdown to HTML/PDF. They do not import main.py, execute training, access environment-file values or mutate the application. The README beside this guide contains the exact commands. Rebuilding inventories after source changes updates line links/hashes; curated explanations still require human review when behavior changes.

## Verification scope

For this documentation task, verification includes source-file/symbol/line links, route declarations, diagram-source/render parity, actual Mermaid parsing/rendering, standalone offline HTML navigation/search/zoom, PDF destinations/links/images/page layout and content accounting across both exports. Application source hashes are checked against the pre-documentation snapshot. No expensive training or live paid API call is needed to verify the book.

Earlier work in this session reported 277 backend tests, 29 frontend tests and five browser tests passing, including synthetic CPU training. Those results are historical application evidence, not newly rerun tests or proof of real UAV accuracy. The export verification report records the checks performed specifically for this book, their counts and any remaining limits. Static analysis cannot prove that every possible dynamic branch has executed; the guide names that uncertainty instead of inventing runtime coverage.
'''

GLOSSARY = [
('API','An HTTP interface between the browser and Python, such as POST /api/v1/readiness.'),
('DTO','A JSON-friendly data transfer shape. Frontend aliases describe expected fields; backend models enforce their declared constraints.'),
('Pydantic model','A Python object that validates declared inputs, defaults and cross-field rules. In this project it is unrelated to a YOLO neural network model.'),
('Dataclass','A Python record-like class such as Dataset or DataProfile. Type annotations alone do not validate every assigned value at runtime.'),
('SQLAlchemy model','A Python mapping to a database table. Record maps the generic research_records table.'),
('Dataset snapshot','Saved source identity and parsed evidence for one dataset version. External registered files can still change and are rechecked.'),
('Goal','The user-confirmed task and requirements used as the finish line for assessment.'),
('Requirement','A specific comparison of a metric with an expected value, operator and severity.'),
('Diagnostic','A descriptive measurement without an implicit acceptance threshold.'),
('Evidence','The observed or declared material supporting a result, with method, denominator, scope and provenance.'),
('Insufficient evidence','A conclusion cannot be established from supplied observations. It is not the same as a measured failure or zero score.'),
('Applicability','Whether a metric or component makes sense for the selected task/modality.'),
('Normalization','Mapping a supported raw measurement to a declared 0-100 criterion scale; not predicting model accuracy.'),
('Weight','A configured importance ratio. Its meaning depends on whether it applies within a dimension or across dimensions.'),
('Geometric mean','A multiplicative summary represented by weighted logarithms; an actual positively weighted zero makes the result zero.'),
('Coverage','The weight fraction for which eligible criterion evidence is measured. Adaptive primary dimensions require full coverage.'),
('Gate','An independently evaluated critical check that score weights cannot disable.'),
('Provenance','The origin and processing history of data/artifacts; saved declarations are not automatically authenticated facts.'),
('Hash / SHA256','A deterministic content fingerprint used to identify changed bytes or configurations.'),
('Training partition','Data used to fit model parameters.'),
('Validation partition','Development data used for checkpoint selection and related model-development decisions.'),
('Final test partition','Held-out data used after training, excluded from checkpoint and hyperparameter selection.'),
('Leakage','Information or related observations crossing a boundary that should be independent, such as duplicated development/test images.'),
('Recording group','A declared flight/session/video unit kept together because frames from one collection can be dependent.'),
('Controlled variant','A documented artificial change to training data while preserving fixed validation/test evidence.'),
('Seed','A recorded random-state control. Repeated seeds on the same data do not create new independent datasets.'),
('Checkpoint','A saved model state; a resumable checkpoint can also retain optimizer/epoch state.'),
('CUDA','The GPU execution capability exposed by the installed PyTorch runtime, not merely a physical GPU label.'),
('IoU','Intersection-over-union of two boxes, used as a spatial matching criterion.'),
('TP / FP / FN','Matched true detections, unmatched predicted detections and missed labelled objects under a declared rule.'),
('Precision / recall / F1','Operating-point detection summaries with explicit denominators; undefined cases can remain null.'),
('AP / mAP','Confidence-sweep precision-recall area and its class/IoU averages, delegated to the versioned Ultralytics implementation.'),
('Confusion matrix','Counts of predicted versus true classes; the repository adds background and states its spatial matching method.'),
('Pearson / Spearman','Linear and rank-based association measures over comparable independent dataset observations.'),
('Confidence interval','An interval tied to a stated sampling/variation model; here seed intervals do not cover all domain uncertainty.'),
('Bootstrap','Resampling independent observations to estimate a statistic distribution. The code resamples datasets, not individual seed rows.'),
('Leave-one-dataset-out','Train a simple predictor on all but one independent dataset and evaluate on the held-out one, repeated for each dataset.'),
('MAE / RMSE','Mean absolute error and root mean squared error of held-out predictions.'),
('Polling','The browser repeatedly asks for current saved state through new HTTP requests.'),
('Subprocess','A separate Python process used for compute. An inline experiment child job is a record, not necessarily another subprocess.'),
('Immutable evidence','Historical snapshots are retained rather than relabelled as results of a new goal or procedure.'),
]

DEBUG = [
('Frontend cannot connect','Shell -> api -> /api/v1/datasets, /jobs, /compute. Check backend port, Vite /api proxy and whether dist existed at server import. Browser network errors become query alerts.','main.py; frontend/vite.config.ts; frontend/src/api.ts','frontend/src/workbench.test.tsx'),
('Upload or parser choice fails','DatasetUpload XHR -> api.upload -> import_file -> load_dataset_stream/load_container. Inspect detail, parser metadata, source format/limits. Original selection endpoint is richer than persistent import.','backend/workbench/datasets.py; backend/dataset_archives.py; backend/record_formats.py','tests/test_dataset_formats.py; tests/test_csv_encodings.py; tests/test_large_datasets.py'),
('Image labels or boxes look wrong','Image inspection -> format adapter -> canonical objects. Check class names/order, normalized versus pixel coordinates, VOC base and ignored-region semantics; inspect profile issues.','backend/image_ingestion.py; backend/universal_images.py','tests/test_visdrone.py; tests/test_dataset_formats.py'),
('Goal interpretation falls back','GoalChat -> project_goal -> interpret -> provider.extract. Provider failure is swallowed into an identified local fallback; review provider label/questions/configuration rather than expecting provider stack traces in browser.','backend/context_extractor.py; backend/workbench/workflow_api.py','tests/test_gemini_provider.py; tests/test_guided_workflow.py'),
('Unexpected metric value','Read saved job context/scope -> build_requirements -> run_metric -> pandas/disk/UAV implementation. Check exact parameters, output_path, unit, denominator and reference.','metrics/core.py; metrics/disk_metrics.py; metrics/uav_metrics/assessment.py','tests/test_readiness.py; tests/test_uav_metrics.py'),
('A score is unknown','aggregate_quality -> applicability -> _measure -> weighted coverage. Read the component reason and required evidence/criterion. Never replace missing measurement with zero or infer data from a chart.','metrics/quality_aggregation.py; metrics/config/adaptive_quality.json','tests/test_adaptive_quality.py'),
('Weights change importance but not score','Dimension overrides alter inter-dimension importance; component overrides affect the within-dimension mean. Gates are independent. Check frozen versus current profile and confirmed overrides.','metrics/quality_aggregation.py; backend/workbench/quality_api.py','tests/test_adaptive_quality.py; frontend/src/adaptiveQuality.test.tsx'),
('Training does not start','API -> validate_training/validate_experiment -> enqueue. Verify eligibility, exact training-only report, classes, split and source identities. A queued job needs the local manager and worker environment.','backend/workbench/detector.py; backend/workbench/jobs.py','tests/test_workbench.py; tests/test_validation_lab.py'),
('CUDA unavailable or out of memory','compute/device inspects installed PyTorch. Explicit GPU can fail; worker adds OOM guidance. Inspect worker.log, frozen config, batch/image size. No automatic protocol-changing retry occurs.','backend/workbench/detector.py; backend/workbench/worker.py','tests/test_workbench.py; backend/scripts/smoke_workbench.py'),
('Resume lacks a checkpoint','lab_api.resume -> retained run/weights/last.pt -> resume_training. At least a usable completed epoch/checkpoint is needed; preserve original protocol. A failed parent experiment is a different recovery case.','backend/workbench/lab_api.py; backend/workbench/detector.py','backend/scripts/smoke_resume.py; tests/test_validation_lab.py'),
('Final evaluation rejects data','model_evaluation/test_manifest -> class_mapping -> assert_independent. Check annotations, label schema, exact/near duplicate leakage and comparable recording groups.','backend/workbench/detector.py; backend/workbench/splits.py','tests/test_workbench.py; tests/test_validation_lab.py'),
('Comparison appears inconsistent','Check image+annotation test hash, evaluation thresholds, architecture/initialization/budget/seed and training sizes. Operating metrics and AP sweep are distinct. Predictions align only under matching evidence.','backend/workbench/analytics.py; frontend/src/Workbench.tsx','tests/test_validation_lab.py; frontend/e2e/workbench.spec.ts'),
('Research plot is empty','research joins saved evaluated models then filters independent comparable sources; check provenance, shared images/groups, frozen policies, repeated seed grouping and nonconstant data. At least 3 sources for coefficients, 6 for held-out prediction.','backend/workbench/analytics.py','tests/test_workbench.py; tests/test_validation_lab.py'),
('Report generation fails or differs','api.report -> reporting.document -> HTML/PDF. Verify record kind/state and scope. Original evidence, saved aggregation, whole workspace and selected experiment are different exports. Read full JSON for abbreviated application-PDF structures.','backend/workbench/api.py; backend/workbench/reporting.py; backend/reports.py','tests/test_validation_lab.py; frontend/e2e/adaptive-lab.spec.ts'),
]

from current_pipeline import update_manuscript
update_manuscript(TEXT, FILE_ROLES, GLOSSARY, DEBUG)

FILE_ROLES.update({
    'sql/models.py': 'SQLAlchemy Base and Record table schema for persisted research records.',
    'sql/datasets.py': 'Bounded DuckDB connection configuration for disk-backed datasets.',
})
TEXT['ch02'] += '''
## Current folder boundaries

The application now has three primary source folders: frontend/ for the React UI, backend/ for FastAPI, ingestion, context interpretation, orchestration and experiments, and metrics/ for profiling, metric execution, UAV-01 through UAV-20 and scoring policies. The sql/ package owns the actual SQLAlchemy table models, SQLite persistence and DuckDB connection configuration. Domain and request models remain with the backend. Reading material, documentation tools, examples and diagram assets live together under reading/.

The metric reading order is profiling, shared registry/dispatch, full-table or UAV evidence, quality policy and aggregation, then the optional experimental summary. The backend owns the confirmed-goal and readiness orchestration around those measurements. The UAV property identifiers retain their established numeric order; reorganizing source files does not change formulas, policy weights, gates or stored database records.

Start from the repository root using python -m uvicorn main:app, or backend/scripts/run_backend.ps1. Workers now use python -m backend.workbench.worker and standalone research uses python -m backend.research. The former demo_data directory was removed; optional sample generation targets .runtime/demo_data and automated browser fixtures use temporary directories.
'''
