# Adaptive UAV readiness and Model Validation Lab

The frontend now follows three guided stages: **Dataset & goal → Understand your data → Validate with models**. Upload first, review and save the goal interpreted by Gemini (or the local fallback), run the readiness pipeline, then configure and execute a controlled model comparison. Technical research pages are available under **Workspace tools**.

**[Guided workflow, Gemini setup, saved goals and browser verification](guided-workflow.md)**

The application now includes six evidence-aware quality dimensions, versioned task/scenario weighting, independent critical gates, a separate Model Validation Lab, named multi-seed experiments, shared validation data, complete combined-training assessment, matched-size protocols, actual benchmarking, research analytics and JSON/CSV/HTML/PDF exports.

**[Full handoff: architecture, changed files, mathematics, weight provenance, APIs, exact PowerShell startup and first real experiment](adaptive-readiness-and-validation.md)**

[Implementation audit and checklist](adaptive-platform-checklist.md) | [Executed verification and limitations](adaptive-platform-verification.md)

Weights and review thresholds are configurable engineering hypotheses, not calibrated predictions of accuracy. Missing evidence remains unknown. Existing raw metrics and original requirement-based decisions are preserved.

---

# FlightReady ? UAV Dataset Readiness and Model Validation

A local full-stack research workbench integrating the existing context-aware readiness pipeline with persistent dataset management, YOLO detector training, independent model evaluation, comparison, controlled variations, and research exports.

The guided project uses three primary destinations. The advanced workspace retains image/annotation previews, conservative context review, real job status/logs/loss curves, same-image prediction comparisons, and seed-aware research analytics. No composite readiness score or model result is invented.

**[Platform setup, complete workflow, model licensing, storage, CUDA status, tests and limitations](research-platform.md)** ? [Implementation plan](platform-implementation-plan.md)

[Measured dataset clarity versus model performance: mock experiment results, training metrics and limitations](quality-performance-experiment.md)

**Temporary feature:** [experimental weighted dataset score](experimental-weighted-score.md), with editable importance weights, component contributions, evidence coverage and read-only report previews. Defaults are in [metrics/config/experimental_score.json](../metrics/config/experimental_score.json); the readiness decision remains requirement-based.

Start the already configured environment:

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The frontend production build is served by FastAPI. On a fresh checkout, install `requirements/requirements-training.txt`, run `npm.cmd ci` and `npm.cmd run build` in `frontend` first. SQLite initializes automatically. The current installed detector runtime is CPU-only; GPU verification requires a compatible CUDA-enabled PyTorch build.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\pytest-check -o cache_dir=.runtime/pytest-cache
.\.venv\Scripts\python.exe backend/scripts/smoke_workbench.py --device cpu
# In frontend:
npm.cmd test
npm.cmd run build
```

The original assessment application remains at `/?legacy=1`; its historical session-storage limits below apply to the original `/api` endpoints. The default `/api/v1` workbench persists datasets, context, reports, jobs, artifacts and experiment metadata across restarts.

---

# FlightReady — Context-Aware UAV Dataset Readiness V1

A local Python/FastAPI + React/TypeScript research application for offline text, tabular and image datasets, including common Kaggle and GitHub download layouts. It evaluates a dataset against a **confirmed goal**, not a generic weighted quality score. The original project modules remain reusable and have explanatory comments.

The research extension adds the stable **UAV-01..UAV-20** property registry,
measurement/estimate provenance, six learning-potential views, optional local ONNX
vision, and independent controlled-experiment utilities. See the
[framework guide](uav-framework.md) for algorithms, input contracts, API
examples, model configuration, resource limits, experiment commands, and explicit
unsupported evidence. The existing upload/interpret/confirm/assess/report workflow
is preserved.

**This is a research dataset-readiness assessment, not an official aviation safety certification or a guarantee of model performance. Synthetic demonstrations do not validate the method on real UAV missions.**

## Setup (Windows PowerShell)

Python 3.12 and Node 22.12+ are recommended for the declared dependencies. From `C:\DataReadiness`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements.txt
Copy-Item .env.example .env
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
```

Start the backend in terminal 1:

```powershell
cd C:\DataReadiness
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2 for development:

```powershell
cd C:\DataReadiness\frontend
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to Python. After a frontend build, the backend also serves the complete application at **http://127.0.0.1:8000**, so a second server is optional. Restart Python after creating the first frontend build. API docs: **http://127.0.0.1:8000/docs**.

In the current Codex workspace, packages were downloaded to `.packages` because global installs were unavailable. To use that installation instead of a virtual environment:

```powershell
cd C:\DataReadiness
$env:PYTHONPATH = 'C:\DataReadiness\.packages'
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Run tests and regenerate synthetic demos:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe backend/scripts/generate_demos.py
```

If using `.packages`, use `python` with the `PYTHONPATH` above. `npm.cmd` avoids Windows PowerShell execution-policy restrictions on `npm.ps1`.

### Editor checks

`pyrightconfig.json` and `.vscode/settings.json` let Pyright/Pylance find the project-local `.packages` dependencies. New VS Code terminals also receive the matching `PYTHONPATH`; reopen existing terminals after changing these settings. For a fresh virtual environment, install `requirements/requirements-dev.txt` to include pandas type stubs matching pandas 2.3, then select that environment as the IDE interpreter.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements-dev.txt
npm.cmd exec --yes --package pyright -- pyright
```

The development stubs affect editor analysis only; they do not replace pandas at runtime.

## Walkthrough

1. Enter “Classify observations using label column label” and choose **Tabular data**.
2. Interpret the goal. Set task to **classification**, target to **label**, required inputs to **x**, required classes to **car, pedestrian**. Resolve any clarification messages after making corrections, then confirm.
3. Upload `demo_data/clean.csv`. Optionally enter provenance JSON **before** uploading.
4. Run the assessment. The progress panel shows backend events. Inspect requirements, expand metrics within the eight dimensions, and download JSON or HTML.
5. Change the required class to `bicycle`, confirm, and assess the same upload. Class absence now establishes `NOT_READY` despite the data remaining clean.
6. For imagery, enter the UAV car/pedestrian example, choose **Detect objects** and **Images**, then upload `demo_data/detection_valid.zip`. Its metadata supports daytime, urban, altitude-specific evidence. These drawings demonstrate the software only.
7. Upload `unknown_scenarios.csv` and require altitude coverage in the advanced editor: the result becomes `INSUFFICIENT_EVIDENCE` rather than assuming an altitude.

## Viewing and downloading an assessment

Once the progress panel says **Completed**, it includes a **Your assessment report is ready** area with **View results**, **Download HTML report**, and **Download JSON**. The same controls are available for NOT_READY and insufficient-evidence outcomes. Report-display loading/errors are shown in this panel; **Retry loading results** retries retrieval without rerunning the checks, and the direct download links remain available.

The current assessment ID is saved in the page's `?assessment=...` URL and the current tab's session storage. Reloading the frontend can retrieve that job and its report without uploading the dataset again, provided the backend still retains the session. A report can also be reopened with `http://127.0.0.1:5173/?assessment=ASSESSMENT_ID`. This ID storage does not persist the dataset or certificate in the browser. Keep the backend running until reports have been downloaded; backend restarts and session expiry still make in-memory certificates unavailable.

Frontend regression tests:

```powershell
cd C:\DataReadiness\frontend
npm.cmd test
```

## Context interpretation and confirmation

Text and categorized tags are passed separately. The deterministic fallback recognizes common task phrases, car/pedestrian synonyms, lighting/geography/setting, explicit altitude intervals, qualitative priorities, and some conflicts. It is intentionally conservative; it does **not** claim general natural-language understanding. Unsupported wording remains in the original text for manual review. All constraints can be entered manually without an LLM.

Optional language-model interpretation now defaults to **Gemini 2.5 Flash-Lite**, which Google lists with free-tier input/output tokens, subject to project eligibility and quotas. Obtain a key from [Google AI Studio](https://aistudio.google.com/apikey), then put these settings in your local `.env` and restart Python:

```dotenv
CONTEXT_PROVIDER=gemini
CONTEXT_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your_key_here
```

Keep the key in the backend `.env`, never frontend code or a `VITE_` variable. Only your task description and selected tags are sent to Google; dataset files and metadata stay local. Google's free-tier content may be used to improve its products. Use an eligible free-tier project and check [current pricing](https://ai.google.dev/gemini-api/reading/pricing) and [rate limits](https://ai.google.dev/gemini-api/reading/rate-limits); this code cannot determine your project's billing status or guarantee that a billing-enabled key is free.

Missing keys, quota errors, timeouts and invalid output fall back to local deterministic interpretation, with the provider status recorded in the context. There are no automatic retries or switches to a paid provider. `CONTEXT_PROVIDER=deterministic` disables remote calls. Pydantic validates generated JSON; ungrounded numerical constraints are rejected and user confirmation is always required. Tests mock Gemini HTTP responses; a live call requires your key and has not been performed.

Existing OpenAI-compatible setups remain supported: set `CONTEXT_PROVIDER=openai-compatible`, `CONTEXT_MODEL`, `CONTEXT_API_URL`, and `CONTEXT_API_KEY`. Older configurations with only `CONTEXT_MODEL` and no Gemini key continue to select that provider. No new Python dependency is needed for Gemini's REST API.

The UI exposes common fields and a full **Goal JSON editor** for all other constraints. Click **Apply JSON changes** before confirming. The original interpretation is stored separately by ID; confirmed context is copied into the assessment and never rewritten by the model. Clearing a clarification message is an explicit user assertion that it has been resolved.

`field_sources` marks extracted fields explicit, inferred, or unspecified. This is extractor traceability, not a proof of semantic grounding. Free-text task descriptions and qualitative preferences are not automatically measurable requirements; inspect and confirm the interpretation.

## Supported data

- **Delimited tables:** CSV, TSV, TAB, PSV and delimited DATA/DAT files; comma, tab, semicolon and pipe detection, with manual delimiter and header controls. Headers must be unique and row lengths consistent. UTF-8 is preferred; UTF-16/UTF-32 with a byte-order mark are detected, with recorded Windows-1252 fallback for legacy text. `csv_encoding` overrides encoding. Invalid bytes are never silently discarded. CSV blank cells and default pandas NA tokens are missing. Zero-based data-row references exclude the header.
- **Text and records:** TXT/TEXT/MD/RST/LOG (one line per record by default), JSON, JSONL/NDJSON, XML, XLSX worksheets and Feather/Arrow IPC. JSON/XML paths and worksheet choices resolve ambiguous files. Nested JSON/XML fields are retained as JSON text; scalar type checks on those fields report insufficient evidence. Plain text uses a `text` column; folder corpora use one document per file with `text`, `label` and `source_file`.
- **Parquet:** standard `PAR1` files, using PyArrow. Scalar text, numeric, boolean and timestamp columns are supported; nested/binary tables are rejected by the disk backend. Large or highly compressed tables are read in batches, without the former 500,000-row ceiling.
- **Telemetry:** CSV/Parquet with explicit timestamp, group/session, unit and sensor mappings. No platform constraints are guessed. Declare `valid_ranges` for defensible physical specifications.
- **Images:** standalone static images; ImageFolder classification; unlabelled collections; YOLO with classes.txt, .names or YAML names; native **VisDrone DET**; COCO detection JSON; Pascal VOC XML. ZIP and TAR-family archives can contain a surrounding repository folder. Multiple layouts require an explicit selection. COCO segmentation masks, video/VID/MOT and tracking are not evaluated.
- **Containers:** ZIP, TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2 and TAR.XZ/TXZ; standalone GZ/BZ2/XZ compression. Choose between candidate tables, image layouts or a text corpus without uploading again. Repository scripts are never executed.

See [format guide and parser options](formats.md) for layouts, semantics and limits. The source website does not determine a dataset's format: unusual exports may require explicit parser settings or conversion. This does not promise every possible binary/custom format.

YOLO ZIP:

```text
classes.txt                # one exact class name per line, indexed from zero
images/train/a.png
labels/train/a.txt         # class_id x_center y_center width height
metadata.csv               # optional, keyed by full path: images/train/a.png
```

YOLO coordinates are normalized; width/height must be positive and the entire box must lie inside the image. An empty annotation file is a declared negative image. Missing files are unknown annotations. Missing objects *inside* an otherwise valid annotation file cannot be detected from syntax. Invalid YOLO files contribute no class-coverage evidence; native VisDrone uses the row-specific policy below. Class names are matched exactly; edit names after interpretation to match the dataset.

Native VisDrone DET ZIP:

```text
VisDrone2019-DET-train/
    images/0000002_00005_d_0000014.jpg
    annotations/0000002_00005_d_0000014.txt
```

Each annotation row has eight comma-separated ground-truth fields: `left, top, width, height, score, category, truncation, occlusion`; a trailing comma is accepted. Pixel boxes are checked against the image dimensions before conversion into normalized internal coordinates. Score must be 0/1; truncation 0/1; occlusion 0/1/2. Prediction scores and ten-field tracking annotations are not treated as valid DET ground truth.

Classes 1–10 are `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`. **Pedestrian and people stay separate.** Score-zero boxes, ignored regions (category 0) and others (category 11) are tracked separately and excluded from target counts. Zero-area ignored regions produce diagnostics; a zero-area target box is an annotation validity failure. Valid target rows are retained when another row in the same VisDrone annotation file is invalid. Scenario-image counts still require a completely valid annotation file. These semantics follow the [official DET toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit#readme).

The archive is read from its staged path, one member/image at a time, without extracting or loading the full ZIP into RAM. All images are decoded, exact hashes are computed, and every supported annotation file is checked. Near-duplicate pairwise comparisons are skipped with an explicit insufficient-evidence state above `MAX_NEAR_DUPLICATE_IMAGES`; no perfect score is substituted. If an archive contains multiple datasets or layouts, choose one before assessment. Lighting, altitude, location and independent evaluation coverage are not inferred from the dataset name or image appearance.

Example `metadata.csv`:

```csv
image,lighting,geography_type,altitude_meters,split,flight_id
images/train/a.png,daytime,urban,50,train,flight-A
```

Metadata cannot overwrite computed image fields. Supported static images include JPEG, PNG, WebP, BMP, TIFF, GIF, JFIF, PNM and AVIF when the installed Pillow decoder supports them. Multi-frame images are rejected explicitly. Exact file duplicates use SHA256; optional difference-hash near-duplicate candidates use `NEAR_DUPLICATE_DISTANCE` (1–64; 0 disables).

## Large datasets: up to 2 GiB

Uploads accept **2,147,483,648 bytes (2 GiB)** by default, with format-specific decoded-content limits. XLSX has an additional default 256 MiB limit (`MAX_XLSX_MB`). Update an existing `.env` if it still sets the previous image/ZIP limits, then restart the backend:

```dotenv
MAX_UPLOAD_MB=2048
MAX_IMAGE_UPLOAD_MB=2048
MAX_ZIP_MB=4096
MAX_ZIP_FILES=50000
MAX_ANNOTATED_OBJECTS=1000000
MAX_NEAR_DUPLICATE_IMAGES=2000
LARGE_DATASET_THRESHOLD_MB=16
DATASET_MEMORY_MB=256
```

Requests stream through the HTTP middleware; large multipart files spool to disk. The loader reads at most 1 MiB per copy operation, computes SHA256 incrementally, ingests bounded batches, and stores a DuckDB database locally. The database memory setting applies to DuckDB working memory, **not** a hard process-RAM cap: Python, Arrow, request buffers and decoded rows also use RAM. Leave at least three times the upload size free; compressed Parquet, sorting, concurrent uploads and high-cardinality checks can need more. Hardware, field width and query complexity affect runtime.

Large tables keep only the first 100 rows as a preview. Missingness, numeric summaries, required fields/types/ranges, class/scenario coverage, exact duplicates, group/split overlap and supported annotation-table checks use the **full database**, including records beyond the preview. No sampled preview establishes a hard-requirement pass. Distinct-value dictionaries and duplicate checks are deferred until requested rather than invented in the initial profile. Unsupported disk-backed metrics, notably reference distribution distances, explicitly report insufficient evidence. Large-data timestamp checks require ISO-compatible timestamps; non-ISO evidence remains unverified.

The disk backend supports up to 512 columns and decoded text records/documents up to 8 MiB; record size and Parquet row-group width also affect memory. Archives allow up to 4 GiB total expanded contents and 50,000 entries by default, with at most 20 MiB per image and 20 megapixels/image. Image pixels are decoded one image at a time, but image metadata and parsed object annotations remain in RAM; the configurable object cap bounds this separately. New text/record readers use bounded batches and full-dataset disk metrics, not just the preview. These limits reject oversized records rather than silently truncating them.

Use the built app directly at **http://127.0.0.1:8000** for large local uploads, or the development frontend at port 5173. `/api/config` exposes the server's configured byte limits. New API/engine dependencies are declared in `requirements/requirements.txt` (including DuckDB).

Optional reproducible capacity check, using only generated synthetic data:

```powershell
# Install development dependencies in your chosen environment first.
$env:PYTHONPATH = 'C:\DataReadiness\.packages'
python backend/scripts/validate_large_upload.py --mib 2048
```

This creates an exactly 2 GiB CSV, starts a temporary local backend, streams a real HTTP multipart upload, checks all 2,097,152 rows, verifies a required class appearing only in the final row and exact duplicates, records peak server RSS and timing under `.runtime/large-upload-validation.json`, then removes its generated files. It needs at least four times the generated file size free. This verifies software capacity on synthetic data, not scientific readiness of real UAV datasets.

## Decisions and scientific scope

There is no weighted average and no hidden numerical training-readiness threshold.

| Status | Meaning |
|---|---|
| `NOT_READY` | A confirmed hard requirement failed. This establishes the decision even if other evidence is missing. |
| `ASSESSMENT_ERROR` | A calculation/requirement evaluation failed, without an already established hard failure. |
| `INSUFFICIENT_EVIDENCE` | A hard requirement cannot be established, or the goal is unspecified. |
| `CONDITIONALLY_READY` | Hard requirements passed; a preferred requirement failed or lacks evidence. |
| `READY` | All compiled hard requirements passed and preferred requirements have no outstanding findings. Scope is limited to those requirements. |

Presence of a required class/scenario logically requires at least one usable observation; this is **presence only**, not sufficient training size. Minimum training/scenario sample counts must be supplied by the user. Complete required-column presence does not imply zero missing values: set `max_missing_fraction` explicitly. Optional absent features do not fail readiness. Detection requires readable files and syntactically valid, present annotation files; explicit negatives are allowed.

Diagnostics report measured values with no implicit pass/fail threshold. A metric's `pass` means its calculation produced evidence, not that the dataset met an unstated quality criterion. Requirement findings apply comparisons separately. `None`, `False`, numeric zero, insufficient evidence and not applicable remain distinct.

## Documentation

- [Architecture, extensions, API and operations](architecture.md)
- [Context schema and examples](context-schema.md)
- [Metric definitions, formulas and support limits](metrics.md)
- [Audit and verification record](verification.md)

## Known limits

- Local, single-user, single-process tool. No authentication, shared tenancy, production hosting or aviation safety use. Bind to loopback. Large tabular uploads use private temporary databases under `.runtime/datasets`; small datasets, interpretations, certificates and job records live in memory. Session entries expire lazily after an hour and are lost on restart. Delete-upload, expiry and graceful-shutdown cleanup remove owned databases; active assessments protect their datasets from deletion. Abruptly terminating the process can leave temporary files. Do not use multiple Uvicorn workers.
- No verified label correctness without declared trusted references; annotation independence and metadata provenance are user-declared. No automatic external provenance authentication.
- No automatic Fleiss/alpha, multiple-box annotation matching, learned image quality, model performance evaluation, full temporal leakage proof, sensor calibration, or continuous drift. These are explicit limitations, not invented scores.
- No confidence intervals are invented: independence, sampling design, and within-flight correlation are not established. Audit coverage and observed counts are descriptive. Reference comparisons support selected columns; reference relevance must be independently justified. Subgroup comparisons can be expressed as scenario counts; automated distribution comparisons within subgroups are not implemented.
- Prompt-injection protection uses role separation, no execution/tool capabilities, strict schemas, numerical grounding and a narrow override-text filter. It cannot guarantee detection of every adversarial instruction; always review interpreted requirements. Metadata stays out of model prompts.
- Cancellation is cooperative between stages, not interruption of an individual pandas/Pillow/SQL calculation. Upload progress counts actual transferred bytes; 100% transfer is followed by server ingestion/profiling, which can take minutes. File validation/profile is synchronous during the upload request; assessment runs in a bounded thread pool. Cancelling a transfer cannot undo server work on a file already completely received.
- HTML/JSON exports are deterministic. PDF and LLM-generated narratives are not implemented. Browser-level visual QA may depend on available tooling; API workflow tests and TypeScript builds are provided.
