# Audit and verification

## UAV research extension (engine 1.4.0)

Verified on 2026-10-31 in the existing repository environment:

- Baseline after resolving environment-only blockers: **154 backend tests passed**.
  Initially `.venv` lacked the already-declared PyYAML dependency; it was installed.
  The default pytest temp/cache locations were denied, so writable workspace
  locations were explicitly selected. No product behavior was changed to bypass
  these environment restrictions.
- Final complete backend suite: **198 passed**, one existing Starlette/httpx
  deprecation warning, 34.15 seconds. This includes synthetic known-value tests for
  all deterministic UAV calculation paths, registry/serialization, explicit
  criterion versus diagnostic status, missing metadata/model evidence, malformed
  labels, pixel proxies, object geometry, references, group/content overlap,
  temporal/sensor checks, bounded spatial/model batches, optional local-model
  caching, full HTTP confirmation/assessment/export, source privacy, and provider
  rejection of invented UAV thresholds.
- Frontend: **11 tests passed**; strict TypeScript and Vite production build passed.
  New coverage verifies measurement-status labels and learning-view rendering.
- Research: seeded variants are byte-reproducible; train-only interventions preserve
  validation/test image and annotation bytes, reject known group/content leakage,
  and refuse output overwrite. Readiness-feature/CSV export and mocked optional
  detector orchestration pass. Mock outputs are software fixtures, not benchmark
  evidence. Performance fields are empty before actual training.
- Python compilation and `python -m backend.research --help` passed. The synthetic example
  JSON/HTML reports were generated from the existing demo ZIP without changing it.

Reproduce:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .runtime/pytest-your-new-run -o cache_dir=.runtime/pytest-cache-uav
cd frontend
npm.cmd test
npm.cmd run build
```

Choose a fresh `--basetemp` directory: pytest manages/removes that directory.
No pretrained model was downloaded, no real detector training was launched, and no
real-model accuracy, calibration, metric importance, causal performance effect, or
multi-GiB stress-run result is claimed. Real independent annotated estimation
fixtures remain an external research prerequisite. Follow the
[independent evaluation protocol](vision-evaluation-protocol.md) and
[framework guide](uav-framework.md) for explicit unsupported components and limits.

## Gemini context provider

The Gemini REST adapter uses JSON mode, local Pydantic validation and the existing numerical-grounding checks. It sends only user task text/tags, forces confirmation off on generated contexts, and makes one request without automatic retries or alternate-provider calls. Missing credentials and quota/permission/schema errors retain the deterministic workflow. The key stays in a request header and backend environment, outside URLs and frontend code.

Nine mocked HTTP regression cases cover request shape, restored original inputs, confirmation reset, quota and authorization failures, blocked/truncated output, invalid schema, invented thresholds, disabled/missing-key operation and legacy provider selection. No live Gemini request was made because a user API key has not been supplied; free-tier eligibility and billing depend on the Google project.

## Multi-format import extension (engine 1.3.0)

Readers now cover delimited tables, line/document text, JSON/JSONL, XML, XLSX, Feather/Arrow IPC, compressed files and dataset archives; image adapters add ImageFolder, COCO detection, Pascal VOC, YAML/.names YOLO and unlabelled/static images. Ambiguous archive members, JSON arrays and workbook sheets use a staged selection API without another upload. The UI displays the schema and offers an explicit suggested-goal action that clears confirmation.

The complete backend suite passes **145 tests**, including 43 new format cases. These cover schema evolution, preserved identifiers/types, legacy encodings, worksheet and archive selection, XML entity rejection, YAML code rejection, archive paths/links, nesting/record guards, image corruption, coordinate conventions, missing manifests, inferred labels and API upload/selection/assessment/HTML+JSON exports. **10 frontend DOM tests** pass, including three new upload/selection/context-confirmation tests. Strict TypeScript/Vite build and Pyright pass. The two backend warnings come from existing dependency deprecations in anyio/Starlette.

Format fixtures are synthetic regression examples, not a claim that every Kaggle/GitHub dataset was tested. The previously recorded 2 GiB CSV and real VisDrone capacity runs remain separate measurements; those stress runs were not repeated for this extension. New record readers use batched disk storage, but format-specific bounds still apply. See [the format guide](formats.md).

## Original repository audit

All six original source files were inspected before editing. No existing tests, dependencies, frontend or application implementation were present; `main.py` was empty.

- `backend/context.py`: useful Goal contract retained/extended with validated requirements, scenarios, evidence, findings and certificate models. Required/optional/excluded overlap validation preserved.
- `metrics/dataProfiler.py`: fixed `isna.sum()`, `nuinque()`, and `select_dtype(includes=...)`. Finite numeric summaries are JSON-safe; empty and duplicate-column tables are rejected.
- `metrics/featureAnalyzer.py`: moved missingness method out of `__init__`, implemented feature diagnostics, retained original spelling and supplied `FeatureAnalyzer` alias.
- `metrics/labelAnalyzer.py`: added complete-pair Cohen kappa/raw agreement and empty-label handling. Agreement is explicitly distinguished from correctness.
- `metrics/IsoIecChecker.py`: corrected missing traceability evidence from zero to `None`. This compatibility module retains ISO-inspired descriptive checks; it is **not** the readiness decision engine or an ISO conformance certification. Legacy checks requiring unavailable evidence remain `None`; the main engine provides typed evidence states instead.

## Automated checks

`tests/test_readiness.py` covers schema validation, confirmed-context boundary, profiler regressions, optional inputs, nonfinite values, duplicates, class distributions, kappa/degenerate kappa, detection files/bounds/negatives, explicit ranges/units, class/scenario intersection, unknown coverage, reference comparisons, split/group overlap, status precedence, required metric exceptions, safe report escaping, prompt override rejection, file security, repeatability, Parquet and the complete HTTP interpretation/upload/confirm/assessment/export workflow.

Research cases A–L in the task brief have named tests. Malformed path tests construct raw ZIP entries explicitly. Windows ZipInfo normalizes backslashes during filename handling; ingestion checks orig_filename as well as the normalized path so the raw malicious name cannot evade the guard.

Frontend verification uses strict TypeScript compilation plus a Vite production build. The built app and API are checked over local HTTP. Dependency auditing uses `npm audit`; package-lock.json records resolved frontend dependencies.

The suite uses synthetic data. No real deployment research validation, live provider credentials, model training, aviation safety evaluation or large-scale load test is implied by passing software tests. Browser visual/interaction automation is not part of this suite.

## Final execution record — October 28, 2026

- `python -m pytest -q`: **50 passed**, two dependency deprecation warnings (AnyIO/pytest and Starlette/httpx); no failures.
- `npm.cmd run build`: strict TypeScript check and Vite 7.3.7 production build passed.
- `npm.cmd audit --audit-level=moderate`: **0 vulnerabilities** reported after patching Vite.
- Local Uvicorn startup succeeded; `/`, `/api/health`, and `/openapi.json` returned **HTTP 200**. Frontend HTML contained the application title.
- The full API integration test confirms separate initial/confirmed context, real progress, profiling, readiness decision, JSON download and HTML download.
- No live LLM call or automated browser interaction was performed. The optional provider is covered by unavailable-provider and ungrounded-output tests.

## Follow-up editor and runtime repair

Whole-project Pyright checking exposed unresolved project-local imports, imprecise literal/Series types, and an argument-name collision that crashed assessments with `expected_types`. The collision is fixed and has a regression covering both valid and invalid integer types. Matching pandas stubs and workspace import-path settings support reproducible editor checks without disabling diagnostics.

After this repair, Pyright reports **0 errors and 0 warnings**; the regression suite contains **51 passing tests**.

## Legacy CSV encoding repair

The supplied CSV initially inspected as 161,892 bytes, 25 columns and 405 data records, with consistent record widths. UTF-8 decoding failed on byte 0x92 (a Windows-1252 smart apostrophe). The loader now decodes supported encodings strictly, records fallback decisions and never drops undecodable bytes or malformed records.

Fifteen additional synthetic cases cover legacy punctuation, UTF-8/BOM/UTF-16/UTF-32, explicit encoding overrides, invalid declarations, binary controls, retained row-width checks and a multipart Windows-1252 upload. **66 tests pass**, and Pyright reports **0 errors and 0 warnings**. The original download was unavailable at its supplied path during final verification; the fixed loader has not been revalidated against that exact file. No original dataset content was copied into the repository.

## 2 GiB tabular capacity implementation and verification

Large CSV/Parquet ingestion now streams request bytes, spools uploads to disk and ingests bounded Arrow batches into DuckDB. The assessment engine queries the entire disk-backed table; preview rows never establish a whole-dataset requirement pass. The image ZIP limits are preserved separately. Class, scenario, validity, duplication and split/group checks have whole-table implementations; unsupported whole-table reference comparisons remain unverified.

Final automated verification: **84 tests pass**, with the same two dependency deprecation warnings; Pyright reports **0 errors and 0 warnings**. The React/TypeScript production build passes. Added tests cover late-record failures, whole-table class counts, cross-batch missingness/duplicates, native numeric and boolean type checks, annotations, SQL identifier/value safety, reference-evidence limitations, stream read bounds, size-limit cleanup, disk deletion, Parquet, API limits and repeatability.

An actual local HTTP multipart upload used an exactly **2,147,483,648-byte** synthetic CSV with **2,097,152 data records**. All rows loaded, the class present only in the final record was found, and exact full-dataset duplicate counts were verified. The complete synthetic upload/profile/assessment run took **100.28 seconds** and observed peak server RSS of **521,248,768 bytes (about 497 MiB)** with a 256 MB DuckDB working-memory setting. Python/Arrow/library overhead is additional to that setting. This run used repetitive generated payloads; diverse/wide data, compressed expansion, disk speed and high-cardinality queries can have different resource requirements.

The reproducible command is `python backend/scripts/validate_large_upload.py --mib 2048`. Its temporary server, generated source and owned database were removed afterward. Measurement JSON remains under ignored `.runtime/large-upload-validation.json`. This is a software-capacity check, not evidence of real UAV model performance or mission readiness.

## Native VisDrone DET archive support

The actual supplied `VisDrone2019-DET-train.zip` was **1,549,875,511 bytes**, with **12,945 entries** and **1,556,080,035 expanded bytes**. Previous image-only limits (25 MiB / 2,000 entries / 100 MiB expanded) and YOLO-only/root-only expectations rejected it. Image archive defaults now allow 2 GiB uploads, 50,000 entries and 4 GiB expanded contents while retaining path, compression-ratio, per-member and decoded-pixel guards. ZIPs are passed by staged path and processed one member at a time.

The native eight-field DET parser follows the [official VisDrone toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit#readme), preserves category/score/truncation/occlusion, accepts observed trailing commas and keeps ignored regions separate from evaluated classes. Class 1 pedestrian and class 2 people are not merged. The root dataset folder is recognized without manual repackaging.

A real HTTP upload/profile/assessment/JSON-export/HTML-export test processed the entire supplied archive in **160.11 seconds**. **All 6,471 images decoded**. It inspected **353,550 annotation records**, retained **343,204 valid target boxes**, and excluded **10,345 ignored annotations** (8,813 ignored regions and 1,532 others). It found one invalid target box, a car with zero height at `annotations/9999985_00000_d_0000020.txt:12`, plus two zero-area ignored-region diagnostics. The verification goal therefore correctly returned **NOT_READY** for annotation validity, with no metric crashes. This is a test goal checking car/pedestrian annotations, not the user's confirmed deployment goal.

Observed target counts: pedestrian 79,337; people 27,059; bicycle 10,480; car 144,866; van 24,956; truck 12,875; tricycle 4,812; awning-tricycle 3,246; bus 5,926; motor 29,647. The original download was read only; validation summaries are stored under ignored `.runtime/visdrone-validation.json`.

The full regression suite now contains **102 passing tests**, including native/wrapped DET, wrapped YOLO, ignored categories and flags, invalid target/ignored geometry, trailing commas, per-file streaming, resource bounds, API recognition and corrupt-ZIP error handling. Type checking and the React production build pass. No inference of deployment lighting, altitude or geographic coverage is made from the dataset name or pixels.

## Completed-report display regression

The UI could show every completed stage without the report: the progress effect set job status to completed, triggering its own cleanup while awaiting the certificate request, and its disposed guard then discarded the successful certificate response. Progress polling and terminal report loading now have independent effects and abort controllers. Requests do not overlap, stale previous-job reports are discarded, and report retrieval has dedicated loading/error/retry state.

The completion panel always exposes direct HTML/JSON download links when a certificate is available on the server, even if inline retrieval fails. The frontend remembers the job ID in the query string/tab session storage and can restore it without another upload while that backend session remains alive. No backend restart or dataset changes are required for this frontend fix.

`npm.cmd test`: **7 React DOM regression tests pass**, exercising delayed completion/report arrival, already-complete jobs, report failure/retry with visible downloads, frontend reload recovery, assessment-error certificates, expired jobs and late stale responses. `npm.cmd run build`: strict TypeScript and Vite production build pass. These tests use controlled HTTP responses in jsdom; they are not a claim of visual browser automation.
