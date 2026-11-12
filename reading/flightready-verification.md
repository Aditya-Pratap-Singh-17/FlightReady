# Verification: 10 November 2026

This records the initial platform implementation. Subsequent model-import fixes
and actual pretrained-model verification are recorded separately in
[Model Lab import repair](model-import-fix.md).

All checks used isolated test storage and `PYTHON_DOTENV_DISABLED=1`. The user's
credentials, production datasets and existing saved project store were not read
or changed. The repository initially had all application files untracked; no
commit or destructive Git operation was performed.

## Automated results

- **345 backend tests passed** on the final implementation. This includes the
  existing ingestion, UAV metrics, context, aggregation, reporting, saved-project,
  queue, split/leakage, controlled-variation and validation tests plus new
  compatibility, canonical report, acquisition and real-inference tests.
- **35 frontend tests passed**, including early compatibility feedback,
  executive verdict, evidence disclosure, immutable export links and interpreter
  fallback provenance.
- TypeScript and the production Vite build passed. The application build remains
  served by the existing FastAPI static mount.
- The guided browser journey passed with actual upload, goal interpretation and
  confirmation, project reload, readiness assessment, controlled blur assessment,
  **two real one-epoch YOLO11 CPU training runs**, common validation data and final
  held-out evaluation.
- **Three browser tests passed** in the final combined run, including the guided
  training journey and desktop/mobile canonical reports, linked
  measured outcomes, unchanged assessment hash, model download consent gates and
  VisDrone validation imagery rejected as drone-target evidence before analysis.
  The later inference CSV export check also passed on real fixture inference.

Final evidence logs are under `.runtime/`:

```text
flightready-final-backend.log
frontend-final-tests.log
frontend-final-build.log
flightready-final-e2e.log
```

Pytest's default Windows temporary root was unavailable under the sandbox. Tests
were rerun with a workspace `--basetemp`; no product permission change was
needed. The final backend run has one existing Starlette/httpx deprecation
warning. The initial experimental-score test exposed a missing default JSON
configuration; the documented preset was restored with an explicit new version.

## New correctness coverage

- Native VisDrone's ground-object taxonomy does not support drone targets.
- Missing night metadata is neither zero night coverage nor confirmed absence.
- Validation-only sources permit evaluation but cannot become training splits.
- Explicit training-only manifests require an external validation source.
- Original technical `READY` can coexist with canonical `INDETERMINATE`; missing
  independent/quality evidence is not hidden by technical validity.
- Conservative geometric bounds preserve unknown weights, actual zeros and
  legitimate inapplicability.
- JSON, HTML and PDF use the same assessment fingerprint, version and verdict.
- Strict download consent, license review, full pinned revisions, safe file
  allowlists, failed downloads and artifact-integrity checks.
- Local-cache activation uses the supported Hub API with local-only resolution.
- Actual safetensors YOLOS inference runs on ten labeled images in an isolated
  fixture, persists its real predictions and metrics, reopens after store
  restart and links to the unchanged readiness assessment.
- COCO AP is checked against exact perfect-detection and empty-prediction
  examples; duplicate predictions are checked against TP/FP/FN expectations.
- Scientific conclusions pair the same seed and held-out protocol, calculate
  absolute/relative changes and withhold unknown quality deltas. Different test
  identities cannot support a paired comparison.

## Visual QA

Desktop and 390-pixel mobile screenshots were inspected. Browser checks assert
no horizontal overflow. Status labels, solid/dashed prediction boxes and text
make interpretation independent of color. Research details use disclosures;
unscored dimensions have no empty score graph.

The canonical PDF was rendered with PyMuPDF because Poppler was unavailable in
this environment. The executive page and appendix contact sheet were inspected;
the executive content fits on one page, without clipping or overflow. HTML/PDF
appendices retain original measurements and limitations. JSON is the complete
machine-readable evidence artifact.

The illustrative exported report is `output/pdf/flightready-readiness-example.pdf`.
Its companion HTML/JSON and page previews are generated from the same persisted
synthetic assessment. They are software examples, not UAV research findings.
Screenshots and render evidence are in `.runtime/flightready-verification/`.

## Reproduce

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED='1'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime/pytest-verification -p no:cacheprovider
Set-Location frontend
npm.cmd test
npm.cmd run build
```

The real safe-inference integration test requires
`requirements/requirements-inference.txt`; its tests explicitly skip if the
optional runtime is absent. That runtime was installed and those tests actually
executed in this verification. The YOLO browser path requires the training
runtime.

For the browser journey, start an isolated backend, generate the existing
synthetic detection fixture, and set `WORKBENCH_URL` and `GUIDED_FIXTURE`. See
`frontend/e2e/guided-workflow.spec.ts` and
`frontend/e2e/connected-research.spec.ts`. The latter expects a completed guided
study and a generated native VisDrone fixture. The verification run used ports
8057/8058 with `.runtime/flightready-verification/storage`, not the user's store.

Fresh-store browser verification (run the guided study before the linked-results
checks):

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED='1'
$env:CONTEXT_PROVIDER='deterministic'
$env:WORKBENCH_ROOT='C:\DataReadiness\.runtime\flightready-verification\storage'
.\.venv\Scripts\python.exe -m backend.scripts.connected_research_fixture
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8058
# In a second PowerShell terminal:
Set-Location C:\DataReadiness\frontend
$env:WORKBENCH_URL='http://127.0.0.1:8058'
$env:GUIDED_FIXTURE='../.runtime/flightready-verification/synthetic-drones.zip'
npm.cmd run test:e2e -- guided-workflow.spec.ts
npm.cmd run test:e2e -- connected-research.spec.ts
```

## What this verification does not establish

No pretrained model weights were downloaded. The local-cache/acquisition paths
used generated, safe test fixtures; remote error paths were controlled test
doubles. Actual inference used a tiny, randomly initialized YOLOS model. The
model registry pins a published pretrained revision for a subsequent deliberate
user download, but its accuracy on a real user dataset has not been measured.

No CUDA hardware run, official VisDrone/COCO ignore/crowd benchmark, verified
nighttime subset experiment, label-audit authenticity validation or empirical
calibration of readiness weights was performed. A one-epoch synthetic comparison
does not demonstrate useful drone detection, significant improvement,
predictive validity of readiness scores or causation. These limits are also
documented in the application and model reports.
