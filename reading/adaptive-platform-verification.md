# Adaptive platform verification

Verified on November 2, 2026 (America/New_York). Checks used isolated, explicitly synthetic fixtures. `.env` was not opened; test processes disabled dotenv and remote interpretation.

## Results

| Check | Result |
|---|---|
| Original backend/frontend baselines | 242 Python tests and 22 frontend tests passed |
| Full extended backend suite | **270 passed**; one existing FastAPI/Starlette test-client deprecation warning |
| Final targeted Lab suite after matched-size validation guard | **15 passed** |
| Frontend component/integration suite | **26 passed**, 5 files |
| TypeScript and Vite production build | **Passed**; nonfatal dependency `use client` notices |
| Microsoft Edge Playwright browser suite | **4 passed**, approximately 1.4 minutes |
| Actual detector smoke | Candidate/reference models trained and tested on identical independent synthetic images |
| Named protocol execution | Actual candidate/reference/combined and controlled-blur models trained and independently evaluated |
| Shared validation | Actual training used the selected reference validation partition; contamination checks passed |
| Combined readiness | Complete union of **12 training images**, excluding validation/test records; virtual image provenance verified |
| Saved scoring linkage | Model calculation hash matched its exact selected aggregation |
| Test identity | Annotation-aware protocol/evaluation hashes matched; changed-label/same-image regression passed |
| Checkpoint cancellation/resume | Cancelled after epoch 1 and resumed to the original **12 epochs** |
| Reports | Recorded JSON/CSV/HTML/PDF; incomplete and saved-aggregation exports tested |
| PDF layout | Rendered all **9 dataset and 5 research pages**; readable charts/tables with no clipping or overlap observed |
| Browser layout | Desktop, 390-pixel mobile quality views, Lab navigation, actual comparisons, dark mode and exports checked |

The one-epoch scratch runs had zero measured mAP and no successful detections on the tiny synthetic test set. Those actual results were retained. Functional smoke does not establish predictive validity or real UAV generalization.

## Retained evidence

- `.runtime/adaptive-smoke-initial/verification.json`: original actual CPU smoke.
- `.runtime/adaptive-delivery-smoke/verification.json`: named protocol/model/evaluation IDs, complete combined population, exact scoring and annotation-aware test checks.
- `.runtime/adaptive-delivery-smoke/dataset.{json,csv,html,pdf}` and `experiment.{json,csv,html,pdf}`: actual exports.
- `.runtime/adaptive-delivery-smoke/pdf-review-all-pages.png`: all rendered pages.
- `.runtime/resume-verification.json`: actual cancelled/resumed jobs and final model for the 12-epoch test.
- `.runtime/browser-results/`: browser screenshots/results.
- `.runtime/adaptive-build.log`: production-build output.

One named smoke harness exceeded its ten-minute polling deadline while awaiting the final controlled job. Both experiments subsequently completed. Their retained records, annotation-aware test hashes and saved scoring links were independently checked, and exports regenerated. This timeout is recorded in the verification JSON rather than represented as an uninterrupted harness pass. Earlier named and browser runs completed normally.

An initial browser run found an encoding error in the analysis-tab label. It was fixed, and the complete four-test browser suite passed afterward.

## Executed commands

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\adaptive-acceptance-final -o cache_dir=.runtime/pytest-cache
.\.venv\Scripts\python.exe -m pytest -q tests/test_validation_lab.py --basetemp=.runtime\lab-handoff-final -o cache_dir=.runtime/pytest-cache
Push-Location frontend
npm.cmd test -- --reporter=dot
npm.cmd run build
Pop-Location
```

Generate new isolated fixture storage (use a new output directory each time):

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe backend/scripts/smoke_workbench.py --device cpu --output .runtime\my-new-smoke
```

Serve that fixture storage in a separate terminal:

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED = '1'
$env:WORKBENCH_ROOT = 'C:\DataReadiness\.runtime\my-new-smoke\storage'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8049 --no-access-log
```

Run named experiments, resume and browser tests against it:

```powershell
Set-Location C:\DataReadiness
$env:PYTHON_DOTENV_DISABLED = '1'
.\.venv\Scripts\python.exe backend/scripts/smoke_adaptive_lab.py --url http://127.0.0.1:8049 --output .runtime\my-adaptive-smoke
.\.venv\Scripts\python.exe backend/scripts/smoke_resume.py --url http://127.0.0.1:8049
Push-Location frontend
$env:WORKBENCH_URL = 'http://127.0.0.1:8049'
npm.cmd run test:e2e
Pop-Location
```

These harnesses require fixture storage rather than real user datasets. Ordinary startup and the first real experiment are in [the handoff guide](adaptive-readiness-and-validation.md).

## Unestablished research and resource claims

- Weights, scenario multipliers and warning thresholds are engineering policies requiring empirical calibration.
- Synthetic images do not establish accuracy/generalization across real UAV domains.
- CUDA performance on the RTX 4070 Laptop GPU was not established: this environment used CPU PyTorch. GPU selection, conservative defaults and OOM handling are implemented.
- Arbitrary third-party pretrained/imported checkpoint accuracy or compatibility is not established. Trusted compatible local checkpoints are supported; external weights were not downloaded.
- Semantic annotation correctness, exhaustive visible-object labels, metadata authenticity, sensor calibration and unsupported learned/cross-source evidence remain explicit unknowns.

Original APIs, metrics and stored results are retained. This workspace began with project files untracked; no commit or merge was made.
