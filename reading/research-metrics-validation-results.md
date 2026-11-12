# Research evidence v2: actual software validation results

Date: 3 November 2026. Final implementation: engine `1.5.0`, enhanced properties `2.0`, evidence `research-evidence-2.0`, interventions `interventions-2.0`. This record describes software behavior on synthetic fixtures. It does not establish real UAV scientific validity or model-performance improvements.

## Executed checks

Commands run from `C:\DataReadiness` unless a frontend directory is stated. Workspace-owned temporary directories avoid the environment's inaccessible default pytest temp/cache locations. The final full test pass explicitly disabled dotenv loading.

```powershell
$env:PYTHON_DOTENV_DISABLED='1'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.work/pytest-research-release-02 -p no:cacheprovider --tb=short
```

**326 passed**, one existing Starlette/httpx deprecation warning. The full existing suite was retained; no existing tests were removed or weakened. The 49 added backend cases include 44 new research-metric cases and five intervention cases. Tests cover exact native/input geometry, scale bins/denominators, invalid boxes, uniform/noisy/blurred/exposed images, rings and neighbors, tiny/unsupported crops, deterministic sampling/cache reuse, annotation categories/matching/ignored scope, negatives, unknown scenarios, pairwise/reference comparisons, bounded duplicate queries, source-scope filtering, unchanged scoring rules, persistence, public API upload/readiness and all export formats. Existing guided/advanced/training/validation contract tests also pass; synthetic/mocked training adapters do not establish real detector accuracy.

From `C:\DataReadiness\frontend`:

```powershell
npm.cmd test -- --reporter=dot
npm.cmd run build
```

**32 frontend tests passed across seven files**. Three added tests verify real zero versus unavailable charts, quantity units/denominator/sample/coverage, switching evidence views and historical certificates. TypeScript checking and the production Vite build passed. Vite emitted the existing dependency-level ignored `use client` directive warnings; no build failure occurred. `npm.cmd` was used because this Windows shell blocks `npm.ps1` execution.

The browser check used an isolated temporary workbench store and deterministic context provider:

```powershell
$env:PYTHON_DOTENV_DISABLED='1'
$env:CONTEXT_PROVIDER='deterministic'
$env:WORKBENCH_ROOT='C:\DataReadiness\.work\research-browser\store'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8029
```

In a separate shell, after generating the explicitly synthetic fixture with `workbench.synthetic.detection_fixture('.work/research-browser/synthetic.zip', seed=412)`:

```powershell
Set-Location C:\DataReadiness\frontend
$env:WORKBENCH_URL='http://127.0.0.1:8029'
npm.cmd run test:e2e -- research-evidence.spec.ts --reporter=line
```

**One browser E2E test passed** using installed Microsoft Edge/Playwright; the final repeat took 11.2 seconds. It exercised actual guided upload, goal interpretation/confirmation, training-partition readiness, the existing six-dimension interface, new evidence plots, unavailable independent audits, object-local proxies and a 390-pixel layout without document horizontal overflow or JavaScript errors. The isolated store had zero trained models. Existing E2E scenarios requiring seeded model artifacts or actual training were not invoked automatically. The temporary test server was stopped afterward; user workbench records were not used.

## Export and visual checks

The public API integration test exercises upload → durable dataset → readiness worker → stored certificate → JSON, CSV, HTML and PDF. It verifies structured `scale_analysis` and `negative_coverage` in the raw exports and new readable sections in HTML/PDF.

The synthetic browser certificate was also exported through the real report renderer. Its PDF had **17 pages**, **eight research evidence sections**, zero out-of-page text bounds and zero trained model artifacts. All 17 pages were rasterized. The contact sheet and full-size scale, local-complexity and unavailable-audit pages were visually reviewed. Desktop/mobile browser screenshots were reviewed. New graphs display backend counts with integer count axes; inaccessible measurements are not drawn as zero distributions. Raw precision remains available in JSON/CSV; current readable PDF methodology retains the application's explicitly stated abbreviation policy.

Temporary evidence (regenerable, ignored from version control) is under `.work/research-browser/`: `research-desktop.png`, `research-mobile.png`, `readiness.html`, `readiness.pdf`, `report-verification.json`, `report-contact-01.png` and selected report-page images. Final backend/frontend/build logs are `.work/research-release-tests.txt`, `.work/research-frontend-tests.txt` and `.work/research-build-final.txt`.

At completion of the metric implementation, the guide still described the 2 November snapshot. The subsequent documentation refresh rewrote the Markdown/HTML/PDF for the current 3 November pipeline; see `CODEBASE_VISUAL_GUIDE_VERIFICATION.md` for its export checks. The application test results in this record were not rerun merely to update the book. The preexisting `metrics/config/adaptive_quality.json` SHA-256 was checked against the pre-task source inventory and is unchanged. No historical records/profiles were rewritten and no new scoring weights were introduced. No commit was attempted: repository metadata is read-only in this environment, and the initial checkout reports the application files as untracked.

## Changed / created files

Existing calculation/pipeline files changed:

- `backend/image_ingestion.py`, `backend/ingestion.py`, `metrics/core.py`, `backend/readiness_engine.py`, `metrics/quality_aggregation.py`, `backend/reports.py`
- `metrics/uav_metrics/assessment.py`, `metrics/uav_metrics/registry.py`
- `backend/workbench/datasets.py`, `backend/workbench/splits.py`, `backend/workbench/combined.py`
- `backend/workbench/contracts.py`, `backend/workbench/workflow_api.py`, `backend/workbench/variations.py`, `backend/research/variants.py`
- `backend/workbench/analytics.py`, `backend/workbench/api.py`, `backend/workbench/detector.py`, `backend/workbench/reporting.py`
- `frontend/src/api.ts`, `frontend/src/QualityAnalytics.tsx`, `frontend/src/GuidedValidation.tsx`, `frontend/src/Workbench.tsx`
- `tests/test_research.py`, `.gitignore`

New implementation/test files:

- `metrics/uav_metrics/research.py`: shared scale/scenario calculations and bounded configuration
- `metrics/uav_metrics/pixels.py`: cached global and bounded local pixel proxies
- `metrics/uav_metrics/audit.py`: independent reference matching and negative evidence
- `metrics/uav_metrics/independence.py`: collection groups and budgeted near-duplicate candidates
- `metrics/uav_metrics/presentation.py`: views derived from actual stored metric evidence
- `frontend/src/ResearchEvidence.tsx`, `frontend/src/researchEvidence.test.tsx`
- `frontend/e2e/research-evidence.spec.ts`
- `tests/test_research_metrics.py`

Documentation changed/created:

- `reading/README.md`
- `reading/research-metrics-development-log.md`
- `reading/research-metrics-v2.md`
- `reading/research-metrics-validation-protocol.md`
- `reading/research-metrics-validation-results.md`

## Implemented, synthetic-tested, scientifically pending

**Implemented:** all six targeted UAV property extensions, object contrast/visibility proxies, separate negative/hard-negative evidence, shared conditional coverage, versioned provenance, bounded resources, existing-pipeline integration, evidence UI/exports, additional controlled train-only interventions and native-size operating-point recall outcomes.

**Synthetic-tested:** exact calculations, missing/unknown distinctions, matching/denominators, sampling budgets, immutable evaluation partitions, persistence, profile coverage/gate behavior, API contracts, frontend states, build, guided readiness and report rendering.

**Still requires real scientific validation:** independent human/external audit quality and representativeness, metadata authenticity, genuine collection independence, proxy behavior under real sensors/compression, semantic target visibility, true hard-negative difficulty, actual repeated-seed training/evaluation, independent deployment reference relevance, cross-dataset predictive validity and causal/performance claims. Small-object AP, calibrated visibility probabilities and statistical effective sample size remain explicitly unsupported; none was fabricated.

No new dependencies are required. Users must supply/confirm actual preprocessing, scale/scenario requirements, meaningful references and independent audits to make the corresponding evidence available. See the [full measurement/configuration specification](research-metrics-v2.md) and [prioritized real-data research protocol and reproducibility checklist](research-metrics-validation-protocol.md).
