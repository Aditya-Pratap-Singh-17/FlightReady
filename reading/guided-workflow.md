# Guided dataset readiness workflow

The home page now follows three stages. The original research tools remain under **Workspace tools**.

1. **Dataset & goal.** Upload a dataset, upload a directory, register a local directory, or choose an imported dataset. Describe the desired outcome in the goal assistant. Review the interpreted task, target classes or column, operating conditions and requirements, then confirm the goal. Follow-up messages update the interpretation. The analysis cannot start until the goal is confirmed.
2. **Understand your data.** Start the assessment and see its processing state. The report leads with a plain-language conclusion against the saved goal, quality evidence coverage and task-prioritized next actions. Tabs expose visual evidence and technical measurements. Downloads retain the original full report. Missing evidence remains unknown even when the confirmed requirement checks pass.
3. **Validate with models.** For supported object detection datasets, choose a controlled change, measure the changed training data, set a budget, review the frozen protocol, then start training. The default guided experiment trains an original-data model and a controlled-data model with the same architecture, initialization, seeds and validation partition. Both are evaluated on the same held-out test partition. Compare dimension scores, measured individual criteria, actual model metrics and benchmark visualizations. Advanced tools support other reference datasets, imported models and larger research studies.

## Start locally

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/#Start`. Restart a previously running backend after this update, then refresh the browser. The production frontend has been rebuilt. For frontend development, run `npm.cmd run dev` in `frontend`; Vite proxies API requests to port 8000.

## Gemini interpretation

The existing Gemini REST provider validates its response against `DatasetContext` / `Goal` and rejects invented numerical requirements. Configure `CONTEXT_PROVIDER=gemini` and `GEMINI_API_KEY` in the backend environment yourself, then restart it. `CONTEXT_MODEL` optionally selects a Gemini model. No credential input is added to the frontend.

The home page displays whether the provider is configured. When available, model interpretation is selected by default; the user can choose local interpretation. Only the goal conversation text is supplied to the provider. Dataset files are not submitted. If the provider fails or produces an invalid interpretation, the existing local fallback is clearly identified. Confirmation remains required.

`GET /api/v1/context/provider` returns only a configuration flag and display name. Project records and browser storage do not receive the API key. This does not change the operating system permissions of a local environment file.

The default model is `gemini-2.5-flash-lite`, as requested for the user's free-tier setup. No alternate model is selected automatically. If a configured model returns HTTP 404, check that exact model's availability for the project; an unavailable model does not by itself establish that the API key is invalid or that the model has been retired globally. To select this model without opening the key file, set `$env:CONTEXT_MODEL='gemini-2.5-flash-lite'` before running the backend launch script. The backend must have outbound internet access. When interpretation fails, the goal review displays a fixed, credential-free explanation for model availability, quota, authentication, networking or response-validation failures. Raw provider errors and request headers are never exposed.

## Saved goal and project records

There is one goal contract: the existing `DatasetContext.goal` (`Goal` in `backend/context.py`). Each interpretation and confirmation is saved as a context record in the existing SQLite store. A workflow record links the dataset, conversation, interpreted context ID, confirmed context ID, assessment job ID, split and validation study. The readiness pipeline receives the exact confirmed context. Editing the goal creates another immutable context and clears the current result pointers; older jobs and contexts remain available in the workspace.

The default store is `.runtime/backend/workbench/metadata.sqlite3`, or the configured `WORKBENCH_ROOT`. Browser storage keeps the project selection alongside the existing dataset and theme preferences. Reloading or reopening the project restores its saved goal, analysis and experiment. Unsubmitted chat drafts are not persisted.

## Partitions and comparison

For detection projects, the default analysis uses the training partition. Existing complete partitions are preserved; otherwise the existing split service groups duplicates and recording groups and uses a recorded seed. If recording metadata is absent, explicit acknowledgement is required. The alternative entire-dataset scope produces a readiness report, but cannot be used as training-only evidence for the guided lab.

Controlled variants modify training data only. The guided lab supports blur, reduced resolution, omitted labels and bounding box perturbation. It measures the new training subset before creating a protocol. The two models share original validation images and the held-out original test images. Source checksums, class compatibility, group independence and leakage checks still apply. A repeated action returns its existing active assessment, preparation or protocol rather than creating duplicates. Failed analyses can be retried. Completed or stopped validation studies can be reset to design another comparison; their artifacts are retained.

An arbitrary model can provide another benchmark, but a controlled dataset comparison is needed to investigate the effect of measured quality. One paired experiment does not establish general predictive validity. The guided results scope their research analysis and CSV export to the current experiment; the advanced workspace can analyze the full collection. The advanced analysis retains its independent-dataset and repeated-seed safeguards. Score deltas are withheld when policy fingerprints or effective weights differ. Individual measured criteria remain visible when incomplete dimension coverage prevents a primary score.

## Implementation

- `frontend/src/GuidedWorkflow.tsx`: project restore, upload, goal chat/review, staged assessment, readable report.
- `frontend/src/GuidedValidation.tsx`: sequential comparison design, preparation, protocol review, training monitor and results.
- `frontend/src/workflow.css`: desktop/mobile guided layout, themes and progress states.
- `frontend/src/Workbench.tsx`: three primary destinations, collapsed research tools and existing tool integration.
- `backend/workbench/workflow_api.py`: workflow persistence and orchestration of existing context, split, readiness, variation and experiment services.
- `main.py`: workflow API registration before the static frontend mount.
- `backend/context_extractor.py`: local drone detection/tracking vocabulary for the fallback.

## Verification

All checks used `PYTHON_DOTENV_DISABLED=1` and isolated storage. No local `.env` contents were inspected, and no paid live Gemini call was made. Existing mocked Gemini tests cover its actual request and response validation path.

- Backend: **277 tests passed**, including seven new workflow integration tests.
- Frontend: **29 tests passed**, including new stage gating and reviewed goal persistence checks.
- TypeScript and production Vite build passed.
- Edge browser: the new guided journey and all four retained advanced-workspace tests passed (**5 tests**). The guided journey covers a real synthetic upload, goal confirmation/reload, readiness analysis, controlled blur assessment, two one-epoch CPU training runs, and shared held-out evaluation. Desktop/mobile and dark screenshots were inspected. A first mobile assertion ran before the charts finished resizing; waiting for the responsive layout resolved it.

Reproduce the new browser journey against an isolated server:

```powershell
$env:PYTHON_DOTENV_DISABLED='1'
$env:CONTEXT_PROVIDER='deterministic'
$env:WORKBENCH_ROOT='C:\DataReadiness\.runtime\guided-ui-verification\storage'
.\.venv\Scripts\python.exe -c "from workbench.synthetic import detection_fixture; detection_fixture('.runtime/guided-ui-verification/synthetic-drones.zip',seed=731)"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8052
```

In another PowerShell terminal:

```powershell
Set-Location C:\DataReadiness\frontend
$env:WORKBENCH_URL='http://127.0.0.1:8052'
npm.cmd run test:e2e -- guided-workflow.spec.ts
```

Screenshots are saved in `.runtime/guided-ui-verification/screenshots/`. Synthetic CPU runs verify the software flow; they do not calibrate readiness weights or establish real UAV accuracy or GPU performance.

The final restored-project browser review also verified that the research tab contains only the current study's two model rows, while the synthetic workspace contains six rows, and that its CSV link targets that experiment. The retained evidence is `.runtime/guided-ui-verification/final-review.json`. The temporary verification servers were stopped after review.
