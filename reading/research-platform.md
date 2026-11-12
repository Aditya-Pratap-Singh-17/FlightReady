# FlightReady local research platform

The durable dashboard is the default frontend. It extends the existing React application and FastAPI app; `readiness_engine.assess`, metric calculations, context schemas and report generators remain the source of readiness evidence. The original session workflow is available at `/?legacy=1`, and existing `/api/...` routes remain compatible. New durable routes are under `/api/v1` and are documented by `/docs`.

## Start on Windows PowerShell

Use the existing project virtual environment. Do not recreate it when it already works.

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements-training.txt
# Only create an environment file on a fresh checkout; preserve existing secrets.
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Set-Location frontend
npm.cmd ci
npm.cmd run build
Set-Location ..
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. A production build needs only the Python server. For frontend development, run `npm.cmd run dev` in `frontend` in another terminal and open http://127.0.0.1:5173. The Vite proxy points to backend port 8000. Use one backend process; do not pass `--workers` greater than one. `backend/scripts/run_backend.ps1` now prefers the project virtual environment.

For readiness-only operation, install `requirements/requirements.txt` without the optional training requirements. The UI reports unavailable training dependencies; it does not fabricate compute status.

## Actual workflow

1. **Datasets:** upload a ZIP, use browser directory upload, or register a directory on the backend machine. Inspect gallery annotations, class counts, parser diagnostics and SHA256 version. Registration copies no source images at import. Tables, telemetry and other existing formats remain assessable, with explicit detector ineligibility reasons. Removal from the library is an archive operation that preserves experiment provenance and never deletes a registered external directory.
2. **Model Training → partitions:** create a split manifest. Existing complete train/validation/test assignments are preserved, including directory assignments. Otherwise, seeded allocation operates on entire duplicate/recording groups. Supply a group column or explicitly acknowledge absent recording independence. At least three disjoint groups are needed for automatic splitting.
3. **Dataset Evaluation:** choose that split to measure **only its training partition**. Describe your context, select tags, interpret locally, review/edit the structured JSON, resolve questions and confirm. Run the original readiness pipeline, inspect every returned metric/evidence and requirement finding, and export JSON, CSV or HTML. Entire-dataset reports remain useful but cannot be substituted for training-only evidence.
4. **Model Training:** choose the same dataset, split and completed training-only readiness job. Set architecture, initialization, epochs, batch, resolution, learning rate, seed and device. Candidate and reference runs use the same form; select the broader reference dataset as the working dataset for Model B. An optional complete one-to-one source-to-canonical class mapping can align labels such as `uav` and `drone` before training. Start training; actual epochs, loss curves, worker logs and failures appear from persisted backend data.
5. **Model Comparison:** select both completed models and an external labeled test dataset (entire dataset), or a test-only partition. The service rejects development/test image, likely near-duplicate and comparable known recording-group overlap. Evaluate both models with identical settings. Inspect mAP, operating-threshold precision/recall/F1, FP/FN, per-class AP, inference latency, ground truth and predictions. Audit differing configurations and export the recorded comparison. Different label schemas require explicit mappings; no classes are silently dropped.
6. **Research Analytics:** filter measured quality dimensions, test manifests, datasets and matched model/context groups. Runs join exact readiness reports to downstream evaluation. Seed repeats are aggregated before Pearson/Spearman analysis. There is no invented overall readiness score. Cross-domain cells populate only after real evaluations. The report distinguishes a descriptive validation-minus-test gap from a causal generalization claim.
7. **Controlled experiments:** select the original split and intervention, create a new dataset, create/preserve its split, assess its training subset and train a controlled model. Original datasets are retained. Blur, resolution reduction, size reduction, whole-image class selection, label omission and box-center perturbation are supported. Validation/test image and label bytes are preserved; normalized labels remain valid after resizing. Provenance records mapping, parameters, seed, modifications and hashes.
8. **Reports:** download per-assessment evidence or consolidated JSON/CSV with dataset versions, raw measurements, split manifests, model hashes, configurations, code identity, seeds and evaluation settings.

## Training implementation and initialization

The pinned optional runtime is Ultralytics **8.4.197** with PyTorch **2.15.0** / torchvision **0.29.2**. YOLO11n is the conservative default; YOLO11s is also selectable. Default local settings are 512 px, batch 4, 10 epochs, SGD at 0.001, workers 0, deterministic mode and AMP disabled. These are disclosed engineering defaults, not inferred mission requirements or a claim of sufficient training. Reduce resolution/batch when needed on the RTX 4070 Laptop's 8 GB VRAM. CUDA OOM errors retain logs and recommend smaller settings.

The default scratch initialization needs no download. A fixed initialization seed 0 builds the target class head, saves a local initialization checkpoint and reloads it into the trainer. The training seed controls the run's randomness independently. The adapter verifies the actual target model's tensor SHA256 in `on_train_start` before optimization. Pretrained mode also records the original trusted local `.pt` checksum, builds the target head deterministically, and verifies the actual transferred target weights. Full effective framework arguments are retained in the run's `args.yaml` as well as the app configuration. A comparison audits initialization hashes, architecture, training budgets, settings, seed, class order and dataset sizes.

Ultralytics is offered under **AGPL-3.0 or an Enterprise license**. Review [its licensing terms](https://www.ultralytics.com/license) before distributing or deploying the integrated application/models. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) confirms the supported detection architecture. No pretrained detector's COCO class list is treated as containing a native drone category.

Weight downloads are explicit and optional. For a trusted official checkpoint, run the following yourself (requires network and acknowledges a model download):

```powershell
Set-Location C:\DataReadiness
New-Item -ItemType Directory -Force .runtime\weights | Out-Null
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt').save('.runtime/weights/yolo11n.pt')"
```

Select `C:\DataReadiness\.runtime\weights\yolo11n.pt` in the training form. Only load `.pt` checkpoints you trust; they are framework model files, not untrusted dataset uploads. Normal training/evaluation does not resolve remote dataset/weight URLs. External experiment tracking is disabled.

## CUDA

The Windows NVIDIA driver sees an RTX 4070 Laptop GPU with 8 GB. The installed PyTorch build in this implementation environment is `2.15.0+cpu`, so **CUDA training has not been validated**. The UI correctly selects CPU instead of equating `nvidia-smi` visibility with PyTorch availability. For GPU use, select a matching Windows/Pip/CUDA installation command from the [official PyTorch installer](https://pytorch.org/get-started/locally/); a CPU-only wheel cannot use the GPU. Preserve compatible torch/torchvision versions together, then verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Restart the backend after changing runtime packages. Device status is cached for the lifetime of that backend. Run `backend/scripts/smoke_workbench.py --device 0` after installing a compatible CUDA runtime to validate the GPU path.

## Persistence and execution

`WORKBENCH_ROOT` defaults to `.runtime/workbench`. SQLAlchemy creates SQLite metadata tables automatically on first use; no manual DB initialization is needed. `research_records` contains versioned record envelopes and JSON metadata. Original uploaded sources, parsed dataframes/evidence, optional DuckDB tables, checkpoints and logs live in managed directories, not database blobs. Schema changes currently require a deliberate migration; do not point a future incompatible schema at existing research metadata without a backup.

One isolated subprocess executes readiness, training or evaluation at a time, preventing conflicting GPU jobs. A durable queue survives refreshes. Queued jobs resume on restart; interrupted jobs remain explicit failures with retained evidence and can be retried as new runs. Worker PID/create-time checks clean up a verified orphan worker on restart. Cancellation terminates the active worker, with training workers fixed at zero. A stopped run is never claimed to have completed. Automatic checkpoint resume is not implemented.

Imported image bytes are hash-verified before use; directory edits require re-importing a dataset version. Parsed labels are immutable persisted evidence. Image caches are reused; hard links share managed archive image bytes when possible, falling back to copies if the filesystem disallows linking. Training label files and split lists are explicit. Registration avoids copying at import; actual training needs a verified image cache. Controlled experiments intentionally create separate variants.

Allow space for original sources, parsed tables/evidence, a training image cache, each controlled variant, and several model checkpoints per run. Compression can greatly increase decoded storage. Existing resource/path/file protections apply to archives and directory manifests. `.runtime`, model weights, logs, environment secrets, node modules, build output and browser results are ignored by Git. Back up the complete workbench root with the backend stopped; preserving only SQLite loses filesystem evidence.

## Metrics and research limitations

- mAP is produced by Ultralytics over IoU 0.50 and 0.50:0.95 using confidence floor 0.001. Operating precision/recall/F1 and TP/FP/FN use class-aware confidence-ordered matching at the selected confidence and matching IoU. NMS IoU is a separate setting. Absent class AP stays absent. Negative-only tests have undefined mAP, rather than an invented zero.
- Latency is mean reported **model inference time**, excluding image loading, preprocessing and NMS; sample count/device are retained. It is descriptive, not a controlled hardware benchmark.
- The inherited readiness engine defines requirement-based outcomes, **no composite numerical score**. Numeric measured dimensions can be explored separately. No hard threshold is inferred from a qualitative mission statement.
- Dataset correlations require at least three varying comparable image sources. Small-n results remain explicitly exploratory. Identical source content is grouped, shared-image sources are screened out, artificial variants are excluded from natural-source correlations, and repeated training seeds never increase dataset-level sample size. Shared collection processes and statistical independence cannot be proven by hashes alone.
- Seed means, sample standard deviations and t intervals quantify training-seed variation only. They do not measure independent test-domain uncertainty. Multiple-testing correction, formal causal inference, dataset-level confidence intervals and statistical power claims are not implemented.
- Matched context/model/evaluation groups are deliberately strict. Different task contexts, input sizes or evaluation conditions must not be pooled into a single dataset-readiness correlation. Final test results must not guide tuning; the UI separates test execution, but cannot police a researcher's later decisions.
- dHash near-duplicate matching is a conservative heuristic, not semantic/video independence proof. Complete flight/session metadata is preferable. Grouping by an explicit camera/location/flight/session column is supported. No video/MOT training, segmentation, telemetry detector training, multi-user authentication, distributed queues, PDF export or automatic weight downloads are provided.
- The browser dashboard uses React/TypeScript, TanStack Query, Recharts, accessible native controls, responsive layouts and light/dark themes. Complete context JSON permits all existing scenario and technical requirements. Model confusion is visualized as per-image detection outcomes; a multiclass confusion-matrix plot is not provided.

## Verification commands

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.runtime\pytest-verification -o cache_dir=.runtime/pytest-cache
.\.venv\Scripts\python.exe backend/scripts/smoke_workbench.py --device cpu
Set-Location frontend
npm.cmd test
npm.cmd run build
```

The smoke test performs real candidate and reference model training and inference using explicitly synthetic labeled images. It retains a unique `.runtime/smoke-*` directory containing SQLite, sources, checkpoints, logs, comparison, JSON/CSV exports and `verification.json`. One epoch deliberately checks execution, not detector convergence; measured mAP can be zero and precision unavailable.

For actual browser verification, start a backend at port 8017 (or set `WORKBENCH_URL` to your running backend), then:

```powershell
# Terminal 1, repo root:
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8017
# Terminal 2:
Set-Location C:\DataReadiness\frontend
npm.cmd run test:e2e
```

The Playwright test uses the installed Microsoft Edge browser and real backend requests. It imports the labeled demo ZIP, checks annotation rendering, reviews/confirms context, runs readiness, verifies report data/downloads, switches themes and checks mobile overflow. Screenshots are saved under `.runtime/browser-results`. It adds an explicitly synthetic dataset/report to the selected local workspace; use a separate `WORKBENCH_ROOT` if that is unwanted. The separate Python smoke executes the full detector workflow without a browser; frontend unit tests cover API errors, forms, confirmation and state handling.

A second browser test also starts a real one-epoch detector run from the training controls, evaluates it and a reference model on the same external images, opens the prediction comparison and audits its conditions. To run this test, first complete the Python smoke, start the backend with `WORKBENCH_ROOT` set to that smoke directory's `storage` subdirectory, and set `WORKBENCH_URL` to that backend. Without those prerequisite model artifacts, this additional test is explicitly skipped. The primary browser readiness workflow still runs. Scratch/pretrained differences are reported by the comparison audit rather than silently declared equivalent.

Generalization gaps remain unavailable when a model lacks recorded validation conditions or its validation mAP settings differ from the final evaluation settings.

Troubleshooting: use `npm.cmd` to avoid PowerShell script-policy issues; close a backend before replacing its SQLite/artifact root; use a fresh `--basetemp` path if a Windows temp ACL prevents pytest cleanup; select CPU when CUDA is unavailable; inspect retained worker logs for failed jobs; reduce batch or input size for OOM; keep a complete group/split manifest and resolve overlap errors rather than disabling leakage checks.
