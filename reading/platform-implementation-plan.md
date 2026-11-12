# Local research platform implementation plan

## Verified integration contracts

- `main.py`: existing FastAPI app, streamed uploads, context interpretation and asynchronous assessments; keep its API compatible.
- `ingestion.Dataset`: dataframe, profile, normalized `[cx, cy, w, h]` objects, exact image hashes, issues, source archive, optional DuckDB database.
- `large_ingestion.load_dataset_stream`: reuse for imported supported formats; `universal_images` supplies COCO/YOLO adapters. Archive security remains in `image_ingestion.validate_members`.
- `readiness_engine.assess`: unchanged source of readiness decisions and all metric evidence. No defensible composite score exists; preserve categorical results and individual measurements.
- `context_extractor.interpret`, `context.DatasetContext`: retain conservative extraction, manual confirmation and optional remote text interpretation.
- `research.variants`: reuse canonical YOLO train-only transformations. Existing command-line experiments remain compatible.
- React 19 / TypeScript / Vite frontend and existing regression tests remain available. Extend rather than replace assessment components.
- Initial environment: Python 3.13 virtual environment; NVIDIA RTX 4070 Laptop 8 GB visible; torch, ultralytics and SQLAlchemy absent. Verify installed runtime independently of driver availability.

## Ordered changes

1. `backend/workbench/contracts.py`, `store.py`, `datasets.py`, `splits.py`: versioned contracts, SQLite/SQLAlchemy metadata, durable immutable input snapshots and image access, deterministic grouped manifests.
2. `frontend/src/Workbench.tsx`, `workbench.css`, `api.ts`: application shell and dataset import/inspection; keep original `App` assessment export for compatibility.
3. `backend/workbench/api.py`, `worker.py`: adapt existing context/assessment engine, persist complete certificates, expose raw JSON/CSV/HTML.
4. `backend/workbench/jobs.py`, `detector.py`: serialized subprocess queue, durable status/logs/cancellation, local YOLO11n training and independent held-out evaluation.
5. Training, experiment, comparison and analytics views: real polling, metrics/loss charts, ground truth/predictions and reproducibility exports.
6. `backend/workbench/analytics.py`, controlled-variation API: seed aggregation, dataset-level correlations, explicit limits, reuse existing interventions where supported.
7. `tests/test_workbench.py`, frontend tests, `backend/scripts/smoke_workbench.py`: security, scientific invariants, persistence and real CPU end-to-end test. Update README/metrics/config/dependencies.

## Research safeguards

Training readiness reports must match the exact training manifest. Training uses train/validation only; final tests are separate explicit jobs. Reject exact, likely perceptual and known group overlap. Record unresolved session independence. Compare architecture, initialization hash, training budgets, mappings and evaluation settings. Seed repetitions never count as independent datasets. All unavailable results remain absent.
