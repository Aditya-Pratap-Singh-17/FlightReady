# Project folders and execution order

The application is divided into three main code folders, with database code and reading material alongside them.

```text
C:/DataReadiness/
|-- main.py                   FastAPI entry point; run from the project root
|-- frontend/                 React UI, components, styles and browser tests
|-- backend/                  API routers, ingestion and orchestration
|   |-- context.py            Domain models for goals, evidence and certificates
|   |-- context_extractor.py  Local and optional remote goal interpretation
|   |-- readiness_engine.py   Requirement compilation and readiness decisions
|   |-- workbench/            Persistent APIs, datasets, jobs and model validation
|   |-- research/             Experiment and dataset-variant implementation
|   `-- scripts/              Launch scripts and optional command-line tools
|-- metrics/                  All dataset measurements and scoring policies
|   |-- core.py               Shared registry and general metric calculations
|   |-- disk_metrics.py       Full-dataset measurements through DuckDB
|   |-- dataProfiler.py       Dataset profiling
|   |-- featureAnalyzer.py    Feature measurements
|   |-- labelAnalyzer.py      Label measurements
|   |-- IsoIecChecker.py      Standards-related checks
|   |-- quality_profiles.py   Validated weighting policies
|   |-- quality_aggregation.py Six-dimension aggregation
|   |-- experimental_scoring.py Optional experimental summary
|   |-- config/               Metric and scoring policy JSON
|   `-- uav_metrics/          UAV-01 through UAV-20 and shared evidence helpers
|-- sql/                      Database infrastructure
|   |-- models.py             Actual SQLAlchemy table definitions
|   |-- store.py              SQLite persistence and managed artifact paths
|   `-- datasets.py           Bounded DuckDB connection configuration
|-- reading/                  Guides, reports, examples, diagrams and doc tools
|-- tests/                    Python regression and integration tests
|-- requirements/             Base, training, and development Python packages
```

## Metrics in pipeline order

The source filenames are valid Python modules. Their reading order follows the assessment pipeline; UAV identifiers retain their existing numeric order and meaning.

1. **Profile:** `metrics/dataProfiler.py`, `featureAnalyzer.py`, `labelAnalyzer.py`, `IsoIecChecker.py` describe available data and feature/label evidence. Ingestion prepares these inputs in `backend/`.
2. **Select and execute:** `metrics/core.py` owns the shared metric registry, general calculations and dispatch. `backend/readiness_engine.py` compiles the confirmed goal into requirements and requests the appropriate measurements.
3. **Measure full tables:** `metrics/disk_metrics.py` calculates against the complete disk-backed dataset. Database connection policy lives in `sql/datasets.py`.
4. **Measure UAV properties:** `metrics/uav_metrics/registry.py` lists **UAV-01 to UAV-20 in order**. `assessment.py` dispatches them; `research.py`, `pixels.py`, `audit.py`, `independence.py`, `vision.py` and `learning.py` provide shared evidence. `presentation.py` creates evidence views from those results.
5. **Apply quality policy:** `metrics/quality_profiles.py` loads `metrics/config/adaptive_quality.json`; `quality_aggregation.py` computes the existing six dimensions, coverage and gates.
6. **Calculate the optional summary:** `metrics/experimental_scoring.py` uses `metrics/config/experimental_score.json`. It remains separate from the requirement-based readiness decision.
7. **Persist and present:** `backend/` saves and reports the result using `sql/`; `frontend/` presents it to the user.

## Start and check the app

Run from `C:/DataReadiness`:

```powershell
.\.venv\Scripts\python.exe main.py
# Equivalent direct command:
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The existing `backend/scripts/run_backend.ps1` helper also starts `main:app` from the repository root.

The frontend commands remain in `frontend/`: `npm.cmd run dev`, `npm.cmd run build`, `npm.cmd test`. Python tests run from the repository root with `.\.venv\Scripts\python.exe -m pytest`.

Standalone experiment commands now start with `python -m backend.research`. Optional Python utilities are under `backend/scripts/` and can be invoked as modules, for example `python -m backend.scripts.preview_weighted_scores --help`.

## Reading material and local data

The former `docs/` contents now live directly in `reading/`. The former root README is [PROJECT_README.md](PROJECT_README.md). [README.md](README.md) is the documentation index. Documentation sources and exports are together, with their tooling under `reading/tools/`, diagrams under `reading/assets/`, and report examples under `reading/examples/`.

`demo_data/` was removed. Optional sample generation remains available through `backend/scripts/generate_demos.py`; it writes to `.runtime/demo_data/` only when explicitly invoked. The example-report command and browser test generate samples in temporary directories.

Repository/editor settings, the private environment file, existing runtime data and local dependency environments remain at the project root. Python dependency declarations are together in `requirements/`. The database layout and stored records are unchanged. Domain/request models in `backend/context.py` and `backend/workbench/contracts.py` remain with the backend; SQL table models live in `sql/models.py`.
