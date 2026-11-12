# FlightReady documentation

For the repaired **Hugging Face/local model importer**, start with
[Model Lab import repair](model-import-fix.md). It includes the actual pretrained
YOLOS run on the saved 548-image VisDrone dataset and current usage instructions.

For the **10 November 2026 implementation**, start with the [connected research
platform runbook](flightready-research-platform.md), [codebase audit](flightready-audit.md)
and [verification record](flightready-verification.md). These cover engine 1.6.0,
the canonical final report, immediate compatibility checks and safe inference.
The textbook below remains a historical 3 November snapshot.

The current server entry point is `C:/DataReadiness/main.py`. From the project root, run `.\.venv\Scripts\python.exe main.py` and open `http://127.0.0.1:8000`. The exported textbook records its earlier source snapshot; use these current startup instructions after the entry-point move.

For the current folder layout and metric reading order, start with [Project structure](PROJECT_STRUCTURE.md). Application code is grouped under `backend/`, `frontend/` and `metrics/`, database infrastructure under `sql/`, and all reading material under `reading/`.

The textbook editions below now describe the **3 November 2026 pipeline**, including engine 1.5.0 and research-evidence-2.0. Their chapters, diagrams, function cards and source references have been refreshed against current code. The [metric specification](research-metrics-v2.md), [research protocol/checklist](research-metrics-validation-protocol.md) and [recorded application validation results](research-metrics-validation-results.md) remain focused companions. The core measurement definitions and full research protocol are also included inside the textbook.

Start with the complete source-derived textbook:

- [Visual codebase guide - Markdown](CODEBASE_VISUAL_GUIDE.md)
- [Interactive offline textbook - HTML](CODEBASE_VISUAL_GUIDE.html)
- [Complete printable textbook - PDF](CODEBASE_VISUAL_GUIDE.pdf)

All three editions contain the same explanations, three worked examples, complete symbol reference and 25 diagrams. The HTML embeds SVGs, downloadable diagram assets, styles and interaction code. It requires no server, internet connection or external font. Double-click it to open it in Chrome. Use the sidebar, **Ctrl K** search, **Open diagram / zoom**, and expandable explanations. The PDF expands explanations, renders diagrams as high-resolution images and includes clickable contents, bookmarks, page numbering and source links. Larger diagrams use dedicated portrait/landscape spreads rather than reducing their labels below the documented legibility threshold.

Source references point to real repository files. The HTML/PDF remain independently readable if copied elsewhere; source links naturally require the corresponding repository files to remain available at their referenced locations. The PDF has embedded fonts. Browser print is available, but the supplied PDF is a separately formatted ReportLab edition of the same Markdown, not a shortened browser summary.

## What is covered

The guide covers startup, the guided/advanced/legacy frontend, all repository-declared API routes, bounded ingestion, image/annotation formats, context providers and goal confirmation, metric dispatch, UAV properties, both scoring systems, gates and missing evidence, splits/leakage, real training/resume, experiments, independent evaluation, model comparison, statistical research analysis, persistence, reports, errors, tests, extension points and architectural limitations.

The inventories include 83 Python files and 23 TypeScript files, with 851 Python symbols and 152 TypeScript declarations, including tests. There are 61 detailed boundary/function cards and a compact reference for the remaining declarations. The snapshot records 74 route decorators; framework-supplied API documentation routes are not counted as repository-declared handlers. Counts are from static source analysis, not a claim that every possible runtime path executed.

## Refresh the book after code changes

Run from the repository root in PowerShell:

```powershell
Set-Location C:\DataReadiness
.\.venv\Scripts\python.exe reading\tools\inventory.py
node reading\tools\typescript_inventory.mjs
.\.venv\Scripts\python.exe reading\tools\build_guide.py
node reading\tools\render_diagrams.mjs
.\.venv\Scripts\python.exe reading\tools\export_guide.py
.\.venv\Scripts\python.exe reading\tools\verify_guide.py
node reading\tools\verify_html.mjs
```

`build_guide.py` combines curated explanations in `manuscript.py`/`deep_reference.py`/`diagrams.py` with current-pipeline revisions in `current_pipeline.py`/`current_diagrams.py` and the maintained research specification/protocol with current source inventories. Review those explanations and diagrams when behavior changes; an automatically refreshed signature table cannot validate an outdated architectural claim. Both exporters then read the generated Markdown and rendered diagrams. They do not independently regenerate the explanations. For an editorial-only change made directly to Markdown, skip the build step and re-export; if a Mermaid block changes, synchronize its curated definition/diagram manifest before rendering.

The scripts import no application entry point, access no `.env` values, and run no training jobs. They only read source/configuration/test files and write documentation/export artifacts. `inventory.py` records source hashes; verification checks that the application files have not changed during documentation work.

## Local documentation tooling

Mermaid 12.2.0 is vendored as `tools/mermaid.min.js`, with its license, so diagram rendering is offline after setup. Markdown-it-py and mdurl are vendored under `tools/vendor`; their licenses are included. The TypeScript parser and Playwright are reused from the existing frontend installation. Diagram generation uses installed Microsoft Edge; browser verification uses installed Google Chrome. Neither is a global Mermaid installation.

The exporter uses the existing project environment's ReportLab, Pygments, Pillow, PyMuPDF and Windows Segoe UI/Georgia/Consolas fonts. The generated HTML/PDF are portable; regeneration currently assumes these Windows fonts or an explicit adjustment to the font paths in `export_guide.py`. If that environment is recreated, install documentation-only prerequisites there rather than changing application logic.

## Verification evidence

- [Export/source/PDF verification JSON](assets/architecture/verification.json)
- [Offline Chrome interaction and content checks](assets/architecture/html-verification.json)
- [Source and export counts](assets/architecture/guide-manifest.json)
- [Actual Mermaid render manifest](assets/architecture/render-manifest.json)
- [Human-readable verification record](CODEBASE_VISUAL_GUIDE_VERIFICATION.md)
- [Diagram SVG/PNG assets and Mermaid sources](assets/architecture/)
- [Rendered page contact sheets and browser screenshots](assets/architecture/review/)

The PDF checker accounts for every source text block, excluding running page furniture/repeated table heads when a paragraph crosses a page. It checks source links, source hashes, internal PDF destinations, diagram count/resolution and bounds, then rasterizes every page. The browser checker opens the HTML through `file://`, blocks network access, verifies all text blocks and internal targets, and exercises search, zoom, expandable explanations and mobile navigation.

This documentation task did not rerun expensive model training, query live paid context providers or establish real UAV/GPU performance. The guide clearly separates static evidence, prior synthetic application checks, optional integrations and unverified scientific claims. Application source code, datasets, checkpoints and schemas were not modified.

## Existing focused guides

- [Guided application workflow](guided-workflow.md)
- [Adaptive readiness and Model Validation Lab](adaptive-readiness-and-validation.md)
- [Research platform setup](research-platform.md)
- [Supported formats](formats.md)
- [Context schema](context-schema.md)
- [Metric descriptions](metrics.md)
- [UAV evidence framework](uav-framework.md)
- [Vision/evaluation protocol](vision-evaluation-protocol.md)
- [Older experimental scalar](experimental-weighted-score.md)

Older planning documents are historical context. The visual guide identifies implemented paths from the inspected source rather than assuming every plan is active.
