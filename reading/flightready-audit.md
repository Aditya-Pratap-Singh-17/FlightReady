# FlightReady implementation audit (2026-11-10)

Inspected before modifying the application. This is an existing FastAPI/Pydantic,
React/TypeScript/Vite application, with a SQLite record store and serialized
subprocess compute queue. Preserve those services and saved records.

## Working foundation

- Ingestion: streaming tables, archives and local directories; YOLO, native
  VisDrone DET, COCO, VOC, image folders and metadata parsers. Native VisDrone
  keeps ground-object categories and ignored annotations distinct.
- `metrics/core.py` is the public measurement registry. It dispatches original
  diagnostics and UAV-01 through UAV-20. UAV modules cover pixels, geometry,
  recording independence, scenario evidence, audits and research diagnostics.
- `readiness_engine.py` compiles confirmed requirements and evaluates hard gates
  separately from diagnostic measurements. Unknown scenarios remain unknown.
- `quality_aggregation.py` uses a frozen six-dimension policy, explicit
  applicability, full-coverage score withholding, geometric means and independent
  gates. Unknown evidence retains weight. Task/scenario priorities, recalculation
  history and sensitivity already exist. Illustrative clarity criteria and
  warning thresholds are engineering choices, not validated performance cutoffs.
- `experimental_scoring.py` is a separate, explicitly experimental historical
  score. Do not relabel it as final readiness or change its historical values.
- Guided projects save confirmed goal versions; assessment jobs freeze metrics,
  registry definitions, profile versions and fingerprints. Reports support JSON,
  CSV, HTML and PDF, but HTML and PDF currently follow different presentation paths.
- Actual YOLO11 training/evaluation, checkpoints, CPU/CUDA selection, cancellation,
  recovery, controlled variants, matched-count studies, common held-out protocols,
  per-class errors and exploratory quality/performance analysis already exist.
  Synthetic integration fixtures are software checks, not research findings.

## Correctness and workflow gaps to address

1. Goal review lacks immediate, evidence-backed compatibility feedback. Native
   VisDrone validation-only role is not captured by its wrapper name. Native
   VisDrone/VOC annotations are readable but excluded by detector eligibility.
2. There are two prominent verdicts (confirmed requirements and quality profile).
   A new canonical report must consolidate them without rewriting historical
   results or claiming missing evidence has passed.
3. Final assessment HTML differs from PDF. A single frozen report model should
   feed both; model results must remain a separately linked supplement.
4. Pretrained acquisition has no registry/download path. Existing import requires
   training metadata, preventing genuine inference-only use on a validation set.
5. Original raw measurements, formulas and evidence are available, but the guided
   executive view needs concise blockers, coverage and actionable drilldowns.

Baseline tests are being executed before changes. New tests will cover exact
class support, unknown conditions, single-role partitions, report consistency,
incomplete-evidence bounds and real model execution. No dataset, saved project,
historical scoring policy or existing API response field is to be removed.
