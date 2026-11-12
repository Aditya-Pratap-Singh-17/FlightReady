# Adaptive readiness and validation implementation audit

## Existing functionality (baseline: 242 Python tests, 22 frontend tests pass)

- FastAPI legacy assessment API and persistent `/api/v1` workbench; React 19, TypeScript, Recharts, React Query.
- Bounded CSV/Parquet/record ingestion, image archives/directories, YOLO/COCO/VOC/VisDrone parsers. Detection training supports YOLO/COCO adapters.
- `metrics.REGISTRY`: 38 original diagnostics/requirements plus UAV-01 through UAV-20. Definitions, measurement scope, raw values, evidence, limitations, and measurement versions are retained in certificates.
- Natural-language/tag interpretation, local fallback, optional remote context provider, mandatory human review.
- SQLite `research_records` stores immutable dataset versions, contexts, splits, jobs, models, evaluations, comparisons and analyses. Managed files contain dataset snapshots, manifests, logs, checkpoints and predictions.
- Isolated serialized training/evaluation subprocesses, cancellation, matched initialization, class mapping, train-only readiness checks, exact/perceptual/group leakage checks, inference gallery, seed aggregation, train-only perturbations.
- JSON/CSV/HTML readiness reports and research JSON/CSV.

## Integration/refactoring needed

- Replace primary experimental scalar UI with six evidence-aware dimensions; preserve the historical experimental feature and all raw metrics.
- Add versioned task/scenario policies, normalized effective weights, explicit confirmed overrides, frozen provenance, geometric aggregation, and independent critical gates.
- Add task previews, missing-measurement detection, immutable saved aggregation versions, charts/drill-down and sensitivity.
- Place existing training/comparison/analytics tools inside a distinct Model Validation Lab, preserving old routes.
- Orchestrate named multi-seed candidate/reference/combined experiments, retain matched-size selections, test conditions and child job provenance.
- Extend actual evaluation evidence with PR/per-class/error-count charts, environment and latency denominators.
- Add resume, checkpoint import, external published benchmark metadata, and readable PDF/research reports.
- Expose dimension predictors and baseline comparisons without counting seeds or related variants as independent datasets.

## New records and endpoints

- `aggregation`, `experiment`, `external_benchmark`; no replacement database or migration of existing records.
- `/quality/*`: registry, profiles, previews/reaggregation, saved records, task comparisons and sensitivity.
- `/experiments/*`: named protocols and background execution; `/models/import`, `/benchmarks/external`, `/jobs/{id}/resume`.
- Existing reporting APIs gain actual dimension rows and PDF/research HTML support.

## Required verification

- Normalization, zero preservation, weight sums, task applicability, missing evidence, gates, masking, deterministic frozen-profile reproduction and validation errors.
- Cache reuse versus changed-task evidence requirements; immutable saved assessments.
- Fair comparison conditions, experiment sizes/seeds, dependency grouping, external benchmark separation, incomplete reports.
- Frontend context/weight interaction, Lab navigation and backend-sourced charts; build and real isolated CPU smoke.

Secrets are excluded from source inspection. Verification processes set `PYTHON_DOTENV_DISABLED=1` and use isolated runtime storage.

## Implemented integration

- [x] Preserve existing metrics, parsers, requirement decisions and original APIs.
- [x] Six-dimensional versioned scoring, task/scenario priorities, exact effective weights and provenance.
- [x] Strict missing-evidence coverage, genuine zeroes, independent gates and component bottleneck warnings.
- [x] Quality cards, score/weight charts, actual histograms, drill-down, saved overrides, task comparison and sensitivity.
- [x] Separate Model Validation Lab with all six sections and the existing compute services.
- [x] Named candidate/reference/combined/controlled protocols, multiple seeds, natural/matched sizes, separate shared validation and unchanged independent tests.
- [x] Complete combined-training readiness from the actual union of cached source evidence.
- [x] Real status/loss/error/log/artifact monitoring, parent/child cancellation, restart recovery and actual epoch-checkpoint resume.
- [x] Compatible local model comparison, per-class/error/PR/confusion evidence, matched-image predictions and separate published benchmark records.
- [x] Source-aware seed grouping, exploratory uncertainty, held-out dataset predictors, paired interventions and clear evidence limits.
- [x] Separate readiness/research JSON, CSV, HTML and PDF reports with real charts/tables and incomplete states.
- [x] Unit/API/browser/build and actual CPU experiment verification; see the linked verification record for final results.

Empirical calibration on independent real UAV datasets and GPU/pretrained validation remain backend/research/resource-dependent work, not manufactured outcomes.
