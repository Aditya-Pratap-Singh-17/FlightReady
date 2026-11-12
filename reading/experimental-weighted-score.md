# Temporary weighted dataset score

The experimental score summarizes selected dataset measurements on a 0–100 scale, with unequal importance weights. It is intentionally isolated in `metrics/experimental_scoring.py` and `metrics/config/experimental_score.json` so its weights, normalization and components can be revised or disabled. It does not change the requirement-based READY / NOT_READY / insufficient-evidence decision, metric calculations, or model training gates.

## Initial importance preset

| Component | Weight | Component score |
|---|---:|---|
| Annotation syntax validity | 20 | 100 × (1 − malformed-annotation image fraction) |
| Label / annotation availability | 15 | 100 × (1 − missing-label fraction) |
| Required class presence | 15 | Percentage of confirmed required-class findings satisfied |
| Image clarity | 15 | Percentage of readable images with Laplacian variance ≥100 |
| Declared split overlap checks | 15 | Percentage of configured exact, group and near-duplicate overlap checks measuring zero |
| Image readability | 10 | 100 × (1 − unreadable-image fraction) |
| Exact content uniqueness | 5 | 100 × (1 − duplicate fraction) |
| Scenario coverage | 5 | Percentage of confirmed scenario findings satisfied |

These are **provisional engineering priorities**, not weights learned from experiments or proven to predict accuracy. The clarity threshold of 100 is illustrative and uncalibrated. Syntax and annotation-file availability do not establish correct or exhaustive object labels. Zero detected overlap does not prove statistical independence; duplicate observations can be legitimate. Class presence does not establish balance or adequate sample size.

Not every UAV property has a defensible good/bad direction. Brightness, object density, altitude, scene diversity and other raw quantities are retained in the report, rather than silently treated as percentages. The profile supports adding scalar metrics with explicit normalization anchors once appropriate task criteria are defined.

## Formula and missing evidence

```text
observed score = Σ(weight × component score) / Σ(observed component weights)
evidence coverage = 100 × observed weight / total applicable positive weight
```

Missing, unsupported and unconfigured evidence has no component score. It remains in the applicable weight denominator for coverage; it is not substituted with 0 or 100. Explicitly non-applicable components are excluded from both denominators. Weight 0 excludes a component. If all weights are 0, or all applicable evidence is unavailable, the numeric score is unavailable.

Reports label the result **partial** whenever applicable positive weight lacks evidence. They also show the possible full-score range obtained by allowing unknown components to range from 0 to 100. That range describes missing evidence, not a confidence interval. Component scores, raw measurements, weights, normalized observed weights and contribution points are inspectable.

A hard requirement can fail even with a weighted score of 100. Such failures remain visible and the original readiness decision is retained.

## Use and adjust it

1. Open **Dataset Evaluation** and select a completed report. New reports include the saved experimental score. For a historical report, select **Preview experimental score for this report**.
2. Expand **Edit weights and inspect contributions**, change relative weights and select **Preview weights**. Weights need not sum to 100. A weight of 20 has twice the influence of 10, among observed components.
3. **Reset preview** returns to the saved score. **Download score preview** exports the temporary profile and results. Preview requests are read-only: they do not modify an assessment, dataset, model, training run or research history.
4. To change defaults for future assessments, edit [metrics/config/experimental_score.json](../metrics/config/experimental_score.json). The backend reads it for each new assessment. Existing saved reports keep their original profile. Rerun an assessment to store a new weighting scheme for future research analytics.

Supported metric normalizations are `fraction`, `one_minus_fraction`, `boolean`, and `linear`. Linear scores use `100 × clip((value − bad)/(good − bad), 0, 1)` with distinct finite `bad`/`good` anchors. `output_path` is an explicit list of nested dictionary keys. Estimated model-derived evidence is excluded by this initial prototype.

Example custom component (the anchors below are illustrative, not recommended values):

```json
{
  "id": "custom_measurement",
  "label": "Task-specific measurement",
  "weight": 10,
  "source": "metric",
  "metric_id": "UAV-14",
  "output_path": ["contrast", "mean"],
  "normalization": "linear",
  "bad": 5,
  "good": 40,
  "modalities": ["image"]
}
```

The existing `goal.criterion_weights` and qualitative priorities are not automatically mixed with this prototype. Use this versioned profile or the explicit preview controls so the applied weights are visible and reproducible.

## Disable or replace it

Set `enabled` to `false` in the profile for future assessments, or set `EXPERIMENTAL_DATASET_SCORE_ENABLED=false` in the backend environment and restart the backend. `EXPERIMENTAL_DATASET_SCORE_CONFIG` can point to a different profile file. Invalid configuration produces an unavailable experimental result and leaves the original readiness decision working; it never silently falls back to another weighting scheme.

Saved historical results remain evidence even after disabling new scoring. The profile snapshot and SHA256 are stored with each new score and included in its assessment fingerprint. Reweighting previews preserve the original measurements and report.

## Reports and research

New JSON reports contain `experimental_score`. HTML reports show a separate experimental section. Assessment CSVs retain their raw metric rows and add an explicitly named `EXPERIMENTAL_WEIGHTED_SCORE` row. Consolidated research exports and the experiment table include stored experimental scores and coverage.

Research Analytics offers `experimental_weighted_score` for newly saved scored assessments. Correlation groups require the same profile hash **and the same observed positive-weight component set**. This prevents partial scores with different available evidence from being pooled. Existing rules still aggregate seeds, screen shared sources, and exclude artificial variants from natural-dataset correlations. Temporary previews are not added to research analytics.

## Preview on the preceding mock experiment

The stored training-only reports were scored without retraining models or changing their historical records:

| Target training dataset | Experimental observed score | Evidence coverage | Missing-evidence full-score range | Previous target mAP@50:95 |
|---|---:|---:|---:|---:|
| Low clarity | 81.25 | 80% | 65–85 | 65.64% |
| Mixed clarity | 90.625 | 80% | 72.5–92.5 | 86.41% |
| High clarity | 100.00 | 80% | 80–100 | 85.19% |

Scenario criteria were not configured and split overlap could not be established from the training-only subset, leaving 20 weight units unknown. Other measured components happened to meet their targets. A high observed score therefore does not establish complete evidence or predicted accuracy. The mixed dataset still had slightly better target mAP than the highest-scoring dataset; the preset has not been validated as a performance predictor.

The read-only preview command is:

```powershell
.\.venv\Scripts\python.exe backend/scripts/preview_weighted_scores.py --store .runtime/quality-performance-20261101/storage
```

Actual preview evidence: [.runtime/weighted-score-preview-20261101/scores.csv](../.runtime/weighted-score-preview-20261101/scores.csv). The source experiment, counts, training losses and performance limitations are documented in [quality-performance-experiment.md](quality-performance-experiment.md).
