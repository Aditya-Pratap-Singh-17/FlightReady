# Context schema

`backend/context.py` defines strict extra-field-rejecting Pydantic models. `/docs` and `/openapi.json` expose the full schema. The original `Goal` fields are retained (including the legacy `trash_inputs` name for excluded features).

`DatasetContext` holds `original_text`, categorized `selected_tags`, validated `goal`, per-field `field_sources`, clarification questions, provider and confirmation flag. The server stores the initial interpretation separately and snapshots the confirmed context on submission.

## Goal example

Apply this JSON in the advanced editor after interpreting a goal:

```json
{
  "task_type": "classification",
  "task_description": "Classify observations in daytime urban flights",
  "prediction_target": "label",
  "prediction_unit": "class",
  "input_modalities": ["tabular"],
  "required_inputs": ["x"],
  "optional_inputs": ["imu"],
  "required_classes": ["car", "pedestrian"],
  "max_missing_fraction": {"x": 0.05, "label": 0},
  "valid_ranges": {"altitude_meters": [20, 80]},
  "expected_units": {"altitude_meters": "m"},
  "split_column": "split",
  "group_column": "flight_id",
  "scenarios": [{
    "name": "Daytime urban pedestrian examples",
    "conditions": {"lighting": ["daytime"], "geography_type": ["urban"]},
    "required_class": "pedestrian",
    "minimum_samples": 2,
    "severity": "hard"
  }],
  "requirements": [{
    "requirement_id": "REQ-SPLIT",
    "description": "No flight appears in more than one split",
    "metric_id": "group_leakage",
    "operator": "equals",
    "expected_value": 0,
    "severity": "hard",
    "source": "manual"
  }]
}
```

The numbers in this example are **illustrative user choices**, not scientifically recommended thresholds. Expected units require dataset unit metadata. A missing unit declaration is insufficient evidence.

Scenario `conditions` are ANDed across fields. A list denotes allowed alternatives within one field; a `{min, max}` object denotes an inclusive numeric interval. To require separate presence of multiple conditions (both day and night), create separate scenarios. `minimum_samples=null` establishes presence only (one observation); it makes no sufficiency claim. Required class/scenario combinations are compiled separately.

`valid_ranges` means every observed value must respect a declared physical/semantic range. `target_environment.altitude_meters` means the dataset must contain observations in that operating envelope; it does not require every observation to be inside it. These are different constraints. No statistical outlier rejection is performed by default.

## Requirement fields

- Unique `requirement_id`, plain-language `description`, registered `metric_id`.
- `operator`: `equals`, `lte`, `gte`, `contains`, `between`.
- `expected_value`: actual typed comparison value; false and zero remain distinct.
- `severity`: `hard`, `preferred`, `diagnostic`.
- `source`: `user_text`, `tag`, `manual`, `inferred`.
- `parameters`: calculation-specific arguments, for example `column`, `class`, `conditions`, `range`, `allowed`, `left`, `right`, `relation`.
- Initial `status` is `not_evaluated`; final findings store state, reason, calculation result and evidence separately.

Example custom cross-field requirement: metric `cross_field_validity`, parameters `{"left":"start_time_numeric","right":"end_time_numeric","relation":"lte"}`, expected value `1`, operator `equals`. This checker uses numeric pairs, not arbitrary Python expressions. It cannot execute user code.

`criterion_weights` is preserved for compatibility but has no effect: V1 makes requirement decisions without weighted scoring. `target_population`, `prediction_horizon`, `relevant_time_period` and free-text `geography` are recorded with limitations; measurable scenario fields and reference data are needed to test them. `reference_columns` maps target columns to declared trusted reference columns, subject to metadata/evidence checks.
