"""Validated contracts shared by the API and assessment engine.

Goal extends the original model; Pydantic also validates untrusted API JSON.
"""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

State = Literal['pass', 'fail', 'warning', 'insufficient_evidence', 'not_applicable', 'error']
Dimension = Literal['integrity', 'validity', 'duplication', 'label_quality', 'coverage', 'representativeness', 'leakage', 'evidence_reliability']
TaskType = Literal['classification', 'regression', 'object_detection', 'tracking', 'anomaly_detection', 'telemetry', 'unspecified']
MeasurementStatus = Literal['measured', 'estimated', 'insufficient_evidence', 'not_applicable', 'unsupported', 'error']
RequirementOperator = Literal['equals', 'lte', 'gte', 'contains', 'between']
RequirementSeverity = Literal['hard', 'preferred', 'diagnostic']

class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Requirement(Model):
    requirement_id: str
    description: str
    metric_id: str
    operator: RequirementOperator = 'equals'
    expected_value: Any = True
    severity: RequirementSeverity = 'hard'
    source: Literal['user_text', 'tag', 'manual', 'inferred'] = 'manual'
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: Literal['not_evaluated'] = 'not_evaluated'

class ScenarioRequirement(Model):
    name: str
    conditions: dict[str, Any]
    required_class: str | None = None
    minimum_samples: int | None = Field(default=None, ge=1)
    severity: Literal['hard', 'preferred'] = 'hard'

class QualityPolicySettings(Model):
    scenario: str = 'default'
    dimension_weights: dict[str, object] = Field(default_factory=dict, max_length=32)
    metric_weights: dict[str, object] = Field(default_factory=dict, max_length=64)
    overrides_confirmed: bool = False

class Goal(Model):
    task_type: TaskType = 'unspecified'
    task_description: str = Field(default='Unspecified goal', min_length=1)
    prediction_target: str = ''
    prediction_unit: str = ''
    input_modalities: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    trash_inputs: list[str] = Field(default_factory=list)
    target_population: str | None = None
    prediction_horizon: str | None = None
    relevant_time_period: str | None = None
    geography: str | None = None
    expected_types: dict[str, str] = Field(default_factory=dict)
    valid_ranges: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    allowed_values: dict[str, list[Any]] = Field(default_factory=dict)
    max_missing_fraction: dict[str, float] = Field(default_factory=dict)
    expected_units: dict[str, str] = Field(default_factory=dict)
    date_column: str | None = None
    max_data_age_days: int | None = Field(default=None, ge=0)
    reference_columns: dict[str, str] = Field(default_factory=dict)
    require_source: bool = False
    require_collection_date: bool = False
    require_dataset_version: bool = False
    criterion_weights: dict[str, float] = Field(default_factory=dict)
    required_classes: list[str] = Field(default_factory=list)
    target_environment: dict[str, Any] = Field(default_factory=dict)
    priorities: dict[str, str] = Field(default_factory=dict)
    scenarios: list[ScenarioRequirement] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    split_column: str | None = None
    group_column: str | None = None
    identifier_column: str | None = None
    # Named metadata mappings and criteria are confirmed with the rest of Goal.
    uav_parameters: dict[str, Any] = Field(default_factory=dict)
    performance_objectives: dict[str, Any] = Field(default_factory=dict)
    quality_policy: QualityPolicySettings = Field(default_factory=QualityPolicySettings)

    @model_validator(mode='after')
    def check_consistency(self):
        required, optional, excluded = map(set, (self.required_inputs, self.optional_inputs, self.trash_inputs))
        if required & optional or required & excluded or optional & excluded:
            raise ValueError('Inputs cannot be required, optional and/or excluded simultaneously')
        if any(not 0 <= n <= 1 for n in self.max_missing_fraction.values()):
            raise ValueError('Missingness fractions must be between 0 and 1')
        for low, high in self.valid_ranges.values():
            if low is not None and high is not None and low > high:
                raise ValueError('Range minimum exceeds maximum')
        ids = [r.requirement_id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError('Requirement identifiers must be unique')
        return self

class DatasetContext(Model):
    original_text: str
    selected_tags: dict[str, list[str]] = Field(default_factory=dict)
    goal: Goal
    field_sources: dict[str, Literal['explicit', 'inferred', 'unspecified']] = Field(default_factory=dict)
    clarification_questions: list[str] = Field(default_factory=list)
    provider: str = 'deterministic'
    provider_error: str | None = None
    provider_error_code: str | None = None
    confirmed: bool = False

class MetricDefinition(Model):
    metric_id: str
    dimension: Dimension
    description: str
    supported_modalities: list[str] = Field(default_factory=lambda: ['tabular', 'image'])
    required_dataset_fields: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list)
    output_type: str = 'number'
    unit: str | None = None
    applicability: str = 'See metric documentation'
    evidence_requirements: str = 'Observed dataset records'
    decision_relevant: bool = False
    calculation_method: str
    name: str | None = None
    assessment_category: str | None = None
    applicable_tasks: list[str] = Field(default_factory=list)
    measurement_type: str = 'observed'
    limitations: list[str] = Field(default_factory=list)
    implementation_status: str = 'implemented'
    implementation_version: str = '1'

class AssessmentEvidence(Model):
    method: str
    sample_size: int | None = None
    references: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None

class MetricResult(Model):
    metric_id: str
    dimension: Dimension
    state: State
    value: Any = None
    reason: str
    evidence: AssessmentEvidence
    assessment_status: MeasurementStatus | None = None
    unit: str | None = None
    dataset_scope: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    estimation_method: str | None = None
    limitations: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)

class AssessmentFinding(Model):
    requirement: Requirement
    result: MetricResult
    state: State
    reason: str

class RemediationRecommendation(Model):
    description: str
    finding_ids: list[str] = Field(default_factory=list)
    conditional: bool = False
    supporting_metric_ids: list[str] = Field(default_factory=list)
    observed_evidence: str | None = None
    hypothesized_effect: str | None = None

class Certificate(Model):
    schema_version: str = '1.0'
    dataset: dict[str, Any]
    initial_context: DatasetContext | None = None
    confirmed_context: DatasetContext
    overall_status: Literal['READY', 'CONDITIONALLY_READY', 'NOT_READY', 'INSUFFICIENT_EVIDENCE', 'ASSESSMENT_ERROR']
    executive_summary: str
    findings: list[AssessmentFinding]
    metrics: list[MetricResult]
    additional_benefits: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendations: list[RemediationRecommendation] = Field(default_factory=list)
    reproducibility: dict[str, Any]
    learning_views: dict[str, Any] = Field(default_factory=dict)
    experimental_score: dict[str, Any] | None = None
    quality_measurements: dict[str, Any] = Field(default_factory=dict)
    quality_profile: dict[str, Any] | None = None
    task_compatibility: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None
    disclaimer: str = 'Research dataset-readiness assessment only; not official aviation safety certification or a guarantee of model performance.'
