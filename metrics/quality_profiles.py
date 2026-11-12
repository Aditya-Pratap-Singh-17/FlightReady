"""Versioned, validated engineering policies. These are not calibrated predictors."""
from pathlib import Path
from typing import Literal, Annotated
import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class QualityComponent(StrictModel):
    id: str
    dimension: str
    label: str
    definition: str
    direction: Literal['higher', 'lower', 'criterion']
    source: Literal['metric', 'findings', 'clarity', 'categories', 'class_balance', 'annotation_audit', 'timestamps', 'temporal_order', 'sensor_completeness', 'object_visibility']
    metric_id: str | None = None
    normalization: Literal['fraction', 'one_minus_fraction', 'zero_count', 'requirement_fraction', 'criterion_fraction', 'entropy']
    tasks: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=lambda: ['image', 'tabular'])
    weight: float = Field(default=1, gt=0, le=1000, strict=True)
    requirements: str
    provenance: str
    limitations: str
    gate: Literal['block', 'warning'] | None = None


class TaskPolicy(StrictModel):
    dimension_weights: dict[str, Annotated[float, Field(strict=True)]]
    metric_weights: dict[str, Annotated[float, Field(strict=True)]] = Field(default_factory=dict)
    explanation: str


class ScenarioPolicy(StrictModel):
    tasks: list[str]
    dimension_multipliers: dict[str, Annotated[float, Field(strict=True)]] = Field(default_factory=dict)
    metric_multipliers: dict[str, Annotated[float, Field(strict=True)]] = Field(default_factory=dict)
    explanation: str


class QualityProfile(StrictModel):
    profile_version: str
    normalization_version: str
    aggregation_version: str
    provenance: str
    dimensions: dict[str, str]
    components: list[QualityComponent]
    tasks: dict[str, TaskPolicy]
    scenarios: dict[str, ScenarioPolicy]
    clarity_threshold: float = Field(ge=0)
    warning_below: float = Field(ge=0, le=100)

    @model_validator(mode='after')
    def validate_policy(self):
        ids = [c.id for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError('Quality component IDs must be unique')
        for component in self.components:
            if component.dimension not in self.dimensions:
                raise ValueError('Unknown quality dimension')
        import math
        for policy in [*self.tasks.values(), *self.scenarios.values()]:
            dw = getattr(policy, 'dimension_weights', getattr(policy, 'dimension_multipliers', {}))
            mw = getattr(policy, 'metric_weights', getattr(policy, 'metric_multipliers', {}))
            if set(dw) - self.dimensions.keys() or set(mw) - set(ids):
                raise ValueError('Unknown dimension or component in weighting policy')
            if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in [*dw.values(), *mw.values()]):
                raise ValueError('Policy weights and multipliers must be finite and positive')
        for policy in self.tasks.values():
            if set(policy.dimension_weights) != set(self.dimensions):
                raise ValueError('Every task must declare every dimension weight')
        return self


def load_profile():
    return QualityProfile.model_validate_json((Path(__file__).parent / 'config' / 'adaptive_quality.json').read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
