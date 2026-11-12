"""Removable, provisional weighted summary. Never used by readiness or training gates.

Unknown evidence is excluded from the observed mean and shown in coverage/bounds.
Defaults live in config/experimental_score.json, not in metric implementations.
"""
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parent/'config'/'experimental_score.json'
NOTICE = 'Experimental weighted quality summary; provisional weights and criteria, not predicted model accuracy or a readiness decision.'


class Component(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z][a-zA-Z0-9_]*$')
    label: str = Field(min_length=1, max_length=160)
    weight: float = Field(ge=0, le=1000000, strict=True)
    source: Literal['metric','required_class_coverage','scenario_coverage','image_clarity','split_overlap_checks']
    metric_id: str | None = None
    metric_ids: list[str] = Field(default_factory=list, max_length=20)
    output_path: list[str] = Field(default_factory=list, max_length=12)
    normalization: Literal['fraction','one_minus_fraction','linear','boolean'] = 'fraction'
    bad: float = 0
    good: float = 1
    threshold: float = Field(default=100, ge=0)
    modalities: list[Literal['image','tabular']] = Field(default_factory=lambda:['image','tabular'])

    @model_validator(mode='after')
    def coherent(self):
        if self.source == 'metric' and not self.metric_id:
            raise ValueError('A metric component requires metric_id')
        if self.normalization == 'linear' and self.good == self.bad:
            raise ValueError('Linear normalization requires distinct good and bad anchors')
        if self.source == 'split_overlap_checks' and not self.metric_ids:
            raise ValueError('Overlap checks require metric_ids')
        return self


class Profile(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    version: str = Field(min_length=1, max_length=80)
    enabled: bool = True
    components: list[Component] = Field(min_length=1, max_length=64)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({c.id for c in self.components}) != len(self.components):
            raise ValueError('Component IDs must be unique')
        return self


def enabled():
    return os.getenv('EXPERIMENTAL_DATASET_SCORE_ENABLED','true').strip().lower() not in ('false','0','no','off')


def load_profile():
    path = Path(os.getenv('EXPERIMENTAL_DATASET_SCORE_CONFIG') or DEFAULT_CONFIG)
    if path.stat().st_size > 1024*1024:
        raise ValueError('Experimental score config exceeds 1 MiB')
    return Profile.model_validate_json(path.read_text(encoding='utf-8'))


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _metric_value(metric):
    if metric is None:
        raise ValueError('Metric evidence is unavailable')
    if metric.state in ('error','insufficient_evidence','not_applicable') or metric.assessment_status in (
            'estimated','insufficient_evidence','unsupported','error','not_applicable') or metric.value is None:
        raise ValueError(metric.reason)
    return metric.value


def _normalize(value, component):
    if component.normalization == 'boolean':
        if not isinstance(value,bool): raise ValueError('Expected an explicit boolean measurement')
        return float(value)
    if not _finite(value): raise ValueError('Expected a finite numeric measurement, not a boolean')
    if component.normalization in ('fraction','one_minus_fraction'):
        if not 0 <= value <= 1: raise ValueError('Fraction measurement is outside [0,1]')
        return float(value) if component.normalization == 'fraction' else 1-float(value)
    return min(1.,max(0.,(float(value)-component.bad)/(component.good-component.bad)))


def _finding_fraction(findings, metric_id):
    selected = [f for f in findings if f.requirement.metric_id == metric_id]
    if not selected: raise ValueError('No confirmed requirements configure this component')
    if any(f.state not in ('pass','fail','warning') for f in selected):
        raise ValueError('At least one required measurement lacks conclusive evidence')
    return sum(f.state == 'pass' for f in selected)/len(selected), {'met':sum(f.state == 'pass' for f in selected),'required':len(selected)}


def _measure(component, ds, goal, metrics, findings):
    if component.source == 'metric':
        metric = metrics.get(component.metric_id)
        # Explicit non-applicability removes weight; missing evidence keeps weight unknown.
        if metric is not None and metric.state == 'not_applicable':
            return None, None, 'not_applicable', metric.reason
        value = _metric_value(metric)
        for key in component.output_path:
            if not isinstance(value,dict) or key not in value:
                raise ValueError('Configured output path is unavailable')
            value = value[key]
        return 100*_normalize(value,component), value, 'measured', metric.reason
    if component.source == 'required_class_coverage':
        if goal.task_type not in ('classification','object_detection'):
            return None, None, 'not_applicable', 'Class presence does not apply to this task'
        # Presence only; NOT balance, sample adequacy, or verified label semantics.
        fraction, raw = _finding_fraction(findings,'required_class_count')
    elif component.source == 'scenario_coverage':
        fraction, raw = _finding_fraction(findings,'scenario_count')
    elif component.source == 'image_clarity':
        if ds.disk_path or 'laplacian_variance' not in ds.frame or 'readable' not in ds.frame:
            raise ValueError('Full-image clarity measurements are unavailable; previews are never scored')
        from pandas import to_numeric
        values = to_numeric(ds.frame.loc[ds.frame.readable,'laplacian_variance'],errors='coerce')
        finite = values.notna() & values.map(lambda v:math.isfinite(float(v)))
        if not len(values) or not finite.all():
            raise ValueError('Complete clarity measurements for readable images are unavailable')
        fraction = float((values >= component.threshold).mean())
        raw = {'meeting_criterion':int((values >= component.threshold).sum()), 'readable_images':len(values),
               'population_images':len(ds.frame),'threshold':component.threshold,
               'criterion':'four-neighbor Laplacian variance >= threshold; illustrative, uncalibrated'}
    else:
        ids = [id for id in component.metric_ids if metrics.get(id) is None or metrics[id].state != 'not_applicable']
        if not ids:
            return None, None, 'not_applicable', 'No overlap checks apply to this modality'
        values = [_metric_value(metrics.get(id)) for id in ids]
        if any(not _finite(v) or v < 0 for v in values): raise ValueError('Invalid overlap counts')
        fraction = sum(v == 0 for v in values)/len(values)
        raw = dict(zip(ids,values))
    return 100*fraction, raw, 'measured', 'Declared component criteria only; no performance guarantee'


def _empty(status, reason):
    return {'experimental':True,'status':status,'score_percent':None,'evidence_coverage_percent':0.,
            'score_range_percent':None,'components':[],'notice':NOTICE,'reason':reason}


def aggregate(snapshot, weights=None):
    """Reweight frozen component measurements without mutating a report or evidence."""
    result = copy.deepcopy(snapshot)
    if result.get('status') in ('disabled','configuration_error'): return result
    components = result['components']
    overrides = weights or {}
    unknown = set(overrides)-{c['id'] for c in components}
    if unknown: raise ValueError('Unknown score component IDs: '+', '.join(sorted(unknown)))
    for value in overrides.values():
        if not _finite(value) or not 0 <= value <= 1000000:
            raise ValueError('Weights must be finite nonnegative numbers, not booleans; maximum 1000000')
    for c in components:
        if c['id'] in overrides: c['weight'] = float(overrides[c['id']])
        c['definition']['weight'] = c['weight']
    active = [c for c in components if c['status'] != 'not_applicable' and c['weight'] > 0]
    known = [c for c in active if c['score_percent'] is not None]
    total, observed = sum(c['weight'] for c in active), sum(c['weight'] for c in known)
    numerator = sum(c['weight']*c['score_percent'] for c in known)
    for c in components:
        c['effective_weight_percent'] = 100*c['weight']/observed if c in known and observed else 0.
        c['contribution_points'] = c['weight']*c['score_percent']/observed if c in known and observed else None
    result.update(status='unavailable' if not observed else 'complete' if observed == total else 'partial',
        score_percent=numerator/observed if observed else None,
        evidence_coverage_percent=100*observed/total if total else 0.,
        observed_weight=observed, total_applicable_weight=total,
        score_range_percent=[numerator/total,(numerator+100*(total-observed))/total] if total else None)
    # The effective profile and content hash separate different weighting schemes.
    profile = copy.deepcopy(result['profile'])
    by_id = {c['id']:c['weight'] for c in components}
    for c in profile['components']: c['weight'] = by_id[c['id']]
    result['profile'] = profile
    result['profile_sha256'] = hashlib.sha256(json.dumps(profile,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result


def calculate(ds, goal, metrics, findings, profile=None):
    if not enabled(): return _empty('disabled','Disabled by EXPERIMENTAL_DATASET_SCORE_ENABLED')
    try:
        profile = profile or load_profile()
    except (OSError,ValueError) as exc:
        log.warning('Experimental scoring unavailable: %s',exc)
        return _empty('configuration_error','Experimental scoring configuration is invalid; correct the config file')
    if not profile.enabled: return _empty('disabled','Disabled in experimental score configuration')
    by_id = {m.metric_id:m for m in metrics}
    rows = []
    for spec in profile.components:
        row = {'id':spec.id,'label':spec.label,'weight':spec.weight,'score_percent':None,
               'raw_value':None,'status':'insufficient_evidence','reason':'','definition':spec.model_dump()}
        if ds.modality not in spec.modalities:
            row.update(status='not_applicable',reason='Component does not apply to this modality')
        else:
            try:
                score, raw, state, reason = _measure(spec,ds,goal,by_id,findings)
                row.update(score_percent=score,raw_value=raw,status=state,reason=reason)
            except (ValueError,TypeError,KeyError) as exc:
                row['reason'] = str(exc)
        rows.append(row)
    return aggregate({'experimental':True,'version':profile.version,'profile':profile.model_dump(),
        'components':rows,'notice':NOTICE,'formula':'sum(weight * component_score) / sum(observed component weights)',
        'scope':ds.profile.get('assessment_scope','full ingested dataset'), 'dataset_sha256':ds.sha256,
        'hard_failure_ids':[f.requirement.requirement_id for f in findings if f.requirement.severity == 'hard' and f.state == 'fail'],
        'unscored_metric_ids':sorted(set(by_id)-{s.metric_id for s in profile.components if s.metric_id}-
                                    {i for s in profile.components for i in s.metric_ids}-
                                    {'UAV-14' for s in profile.components if s.source=='image_clarity'})})
