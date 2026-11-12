"""Metric registry and calculations independent of HTTP and presentation.

Every calculation returns its denominator and row/file references. Diagnostics
have no implicit acceptance threshold; the decision engine applies only declared
requirements. Unsupported or absent evidence is never represented by zero.
"""
from collections import Counter
from dataclasses import asdict
import math
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from backend.context import MetricDefinition, MetricResult, AssessmentEvidence, State
from backend.ingestion import Dataset
from metrics.featureAnalyzer import FeatureAnalyzer
from metrics.labelAnalyzer import LabelAnalyzer

REGISTRY = {}
CALCULATIONS = {}

def metric(id, dimension, method, modalities=('tabular', 'image'), **kwargs):
    def register(fn):
        REGISTRY[id] = MetricDefinition(metric_id=id, dimension=dimension, description=fn.__doc__ or id,
            calculation_method=method, supported_modalities=list(modalities), **kwargs)
        CALCULATIONS[id] = fn
        return fn
    return register

class MissingEvidence(Exception):
    pass

def result(id, value=None, reason='Observed diagnostic; no acceptance threshold specified', state: State='pass', n=None, refs=None, details=None):
    definition = REGISTRY.get(id)
    return MetricResult(metric_id=id, dimension=definition.dimension if definition else 'evidence_reliability',
        value=value, reason=reason, state=state, evidence=AssessmentEvidence(
            method=definition.calculation_method if definition else 'No implementation registered',
            sample_size=n, references=(refs or [])[:30], details=details or {}))

def run_metric(id, ds, goal, parameters=None, reference=None):
    if id not in REGISTRY:
        return result(id, state='insufficient_evidence', reason='Metric is not implemented; no value was fabricated')
    if ds.modality not in REGISTRY[id].supported_modalities:
        if id.startswith('UAV-'):
            from metrics.uav_metrics.assessment import unavailable
            return unavailable(id,ds,'Metric does not support this dataset modality.','not_applicable')
        return result(id, state='not_applicable', reason='Metric does not support this dataset modality')
    try:
        if id.startswith('UAV-'):
            calculated=CALCULATIONS[id](ds,goal,parameters or {},reference)
            # Structured diagnostics can be used in explicit requirements by
            # selecting an exact path; no aggregate score or hidden cutoff.
            path=(parameters or {}).get('output_path')
            if path and calculated.assessment_status in ('measured','estimated'):
                value=calculated.value
                parent=None
                for key in path:
                    if isinstance(value,dict) and value.get('status') in ('insufficient_evidence','unsupported','unverified','error'):
                        raise MissingEvidence('Requested submeasurement is unavailable; counts of inspected records are not a measured quality value')
                    if not isinstance(value,dict) or key not in value:
                        raise MissingEvidence('Requested structured measurement path is unavailable')
                    parent=value
                    value=value[key]
                if isinstance(value,dict) and value.get('status') in ('insufficient_evidence','unsupported','unverified','error') or isinstance(value,str) and value in ('unverified','insufficient_evidence','unsupported'):
                    raise MissingEvidence('Requested submeasurement is unavailable')
                calculated.value=value
                calculated.evidence.details['output_path']=path
                calculated.evidence.details['source_denominator']=calculated.evidence.sample_size
                if isinstance(parent,dict):
                    calculated.evidence.sample_size=parent.get('denominator',parent.get('count',calculated.evidence.sample_size))
                    calculated.unit=parent.get('unit',calculated.unit)
                if any(str(key).startswith('estimated_') for key in path):
                    calculated.assessment_status='estimated'
            return calculated
        if ds.disk_path is not None:
            from metrics.disk_metrics import run_disk_metric
            return run_disk_metric(id, ds, goal, parameters or {}, reference)
        return CALCULATIONS[id](ds, goal, parameters or {}, reference)
    except MissingEvidence as exc:
        if id.startswith('UAV-'):
            from metrics.uav_metrics.assessment import unavailable
            return unavailable(id,ds,str(exc))
        return result(id, state='insufficient_evidence', reason=str(exc))
    except Exception:
        # A failed required calculation must prevent successful certification.
        import logging
        logging.getLogger(__name__).exception('Metric failed: %s', id)
        if id.startswith('UAV-'):
            from metrics.uav_metrics.assessment import unavailable
            return unavailable(id,ds,'Calculation failed; inspect the local server log.','error')
        return result(id, state='error', reason='Calculation failed; inspect the local server log')

def need_column(ds: Dataset, column: str | None) -> pd.Series:
    if not column or column not in ds.frame:
        raise MissingEvidence(f'Required evidence column is unavailable: {column or "unspecified"}')
    series = ds.frame[column]
    # Returning a Series explicitly lets the editor distinguish column-vector
    # operations from scalar pandas operations and duplicate-name DataFrames.
    if not isinstance(series, pd.Series):
        raise MissingEvidence(f'Column must identify exactly one field: {column}')
    return series

def references(ds, mask):
    return [str(ds.frame.loc[i, 'image']) if ds.modality == 'image' else f'row:{i}' for i in ds.frame.index[mask]][:30]

@metric('row_count', 'integrity', 'Number of dataset rows or image files')
def row_count(ds,g,p,r):
    return result('row_count', len(ds.frame), n=len(ds.frame))

@metric('field_presence', 'integrity', 'Membership of named field in dataset schema', decision_relevant=True)
def field_presence(ds,g,p,r):
    if not p.get('column'):
        raise MissingEvidence('An exact column name has not been specified')
    return result('field_presence', p['column'] in ds.frame, details={'column': p['column']})

@metric('missingness', 'integrity', 'Missing cells / all cells, or missing values / rows for a named field', decision_relevant=True, unit='fraction')
def missingness(ds,g,p,r):
    df = ds.frame
    if p.get('column'):
        s = need_column(ds,p['column'])
        return result('missingness', float(s.isna().mean()), n=len(s), refs=references(ds,s.isna()))
    return result('missingness', float(df.isna().sum().sum()/df.size), n=df.size)

@metric('non_finite_rate', 'integrity', 'Positive or negative infinity / numeric cells (NaN reported separately)', unit='fraction')
def non_finite(ds,g,p,r):
    numeric = ds.frame.select_dtypes(include='number')
    if numeric.size == 0:
        return result('non_finite_rate', state='not_applicable', reason='No numeric cells')
    mask = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
    return result('non_finite_rate', float(mask.mean()), n=numeric.size, refs=references(ds,mask.any(axis=1)))

@metric('unreadable_rate', 'integrity', 'Unreadable image files / image files', modalities=('image',), unit='fraction')
def unreadable(ds,g,p,r):
    mask = ~ds.frame.readable
    return result('unreadable_rate', float(mask.mean()), n=len(mask), refs=references(ds,mask))

@metric('missing_label_rate', 'integrity', 'Missing target values / rows, or missing annotation files / images', unit='fraction')
def missing_label(ds,g,p,r):
    classification=ds.modality=='image' and g.task_type=='classification'
    mask = ~ds.frame.annotation_present if ds.modality == 'image' and not classification else need_column(ds,g.prediction_target).isna()
    return result('missing_label_rate', float(mask.mean()), n=len(mask), refs=references(ds,mask),
        reason='Empty annotation files are annotated negatives; missing objects within annotated images cannot be inferred' if ds.modality == 'image' else 'Missing target cells / rows')

@metric('usable_target_count', 'integrity', 'Observed nonmissing target rows; regression additionally requires finite numeric targets', decision_relevant=True)
def usable_targets(ds,g,p,r):
    s=need_column(ds,g.prediction_target)
    if g.task_type=='regression':
        values=pd.to_numeric(s,errors='coerce')
        mask=values.notna() & np.isfinite(values)
    else:
        mask=s.notna()
    return result('usable_target_count',int(mask.sum()),n=len(s),reason='At least one usable target is necessary for supervised learning; this does not establish training sample sufficiency')

@metric('type_validity', 'validity', 'Fraction of nonmissing values conforming to the explicitly declared semantic type', decision_relevant=True)
def type_validity(ds,g,p,r):
    s = need_column(ds,p['column']).dropna()
    if not len(s):
        raise MissingEvidence('No nonmissing values available for type checking')
    expected = p['expected'].lower()
    if expected in ('number','float','numeric','float64','int','integer','int64'):
        parsed = pd.to_numeric(s,errors='coerce')
        valid = parsed.notna() & np.isfinite(parsed)
        if expected in ('int','integer','int64'):
            valid &= parsed.mod(1).eq(0)
    elif expected in ('datetime','date','timestamp'):
        valid = pd.to_datetime(s,errors='coerce',utc=True).notna()
    elif expected in ('str','string','object'):
        valid = s.map(lambda x: isinstance(x,str))
    elif expected in ('bool','boolean'):
        valid = s.map(lambda x: isinstance(x,(bool,np.bool_)))
    else:
        raise MissingEvidence('Unsupported expected semantic type: ' + expected)
    return result('type_validity', float(valid.mean()), n=len(s), refs=[f'row:{i}' for i in s.index[~valid]])

@metric('range_violations', 'validity', 'Out-of-declared-range or nonnumeric/nonfinite nonmissing values / nonmissing values', unit='fraction')
def ranges(ds,g,p,r):
    s = need_column(ds,p['column']).dropna()
    if not len(s):
        raise MissingEvidence('No observed values for range validation')
    number = pd.to_numeric(s,errors='coerce')
    lo,hi = p['range']
    bad = number.isna() | ~np.isfinite(number)
    if lo is not None: bad |= number < lo
    if hi is not None: bad |= number > hi
    return result('range_violations', float(bad.mean()), n=len(s), refs=[f'row:{i}' for i in s.index[bad]])

@metric('allowed_violations', 'validity', 'Nonmissing values outside an explicit allowed set / nonmissing values', unit='fraction')
def allowed(ds,g,p,r):
    s = need_column(ds,p['column']).dropna()
    if not len(s): raise MissingEvidence('No observed values for allowed-value checking')
    bad = ~s.isin(p['allowed'])
    return result('allowed_violations',float(bad.mean()),n=len(s),refs=[f'row:{i}' for i in s.index[bad]])

@metric('unit_consistency', 'validity', 'Declared dataset unit equals required unit; no implicit conversions')
def units(ds,g,p,r):
    actual = ds.metadata.get('units',{}).get(p['column'])
    if actual is None: raise MissingEvidence('Dataset unit metadata is absent')
    return result('unit_consistency',actual == p['unit'],details={'actual':actual,'expected':p['unit']})

@metric('cross_field_validity', 'validity', 'Fraction of complete pairs satisfying left <= right (or left < right)')
def cross_field(ds,g,p,r):
    a,b = need_column(ds,p['left']),need_column(ds,p['right'])
    pairs = pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'b':pd.to_numeric(b,errors='coerce')}).dropna()
    if not len(pairs): raise MissingEvidence('No comparable numeric pairs')
    good = pairs.a < pairs.b if p.get('relation') == 'lt' else pairs.a <= pairs.b
    return result('cross_field_validity',float(good.mean()),n=len(pairs),refs=[f'row:{i}' for i in pairs.index[~good]])

@metric('bbox_invalid_rate', 'validity', 'Images with malformed annotations / images with annotation files', modalities=('image',), unit='fraction')
def bbox(ds,g,p,r):
    if ds.profile.get('dataset_kind') in ('image_classification','unlabelled_images'):
        return result('bbox_invalid_rate',state='not_applicable',reason='This collection does not provide object-detection bounding boxes')
    annotated = ds.frame[ds.frame.annotation_present]
    if annotated.empty: raise MissingEvidence('No annotation files')
    bad = ~annotated.annotation_valid
    issues=[issue for issue in ds.issues if issue['kind']=='invalid_annotation']
    ignored=[issue for issue in ds.issues if issue['kind']=='invalid_ignored_box']
    return result('bbox_invalid_rate',float(bad.mean()),n=len(annotated),
        refs=[issue['reference'] for issue in issues] or annotated.loc[bad,'image'].tolist(),
        reason=f'{int(bad.sum())} of {len(annotated)} annotated images contain invalid target annotations. Ignored-region geometry diagnostics are reported separately.',
        details={'annotation_format':ds.profile.get('annotation_format','yolo'),'invalid_target_examples':issues[:30],
                 'ignored_region_diagnostics':ignored[:30]})

@metric('duplicate_rate', 'duplication', 'Redundant full rows / rows, or repeated exact file SHA256 hashes / images', unit='fraction')
def duplicates(ds,g,p,r):
    mask = ds.frame.sha256.notna() & ds.frame.sha256.duplicated() if ds.modality == 'image' else ds.frame.duplicated()
    return result('duplicate_rate',float(mask.mean()),n=len(mask),refs=references(ds,mask),reason='Repeated observations may be legitimate; review acquisition context before removal')

@metric('duplicate_identifiers', 'duplication', 'Repeated nonmissing identifier values beyond their first occurrence', modalities=('tabular',))
def duplicate_ids(ds,g,p,r):
    s = need_column(ds,p.get('column') or g.identifier_column).dropna()
    mask = s.duplicated()
    return result('duplicate_identifiers',int(mask.sum()),n=len(s),refs=[f'row:{i}' for i in s.index[mask]])

@metric('conflicting_labels', 'duplication', 'Feature-identical groups with more than one nonmissing target value', modalities=('tabular',))
def conflicting(ds,g,p,r):
    need_column(ds,g.prediction_target)
    features = [c for c in ds.frame if c != g.prediction_target]
    if not features: raise MissingEvidence('No input features to group')
    counts = ds.frame.groupby(features,dropna=False)[g.prediction_target].nunique()
    return result('conflicting_labels',int((counts > 1).sum()),n=len(counts),reason='Label conflicts among feature-identical groups; repeated observations may have legitimate different outcomes')

@metric('near_duplicates', 'duplication', 'Candidate image pairs with 64-bit difference-hash Hamming distance <= configured threshold', modalities=('image',))
def near(ds,g,p,r):
    if ds.near_distance is None: raise MissingEvidence(ds.profile.get('near_duplicate_analysis','Near-duplicate analysis disabled; configure NEAR_DUPLICATE_DISTANCE'))
    return result('near_duplicates',len(ds.near_pairs),n=len(ds.frame),details={'threshold':ds.near_distance,'pairs':ds.near_pairs[:30]},reason='Perceptual candidates require review; image hashes do not establish semantic equivalence')

def class_counts(ds,g):
    if ds.modality == 'image':
        if g.task_type=='classification':
            labels=need_column(ds,g.prediction_target)
            return {str(k):int(v) for k,v in labels[ds.frame.readable].value_counts().items()}
        readable = set(ds.frame.loc[ds.frame.readable,'image'])
        return dict(Counter(o['class'] for o in ds.objects if o['image'] in readable))
    return {str(k):int(v) for k,v in need_column(ds,g.prediction_target).value_counts().items()}

@metric('class_distribution', 'label_quality', 'Class object counts for detection; class row counts for categorical targets')
def distribution(ds,g,p,r):
    if g.task_type not in ('classification','object_detection'):
        return result('class_distribution',state='not_applicable',reason='Class distribution does not apply to this task')
    counts = class_counts(ds,g)
    total = sum(counts.values())
    return result('class_distribution',counts,n=total,details={'fractions':{k:v/total for k,v in counts.items()} if total else {}, 'imbalance_ratio':max(counts.values())/min(counts.values()) if counts else None})

@metric('label_agreement', 'label_quality', 'Cohen kappa = (observed agreement - expected agreement)/(1 - expected agreement)', modalities=('tabular',))
def agreement(ds,g,p,r):
    second = p.get('second_column') or ds.metadata.get('independent_annotation_column')
    if not second: raise MissingEvidence('Two declared independent annotation sets are required')
    need_column(ds,second)
    need_column(ds,g.prediction_target)
    analysis = LabelAnalyzer(ds.frame,g.prediction_target,second).analyze()
    n = len(ds.frame[[g.prediction_target,second]].dropna())
    if analysis.cohen_kappa is None: raise MissingEvidence('Kappa is undefined for these complete annotation pairs')
    return result('label_agreement',analysis.cohen_kappa,n=n,details={'raw_agreement':analysis.raw_agreement},reason='Agreement measures consistency, not correctness or annotation completeness; independence is user-declared')

@metric('verified_label_error', 'label_quality', 'Mismatches / complete target-reference pairs; reference must be explicitly declared trusted', modalities=('tabular',))
def verified(ds,g,p,r):
    reference_column = g.reference_columns.get(g.prediction_target)
    if not reference_column or not ds.metadata.get('trusted_reference_labels'):
        raise MissingEvidence('No explicitly trusted reference labels or audited sample')
    a,b = need_column(ds,g.prediction_target), need_column(ds,reference_column)
    mask = a.notna() & b.notna()
    if not mask.any(): raise MissingEvidence('No complete trusted-reference pairs')
    return result('verified_label_error',float((a[mask] != b[mask]).mean()),n=int(mask.sum()),details={'audit_coverage':float(mask.mean())},reason='Observed disagreement with declared trusted references; generalization to unaudited records is unverified')

@metric('required_class_count', 'coverage', 'Usable annotated object count or labeled row count for the exact required class', decision_relevant=True)
def required_class(ds,g,p,r):
    if ds.profile.get('class_name_mapping')=='unavailable' and not p['class'].startswith('class_id:'):
        raise MissingEvidence('YOLO class-name mapping is missing; provide class_names or a dataset YAML file before checking semantic classes')
    counts = class_counts(ds,g)
    if ds.modality == 'image' and any(not x for x in ds.frame.annotation_valid) and p['class'] not in counts:
        raise MissingEvidence('Invalid annotation files prevent establishing class absence')
    return result('required_class_count',counts.get(p['class'],0),n=sum(counts.values()),details={'class':p['class']})

@metric('annotation_manifest_errors','validity','Count of unresolved image/category references in an annotation manifest',modalities=('image',))
def manifest_errors(ds,g,p,r):
    if ds.profile.get('annotation_format') not in ('coco','pascal_voc'):
        return result('annotation_manifest_errors',state='not_applicable',reason='No COCO/VOC manifest in this dataset')
    return result('annotation_manifest_errors',ds.profile.get('manifest_errors',0),refs=[i['reference'] for i in ds.issues if i['kind']=='orphan_annotation'])

@metric('scenario_count', 'coverage', 'Images/rows satisfying all declared conditions and, optionally, containing the required class', decision_relevant=True)
def scenario(ds,g,p,r):
    from metrics.uav_metrics.research import scenario_evidence
    evidence=scenario_evidence(ds,g,p['conditions'],p.get('required_class'))
    count=evidence['matching_image_count']
    if count == 0 and evidence['unknown_count']:
        raise MissingEvidence('Unknown scenario metadata prevents confirming absence of matching samples')
    return result('scenario_count',count,n=len(ds.frame),refs=evidence['matching_references'],
        details={**evidence,'conditions':p['conditions'],'unknown_metadata_rows':evidence['unknown_count']},
        reason='Observed matching image/row count (a lower bound when evidence is missing); sufficiency needs an explicit minimum')

@metric('sample_sufficiency', 'coverage', 'Compare observed counts with an explicitly supplied minimum')
def sufficiency(ds,g,p,r):
    raise MissingEvidence('No universal training sample minimum exists; supply scenario minimum_samples or an explicit count requirement')

@metric('reference_comparison', 'representativeness', 'Per-feature Wasserstein distance for numeric data; base-2 Jensen-Shannon divergence for categorical data')
def reference_comparison(ds,g,p,r):
    if r is None: raise MissingEvidence('No reference dataset supplied; diversity alone does not establish representativeness')
    columns = p.get('columns') or g.required_inputs
    if not columns: raise MissingEvidence('Select reference comparison columns explicitly')
    values = {}
    for column in columns:
        a,b = need_column(ds,column).dropna(),need_column(r,column).dropna()
        if not len(a) or not len(b): raise MissingEvidence('Empty reference comparison column')
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            unit_a=ds.metadata.get('units',{}).get(column)
            unit_b=r.metadata.get('units',{}).get(column)
            if unit_a!=unit_b:
                raise MissingEvidence('Reference numeric units must match explicitly; no implicit conversion is performed')
            a,b = a[np.isfinite(a)],b[np.isfinite(b)]
            if not len(a) or not len(b): raise MissingEvidence('No finite reference values')
            values[column] = {'wasserstein':float(wasserstein_distance(a,b)),'unit':ds.metadata.get('units',{}).get(column),
                'dataset_count':len(a),'reference_count':len(b)}
        else:
            aa,bb = a.astype(str).value_counts(normalize=True),b.astype(str).value_counts(normalize=True)
            cats = sorted(set(aa.index) | set(bb.index))
            values[column] = {'jensen_shannon_divergence':float(jensenshannon([aa.get(c,0) for c in cats],[bb.get(c,0) for c in cats],base=2)**2),
                'unit':'bits','dataset_count':len(a),'reference_count':len(b)}
    return result('reference_comparison',values,n=len(ds.frame),details={'reference_sha256':r.sha256},reason='Distances describe selected reference differences; reference deployment relevance remains user-declared')

@metric('split_overlap', 'leakage', 'Exact duplicate content observed in more than one declared split')
def split_overlap(ds,g,p,r):
    split = need_column(ds,g.split_column)
    if split.isna().any(): raise MissingEvidence('Split assignments are incomplete')
    if split.nunique() < 2: raise MissingEvidence('At least two distinct split assignments are required')
    if ds.modality == 'image':
        keys = ds.frame.sha256
    else:
        # Administrative identifiers and split assignments cannot hide identical model inputs.
        columns = g.required_inputs or [c for c in ds.frame if c not in {g.split_column,g.group_column,g.identifier_column}]
        if not columns: raise MissingEvidence('No content columns for overlap analysis')
        for c in columns: need_column(ds,c)
        keys = hash_pandas_object(ds.frame[columns],index=False).astype(str)
    grouped = pd.DataFrame({'key':keys,'split':split}).groupby('key').split.agg(lambda x:sorted(set(map(str,x))))
    overlaps = grouped[grouped.map(len) > 1]
    return result('split_overlap',len(overlaps),n=len(ds.frame),details={'split_combinations':overlaps.tolist()[:30]},reason='Exact overlap only; no overlap does not prove absence of all leakage')

@metric('group_leakage', 'leakage', 'Flight/session/group identifiers occurring in multiple declared splits')
def group_leakage(ds,g,p,r):
    group,split = need_column(ds,g.group_column),need_column(ds,g.split_column)
    if group.isna().any() or split.isna().any(): raise MissingEvidence('Group or split identifiers are incomplete')
    if split.nunique() < 2: raise MissingEvidence('At least two splits required')
    counts = ds.frame.groupby(g.group_column)[g.split_column].nunique()
    return result('group_leakage',int((counts > 1).sum()),n=len(counts),refs=[f'group:{i}' for i in counts.index[counts>1]])

@metric('near_split_overlap', 'leakage', 'Perceptually similar image pairs assigned to different splits', modalities=('image',))
def near_split(ds,g,p,r):
    if ds.near_distance is None: raise MissingEvidence('Near-duplicate analysis is disabled')
    split = need_column(ds,g.split_column)
    if split.isna().any() or split.nunique()<2: raise MissingEvidence('Complete multi-split assignments required')
    assignment = dict(zip(ds.frame.image,split))
    pairs = [(a,b) for a,b in ds.near_pairs if assignment[a] != assignment[b]]
    return result('near_split_overlap',len(pairs),n=len(ds.frame),details={'pairs':pairs[:30]},reason='Candidates for manual leakage review, not proof of semantic equivalence')

@metric('target_leakage', 'leakage', 'Input-target identity and feature heuristics; no proof of causal leakage', modalities=('tabular',))
def target_leakage(ds,g,p,r):
    need_column(ds,g.prediction_target)
    analysis = FeatureAnalyzer(ds.frame,g.prediction_target).analyze()
    return result('target_leakage',analysis.leakage_warnings,n=len(ds.frame),details={'suspicious_identifiers':analysis.suspicious_identifiers,'redundant_pairs':analysis.redundant_pairs},reason='Heuristic diagnostics; pipeline and temporal availability require independent review')

@metric('preprocessing_leakage', 'leakage', 'Inspect explicitly supplied fit-scope metadata')
def preprocessing(ds,g,p,r):
    scope = ds.metadata.get('preprocessing_fit_scope')
    if scope is None: raise MissingEvidence('Preprocessing pipeline metadata is absent')
    return result('preprocessing_leakage',scope,reason='Self-reported preprocessing fit scope; fitting on validation/test data risks leakage')

@metric('provenance', 'evidence_reliability', 'Availability of an explicitly requested provenance field', decision_relevant=True)
def provenance(ds,g,p,r):
    field = p.get('field','source')
    value = ds.metadata.get(field)
    if value is None or value == '': raise MissingEvidence(f'Missing provenance evidence: {field}')
    return result('provenance',True,details={'field':field,'declared_value_present':True},reason='User-supplied provenance is present; authenticity is not independently verified')

@metric('timeliness', 'validity', 'Fraction of parseable nonmissing dates between a fixed declared as_of date minus maximum age and as_of')
def timeliness(ds,g,p,r):
    dates = pd.to_datetime(need_column(ds,g.date_column),errors='coerce',utc=True)
    as_of = ds.metadata.get('as_of')
    if not as_of: raise MissingEvidence('Supply a fixed as_of date in metadata for reproducible age checks')
    if g.max_data_age_days is None: raise MissingEvidence('No maximum data age specified')
    end = pd.to_datetime(as_of,utc=True,errors='coerce')
    if pd.isna(end): raise MissingEvidence('Invalid as_of date')
    start = end-pd.Timedelta(days=g.max_data_age_days)
    valid = dates.between(start,end)
    return result('timeliness',float(valid.mean()),n=len(dates),refs=references(ds,~valid),details={'as_of':str(end),'maximum_age_days':g.max_data_age_days})

@metric('sensor_modality', 'integrity', 'Presence of the declared sensor modality in user-supplied sensor_types metadata')
def sensor_modality(ds,g,p,r):
    types = ds.metadata.get('sensor_types')
    if not types: raise MissingEvidence('Sensor types and their column mappings require declared metadata')
    declared = types.get(p['modality']) if isinstance(types,dict) else None
    if declared is None: raise MissingEvidence('No column mapping for requested sensor modality')
    if not isinstance(declared,list) or not declared: raise MissingEvidence('Sensor mapping must be a nonempty list of columns')
    return result('sensor_modality',all(c in ds.frame for c in declared),details={'declared_columns':declared},reason='Schema support only; sensor calibration and synchronization are not established')

@metric('excluded_input_usage', 'leakage', 'Check a declared model-input list against excluded fields')
def excluded_usage(ds,g,p,r):
    inputs = ds.metadata.get('model_input_columns')
    if not isinstance(inputs,list): raise MissingEvidence('Excluded input use requires declared model_input_columns pipeline metadata')
    return result('excluded_input_usage',p['column'] not in inputs,reason='Self-reported pipeline inputs; dataset column presence alone does not prove model use')

@metric('evidence_summary', 'evidence_reliability', 'Observed evidence availability; no invented confidence intervals for correlated samples')
def evidence_summary(ds,g,p,r):
    groups = None
    if g.group_column and g.group_column in ds.frame:
        groups = int(ds.frame[g.group_column].nunique())
    missing = [f for f in ['source','collection_date','dataset_version'] if not ds.metadata.get(f)]
    return result('evidence_summary',{'rows':len(ds.frame),'independent_groups':groups,'missing_provenance':missing},n=len(ds.frame),reason='No population confidence interval: sampling design and within-flight correlation are not established; neighboring frames are not assumed independent')

DIAGNOSTICS = ['row_count','missingness','non_finite_rate','unreadable_rate','missing_label_rate','bbox_invalid_rate',
    'duplicate_rate','duplicate_identifiers','conflicting_labels','near_duplicates','class_distribution',
    'label_agreement','verified_label_error','sample_sufficiency','reference_comparison','split_overlap',
    'group_leakage','near_split_overlap','target_leakage','preprocessing_leakage','evidence_summary']

# Parameterized columns are declared symbolically, because their actual names
# come from the confirmed context rather than a fixed UAV schema.
for id in ['field_presence','missingness','type_validity','range_violations','allowed_violations','unit_consistency','duplicate_identifiers']:
    REGISTRY[id].required_dataset_fields = ['parameters.column (optional for overall missingness)']
    REGISTRY[id].required_context = ['parameters']
for id in ['required_class_count','scenario_count']:
    REGISTRY[id].required_context = ['required class / scenario parameters']
    REGISTRY[id].applicability = 'Usable class annotations or declared scenario metadata are required'
for id in ['split_overlap','group_leakage','near_split_overlap']:
    REGISTRY[id].required_context = ['split_column; group_column for group leakage']
    REGISTRY[id].evidence_requirements = 'Complete observed split assignments; independent group identifiers when relevant'
for id in ['label_agreement','verified_label_error']:
    REGISTRY[id].required_context = ['prediction_target and explicitly independent/trusted annotation metadata']
    REGISTRY[id].evidence_requirements = 'Complete paired annotations; declared independence/trust is not authenticated'
for id in ['field_presence','unit_consistency','provenance','sensor_modality','excluded_input_usage']:
    REGISTRY[id].output_type = 'boolean'
for id in ['class_distribution','reference_comparison','target_leakage','evidence_summary']:
    REGISTRY[id].output_type = 'object'
REGISTRY['preprocessing_leakage'].output_type = 'string'
REGISTRY['reference_comparison'].evidence_requirements = 'User-selected corresponding columns in a supplied reference dataset'
REGISTRY['scenario_count'].required_dataset_fields = ['All declared scenario condition fields; usable classes if requested']
REGISTRY['timeliness'].required_context = ['date_column, max_data_age_days, fixed metadata.as_of']

# One registry and one public execution path for both legacy and UAV metrics.
from metrics.uav_metrics.registry import register as register_uav
register_uav(metric,CALCULATIONS,REGISTRY)
