"""Exact, whole-dataset SQL checks for disk-backed tables.

Column identifiers are generated internally; user values are bound parameters.
No metric is allowed to substitute the preview for the population. Unsupported
large-data calculations return insufficient evidence instead of sampled passes.
"""
import math
import pandas as pd
from backend.large_ingestion import connect
from metrics.core import result, MissingEvidence, CALCULATIONS
from backend.context import State

METADATA_ONLY = {'unit_consistency','provenance','sensor_modality','excluded_input_usage','preprocessing_leakage'}

def run_disk_metric(id,ds,g,p,reference=None):
    n=ds.profile['row_count']
    columns=ds.profile['columns']
    def col(name):
        if not name or name not in columns:
            raise MissingEvidence(f'Required evidence column is unavailable: {name or "unspecified"}')
        return f'c{columns.index(name)}'
    def label_text(name):
        c=col(name)
        return f"CASE WHEN {c} THEN 'True' WHEN NOT {c} THEN 'False' ELSE NULL END" if ds.profile['data_types'][name]=='bool' else f'cast({c} AS VARCHAR)'
    def emit(value=None,reason='Exact whole-dataset calculation; no acceptance threshold specified',state: State='pass',refs=None,details=None):
        return result(id,value,reason,state,n=n,refs=refs,details={'scope':'full_dataset','backend':'duckdb','population_rows':n,**(details or {})})
    if id in METADATA_ONLY:
        value=CALCULATIONS[id](ds,g,p,reference)
        value.evidence.details.update(scope='dataset_metadata',backend='duckdb')
        return value
    if id=='row_count': return emit(n)
    if id=='field_presence':
        if not p.get('column'): raise MissingEvidence('An exact column name has not been specified')
        return emit(p['column'] in columns)
    if id=='missingness':
        if p.get('column'):
            col(p['column'])
            return emit(ds.profile['missing_fraction'][p['column']])
        return emit(sum(ds.profile['missing_count'].values())/(n*len(columns)))
    if id=='non_finite_rate':
        numeric=[c for c in columns if ds.profile['data_types'][c] in ('int64','float64')]
        if not numeric: return emit(state='not_applicable',reason='No numeric cells')
        return emit(sum(ds.profile['non_finite_count'].get(c,0) for c in numeric)/(n*len(numeric)),details={'denominator':n*len(numeric)})
    if id=='sample_sufficiency': raise MissingEvidence('No universal training sample minimum exists; supply explicit count requirements')
    if id=='reference_comparison': raise MissingEvidence('Whole-dataset reference distances are not implemented for disk-backed tables; no preview-based representativeness score was substituted')
    if id=='class_distribution' and g.task_type not in ('classification','object_detection'):
        return emit(state='not_applicable',reason='Class distribution does not apply to this task')
    with connect(ds.disk_path) as db:
        def scalar(sql,params=None):
            row=db.execute(sql,params or []).fetchone()
            if row is None: raise MissingEvidence('Aggregate query returned no evidence')
            return row[0]
        def count(where='true',params=None): return int(scalar('SELECT count(*) FROM records WHERE '+where,params))
        def refs(where,params=None):
            return [f'row:{r[0]}' for r in db.execute('SELECT __rowid FROM records WHERE '+where+' ORDER BY __rowid LIMIT 30',params or []).fetchall()]
        if id=='missing_label_rate':
            c=col(g.prediction_target)
            return emit(count(c+' IS NULL')/n,refs=refs(c+' IS NULL'))
        if id=='usable_target_count':
            c=col(g.prediction_target)
            condition=f'isfinite(try_cast({c} AS DOUBLE))' if g.task_type=='regression' else c+' IS NOT NULL'
            return emit(count(condition),reason='Observed usable target count; not a training-sufficiency claim')
        if id in ('type_validity','range_violations','allowed_violations'):
            c=col(p['column'])
            observed=count(c+' IS NOT NULL')
            if not observed: raise MissingEvidence('No nonmissing values available for checking')
            parameters=[]
            numeric=f'try_cast({c} AS DOUBLE)'
            if id=='type_validity':
                expected=p['expected'].lower()
                parsing=ds.profile.get('parsing',{})
                if p['column'] in parsing.get('json_encoded_fields',[]):raise MissingEvidence('This field contains structured JSON values; flatten or explicitly transform it before scalar type validation')
                source_counts=parsing.get('source_type_counts',{}).get(p['column'])
                if source_counts and parsing.get('format') in ('json','jsonl','xlsx','feather','arrow'):
                    if expected in ('str','string','object'):return emit(source_counts.get('string',0)/observed,details={'denominator':observed,'source_types':source_counts})
                    if expected in ('bool','boolean'):return emit(source_counts.get('boolean',0)/observed,details={'denominator':observed,'source_types':source_counts})
                if expected in ('number','float','numeric','float64','int','integer','int64'):
                    condition=f'coalesce(isfinite({numeric}),false)'
                    if expected in ('int','integer','int64'): condition+=f' AND {numeric}=trunc({numeric})'
                elif expected in ('str','string','object'):
                    condition='true' if ds.profile['data_types'][p['column']]=='object' else 'false'
                elif expected in ('bool','boolean'):
                    condition='true' if ds.profile['data_types'][p['column']]=='bool' else 'false'
                elif expected in ('datetime','date','timestamp'):
                    condition=f'try_cast({c} AS TIMESTAMP) IS NOT NULL'
                    if count(f'{c} IS NOT NULL AND NOT ({condition})'):
                        raise MissingEvidence('Large-data timestamp checker cannot establish all values as ISO timestamps; supply a normalized timestamp column')
                else: raise MissingEvidence('Unsupported expected semantic type: '+expected)
                good=count(f'{c} IS NOT NULL AND ({condition})')
                return emit(good/observed,details={'denominator':observed})
            if id=='range_violations':
                low,high=p['range']
                good=f'coalesce(isfinite({numeric}),false)'
                if low is not None: good+=f' AND {numeric}>=?';parameters.append(low)
                if high is not None: good+=f' AND {numeric}<=?';parameters.append(high)
            else:
                kind=ds.profile['data_types'][p['column']]
                allowed=[v for v in p['allowed'] if isinstance(v,str)] if kind=='object' else [v for v in p['allowed'] if isinstance(v,bool)] if kind=='bool' else [v for v in p['allowed'] if isinstance(v,(int,float,bool))]
                good=c+' IN ('+','.join('?' for _ in allowed)+')' if allowed else 'false'
                parameters=list(allowed)
            bad=f'{c} IS NOT NULL AND NOT coalesce(({good}),false)'
            return emit(count(bad,parameters)/observed,refs=refs(bad,parameters),details={'denominator':observed})
        if id=='cross_field_validity':
            a,b=col(p['left']),col(p['right'])
            a,b=f'try_cast({a} AS DOUBLE)',f'try_cast({b} AS DOUBLE)'
            known=f'isfinite({a}) AND isfinite({b})'
            total=count(known)
            if not total: raise MissingEvidence('No comparable finite numeric pairs')
            relation='<' if p.get('relation')=='lt' else '<='
            return emit(count(f'{known} AND {a}{relation}{b}')/total,details={'denominator':total})
        if id=='duplicate_rate':
            fields=','.join(col(c) for c in columns)
            duplicates=int(scalar('SELECT coalesce(sum(k-1),0) FROM (SELECT count(*) k FROM records GROUP BY '+fields+')'))
            ds.profile['duplicate_rows']=duplicates
            return emit(duplicates/n,reason='Exact repeated rows may be legitimate; review before removal')
        if id=='duplicate_identifiers':
            c=col(p.get('column') or g.identifier_column)
            return emit(int(scalar(f'SELECT count({c})-count(DISTINCT {c}) FROM records')))
        if id=='conflicting_labels':
            target=col(g.prediction_target)
            fields=[col(c) for c in columns if c!=g.prediction_target]
            if not fields: raise MissingEvidence('No input features to group')
            return emit(int(scalar('SELECT count(*) FROM (SELECT 1 FROM records GROUP BY '+','.join(fields)+f' HAVING count(DISTINCT {target})>1)')),
                reason='Feature-identical groups with differing labels; repeated observations may have legitimate outcomes')
        if id in ('class_distribution','required_class_count'):
            target=col(g.prediction_target)
            if id=='required_class_count': return emit(count(f'({label_text(g.prediction_target)})=?',[p['class']]),details={'class':p['class']})
            rows=db.execute(f'SELECT {label_text(g.prediction_target)},count(*) FROM records WHERE {target} IS NOT NULL GROUP BY 1 ORDER BY 1 LIMIT 10001').fetchall()
            if len(rows)>10000: raise MissingEvidence('More than 10,000 classes; full label dictionary would exceed the report bound. Required class counts can still be evaluated exactly.')
            counts={name:int(value) for name,value in rows}
            total=sum(counts.values())
            return emit(counts,details={'observed_labels':total,'fractions':{k:v/total for k,v in counts.items()} if total else {},'imbalance_ratio':max(counts.values())/min(counts.values()) if counts else None})
        if id=='scenario_count':
            conditions,unknown,parameters=[],[],[]
            for name,expected in p['conditions'].items():
                c=col(name)
                if isinstance(expected,dict):
                    numeric=f'try_cast({c} AS DOUBLE)'
                    unknown.append(f'NOT coalesce(isfinite({numeric}),false)')
                    conditions.append(f'coalesce(isfinite({numeric}),false)')
                    for key,operator in [('min','>='),('max','<=')]:
                        if expected.get(key) is not None:
                            conditions.append(f'{numeric}{operator}?');parameters.append(expected[key])
                else:
                    unknown.append(c+' IS NULL')
                    options=expected if isinstance(expected,list) else [expected]
                    conditions.append(f'lower(cast({c} AS VARCHAR)) IN ('+','.join('?' for _ in options)+')' if options else 'false')
                    parameters.extend(str(v).lower() for v in options)
            if p.get('required_class'):
                conditions.append(f'cast({col(g.prediction_target)} AS VARCHAR)=?');parameters.append(p['required_class'])
            condition=' AND '.join(conditions) or 'true'
            observed=count(condition,parameters)
            unknown_count=count(' OR '.join(unknown)) if unknown else 0
            if observed==0 and unknown_count: raise MissingEvidence('Unknown scenario metadata prevents confirming absence of matching samples')
            return emit(observed,refs=refs(condition,parameters),details={'conditions':p['conditions'],'unknown_metadata_rows':unknown_count},reason='Observed intersection count; sufficiency needs an explicit minimum')
        if id in ('split_overlap','group_leakage'):
            split=col(g.split_column)
            if count(split+' IS NULL') or scalar(f'SELECT count(DISTINCT {split}) FROM records')<2:
                raise MissingEvidence('Complete assignments to at least two distinct splits are required')
            if id=='group_leakage':
                group=col(g.group_column)
                if count(group+' IS NULL'): raise MissingEvidence('Group identifiers are incomplete')
                fields=[group]
            else:
                names=g.required_inputs or [c for c in columns if c not in {g.split_column,g.group_column,g.identifier_column}]
                fields=[col(c) for c in names]
                if not fields: raise MissingEvidence('No content fields for overlap analysis')
            overlap=int(scalar('SELECT count(*) FROM (SELECT 1 FROM records GROUP BY '+','.join(fields)+f' HAVING count(DISTINCT {split})>1)'))
            return emit(overlap,reason='Exact whole-dataset overlap; this does not prove absence of all leakage')
        if id=='label_agreement':
            second=p.get('second_column') or ds.metadata.get('independent_annotation_column')
            if not second: raise MissingEvidence('Two declared independent annotation sets are required')
            a,b=col(g.prediction_target),col(second)
            pairs=f'SELECT cast({a} AS VARCHAR) a,cast({b} AS VARCHAR) b FROM records WHERE {a} IS NOT NULL AND {b} IS NOT NULL'
            total=int(scalar('SELECT count(*) FROM ('+pairs+')'))
            if not total: raise MissingEvidence('No complete annotation pairs')
            observed=float(scalar('SELECT avg(cast(a=b AS DOUBLE)) FROM ('+pairs+')'))
            expected=float(scalar('WITH pairs AS ('+pairs+'), a AS (SELECT a label,count(*) k FROM pairs GROUP BY a), b AS (SELECT b label,count(*) k FROM pairs GROUP BY b) SELECT coalesce(sum(a.k::DOUBLE*b.k),0) FROM a JOIN b USING(label)'))/(total*total)
            if expected>=1: raise MissingEvidence('Kappa is undefined for these annotation pairs')
            return emit((observed-expected)/(1-expected),details={'raw_agreement':observed,'complete_pairs':total},reason='Agreement measures consistency, not correctness or completeness; independence is user-declared')
        if id=='verified_label_error':
            trusted=g.reference_columns.get(g.prediction_target)
            if not trusted or not ds.metadata.get('trusted_reference_labels'): raise MissingEvidence('No explicitly trusted reference labels or audited sample')
            a,b=col(g.prediction_target),col(trusted)
            mask=f'{a} IS NOT NULL AND {b} IS NOT NULL'
            total=count(mask)
            if not total: raise MissingEvidence('No complete trusted-reference pairs')
            return emit(count(f'{mask} AND cast({a} AS VARCHAR) IS DISTINCT FROM cast({b} AS VARCHAR)')/total,details={'audit_coverage':total/n,'complete_pairs':total},reason='Observed disagreement with declared trusted references; generalization to unaudited rows is unverified')
        if id=='timeliness':
            c=col(g.date_column)
            if not ds.metadata.get('as_of') or g.max_data_age_days is None: raise MissingEvidence('A fixed as_of date and maximum age are required')
            parsed=f'try_cast({c} AS TIMESTAMPTZ)'
            if count(f'{c} IS NOT NULL AND {parsed} IS NULL'): raise MissingEvidence('Non-ISO dates cannot be verified by the large-data timestamp checker; normalize timestamps first')
            end=pd.to_datetime(ds.metadata['as_of'],utc=True,errors='coerce')
            if pd.isna(end): raise MissingEvidence('Invalid as_of date')
            start=end-pd.Timedelta(days=g.max_data_age_days)
            return emit(count(f'{parsed} BETWEEN ? AND ?',[start.to_pydatetime(),end.to_pydatetime()])/n)
        if id=='target_leakage':
            target=col(g.prediction_target)
            warnings={c:'Identical to target; review prediction-time availability' for c in columns if c!=g.prediction_target and count(f'cast({col(c)} AS VARCHAR) IS DISTINCT FROM cast({target} AS VARCHAR)')==0}
            return emit(warnings,reason='Exact target-copy heuristic only; correlation and mutual information are not calculated for large tables')
        if id=='evidence_summary':
            groups=int(scalar(f'SELECT count(DISTINCT {col(g.group_column)}) FROM records')) if g.group_column and g.group_column in columns else None
            return emit({'rows':n,'independent_groups':groups,'missing_provenance':[f for f in ['source','collection_date','dataset_version'] if not ds.metadata.get(f)]},
                reason='Full observed counts; no confidence interval without justified sampling and within-flight correlation assumptions')
    raise MissingEvidence('This metric has no whole-dataset implementation for disk-backed tables; the preview was not used as evidence')
