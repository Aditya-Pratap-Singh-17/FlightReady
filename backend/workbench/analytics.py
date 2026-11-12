"""Join actual results and aggregate seeds before dataset-level association analysis."""
from collections import defaultdict
import hashlib
import json
import math
import numpy as np
from scipy import stats


def numeric_metrics(certificate):
    result = {}
    def visit(prefix, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[prefix] = float(value)
        elif isinstance(value, dict):
            if value.get('status') in ('insufficient_evidence','unsupported','error','unverified'):
                return
            for key, nested in value.items():
                if key in ('sampling','settings','configuration','protocol','criteria','size_bins','definition','histogram','required_minimum'):
                    continue
                visit(prefix + '.' + key, nested)
    for metric in certificate.get('metrics', []):
        if metric['state'] in ('pass', 'warning', 'fail'):
            visit(metric['metric_id'], metric['value'])
    quality = certificate.get('quality_profile') or {}
    gates=quality.get('critical_requirements',[])
    if gates and (any(g['status']=='fail' for g in gates) or all(g['status']=='pass' for g in gates)):
        result['baseline.critical_gates_pass']=0. if any(g['status']=='fail' for g in gates) else 1.
    for dimension in quality.get('dimensions', []):
        if dimension['score'] is not None:
            id = dimension['id']
            result['dimension.'+id] = dimension['score']
            measured = [m for m in dimension['metrics'] if m['effective_weight'] > 0]
            result['adaptive_arithmetic.'+id] = sum(m['effective_weight'] * m['score'] for m in measured)
            result['equal_arithmetic.'+id] = float(np.mean([m['score'] for m in measured]))
    # Alternative global predictors are research baselines only, never readiness
    # decisions. Their values exist only if every applicable dimension is scored.
    dimensions = [d for d in quality.get('dimensions', []) if d['effective_weight'] > 0]
    if dimensions and all(d['score'] is not None for d in dimensions):
        from metrics.quality_aggregation import geometric
        result['baseline.equal_dimension_mean'] = float(np.mean([d['score'] for d in dimensions]))
        result['baseline.adaptive_arithmetic'] = sum(d['score']*d['effective_weight'] for d in dimensions)
        result['baseline.adaptive_geometric'] = geometric([d['score'] for d in dimensions], [d['effective_weight'] for d in dimensions])
    score=certificate.get('experimental_score') or {}
    if score.get('status') in ('complete','partial') and score.get('score_percent') is not None:
        result['experimental_weighted_score']=float(score['score_percent'])
    return result


def comparison(store, evaluation_ids):
    records = [store.get(id, 'evaluation') for id in evaluation_ids]
    models = [store.get(r['payload']['model_id'], 'model') for r in records]
    differences = {}
    keys = ('architecture', 'initialization_sha256', 'epochs', 'batch', 'image_size', 'learning_rate', 'seed', 'selected_device', 'classes')
    for key in keys:
        values = [m['payload']['configuration'].get(key) for m in models]
        if any(value != values[0] for value in values[1:]):
            differences[key] = values
    if any(m['payload'].get('imported') for m in models):
        differences['imported_provenance'] = 'Imported checkpoint training conditions are user declared and unverified'
    if any(not r['payload'].get('settings',{}).get('ground_truth_protocol_version') for r in records):
        differences['legacy_ground_truth_identity']='Historical evaluation lacks the annotation-aware protocol hash; re-evaluate before claiming a controlled comparison.'
    for key in ('test_sha256', 'settings'):
        values = [r['payload'][key] for r in records]
        if any(value != values[0] for value in values[1:]):
            differences[key] = values
    sizes = [m['payload']['training_images'] for m in models]
    if len(set(sizes)) > 1:
        differences['training_images'] = sizes
    return {'evaluation_ids': evaluation_ids, 'differences': differences, 'matched_conditions': not differences,
            'conclusion': 'Descriptive comparison only. Differences in training size/distribution, configuration, or test conditions can confound dataset rankings.'}


def research(store, metric=None, performance='mAP50_95', test_sha256=None, configuration=None, dataset_id=None, scenario=None, model_ids=None):
    rows = []
    seen = set()
    # Keep only latest evaluation of each model on identical conditions.
    for evaluation in store.list('evaluation'):
        e = evaluation['payload']
        if model_ids is not None and e['model_id'] not in model_ids:
            continue
        if test_sha256 and e['test_sha256'] != test_sha256:
            continue
        model = store.get(e['model_id'], 'model')
        m = model['payload']
        dataset = store.get(m['dataset_id'], 'dataset')
        d = dataset['payload']
        if dataset_id and dataset_id != m['dataset_id']:
            continue
        readiness = store.get(m['readiness_id'], 'job')['payload']
        context = readiness['config']['context']
        if scenario:
            goal=context.get('goal',{})
            evidence={'tags':context.get('selected_tags',{}),'scenarios':goal.get('scenarios',[]),'target_environment':goal.get('target_environment',{})}
            if scenario.casefold() not in json.dumps(evidence).casefold():continue
        cfg = m['configuration']
        experimental=readiness['result'].get('experimental_score') or {}
        development=m.get('development_manifests',[])
        source_hashes=sorted(set(development[0]['hashes'].values())) if development else []
        source_group_column=development[0].get('group_column') if development else None
        source_groups=sorted(set(development[0].get('groups',{}).values())) if development else []
        source_identity=hashlib.sha256(json.dumps(source_hashes).encode()).hexdigest() if source_hashes else d['sha256']
        comparable = {k: cfg.get(k) for k in ('architecture', 'initialization_sha256', 'epochs', 'batch', 'image_size', 'learning_rate', 'selected_device', 'classes')}
        comparable.update(test_sha256=e['test_sha256'], evaluation_settings=e['settings'], context=context, task='object_detection')
        if experimental.get('profile_sha256'):
            # A partial score is comparable only with identical scoring rules and
            # the same available component set. Never pool unlike denominators.
            comparable['experimental_score_profile']=experimental['profile_sha256']
            comparable['experimental_score_observed_components']=sorted(c['id'] for c in experimental['components']
                if c['weight']>0 and c['score_percent'] is not None and c['status']!='not_applicable')
        quality=m.get('quality_profile') or readiness['result'].get('quality_profile') or {}
        if quality:
            comparable['quality_policy']={'profile_sha256':quality['profile_sha256'],'task':quality['task'],'scenario':quality['scenario'],
                'confirmed_overrides':quality['confirmed_overrides'],
                'measured_components':sorted(c['id'] for c in quality['individual_metrics'] if c['status']=='measured')}
        signature = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()
        if configuration and configuration != signature:
            continue
        unique = (e['model_id'], signature)
        if unique in seen:
            continue
        seen.add(unique)
        vmap=m.get('validation',{}).get('mAP50_95');tmap=e['metrics'].get('mAP50_95')
        validation_settings=m.get('validation_settings')
        matched_reference=validation_settings and all(e['settings'].get(k)==v for k,v in validation_settings.items())
        gap=vmap-tmap if vmap is not None and tmap is not None and matched_reference else None
        rows.append({'evaluation_id': evaluation['id'], 'model_id': e['model_id'], 'dataset_id': m['dataset_id'], 'dataset': d['name'],
            'dataset_version': d['version'], 'dataset_sha256': d['sha256'], 'source_identity':source_identity,'source_image_hashes':source_hashes,
            'source_group_column':source_group_column,'source_recording_groups':source_groups,
            'context': context, 'overall_readiness_score': None,
            'readiness_status': readiness['result']['overall_status'], 'quality_metrics': {**numeric_metrics({**readiness['result'],'quality_profile':quality}), 'baseline.training_images':m['training_images']},
            'quality_profile':quality, 'intervention':d.get('provenance',{}).get('intervention'),
            'training_provenance_verified':not m.get('imported',False),
            'parent_dataset_id':d.get('provenance',{}).get('parent_dataset_id'),
            'experimental_quality_score':experimental.get('score_percent'),
            'experimental_score_coverage':experimental.get('evidence_coverage_percent'),
            'experimental_score_profile_sha256':experimental.get('profile_sha256'),
            'training_images': m['training_images'], 'configuration': cfg, 'configuration_group': signature, 'seed': cfg['seed'],
            'test_dataset_id': e['dataset_id'], 'test_dataset': store.get(e['dataset_id'])['payload']['name'], 'test_sha256': e['test_sha256'],
            'evaluation_settings':e['settings'],
            'metrics': e['metrics'], 'generalization_gap': gap, 'date': evaluation['created_at'], 'artificial': bool(d.get('provenance', {}).get('artificial'))})
    groups = defaultdict(list)
    for row in rows:
        if not row['artificial'] and row['configuration']['model_type'] != 'combined' and row['training_provenance_verified']:
            groups[(row['source_identity'], row['configuration_group'])].append(row)
    aggregated = []
    excluded=[]
    for (dataset_sha, signature), values in groups.items():
        if any(value['quality_metrics']!=values[0]['quality_metrics'] for value in values[1:]):
            excluded.append('Different readiness measurements on the same image source/configuration were excluded from dataset-level correlation; use a fixed training manifest and context across seeds.')
            continue
        seed_groups = defaultdict(list)
        for row in values:
            seed_groups[row['seed']].append(row['metrics'].get(performance))
        samples = [float(np.mean([v for v in numbers if v is not None])) for numbers in seed_groups.values() if any(v is not None for v in numbers)]
        if not samples:
            continue
        n = len(samples)
        sd = float(np.std(samples, ddof=1)) if n > 1 else None
        mean = float(np.mean(samples))
        margin = float(stats.t.ppf(.975, n-1))*sd/math.sqrt(n) if n > 1 else None
        aggregated.append({'dataset_sha256': dataset_sha, 'dataset': values[0]['dataset'], 'configuration_group': signature,
            'independent_seeds': n, 'mean': mean, 'standard_deviation': sd, 'confidence_interval_95': [mean-margin, mean+margin] if margin is not None else None,
            'quality_metrics': values[0]['quality_metrics'],'source_image_hashes':values[0]['source_image_hashes'],
            'source_group_column':values[0]['source_group_column'],'source_recording_groups':values[0]['source_recording_groups'],
            'uncertainty_scope': 'Variation across training seeds; does not quantify independent test-domain uncertainty.'})
    metric_options = sorted({k for row in rows for k in row['quality_metrics']})
    correlations = []
    predictive_validity = []
    selected_metric = metric or (metric_options[0] if metric_options else None)
    by_config = defaultdict(list)
    for group in aggregated:
        if selected_metric in group['quality_metrics']:
            by_config[group['configuration_group']].append(group)
    for signature, values in by_config.items():
        # Shared images make nominally different datasets dependent. Exclude all
        # affected sources instead of pretending that a dataset ID proves independence.
        overlapping=set()
        for i,a in enumerate(values):
            for j,b in enumerate(values[:i]):
                shared_groups=a['source_group_column'] and a['source_group_column']==b['source_group_column'] and set(a['source_recording_groups']) & set(b['source_recording_groups'])
                if set(a['source_image_hashes']) & set(b['source_image_hashes']) or shared_groups:
                    overlapping.update((i,j))
        if overlapping:
            excluded.append('Sources sharing exact image content or declared comparable recording groups were excluded from independent-dataset correlations. Group identifiers must use a common namespace.')
            values=[v for i,v in enumerate(values) if i not in overlapping]
        x = [g['quality_metrics'][selected_metric] for g in values]
        y = [g['mean'] for g in values]
        n = len(values)
        valid = n >= 3 and len(set(x)) > 1 and len(set(y)) > 1
        pearson_result = stats.pearsonr(x,y) if valid else None
        interval = pearson_result.confidence_interval(.95) if valid and n >= 4 else None
        pearson_interval = [float(interval.low),float(interval.high)] if interval and math.isfinite(interval.low) and math.isfinite(interval.high) else None
        # Resample independent datasets, not seed-level rows. Constant bootstrap
        # samples have undefined rho and are excluded explicitly.
        bootstrap=[]
        if valid and n >= 6:
            rng=np.random.default_rng(42)
            for _ in range(1000):
                indexes=rng.integers(0,n,n)
                bx,by=np.asarray(x)[indexes],np.asarray(y)[indexes]
                if len(set(bx))>1 and len(set(by))>1:
                    bootstrap.append(float(stats.spearmanr(bx,by).statistic))
        correlations.append({'configuration_group': signature, 'metric': selected_metric, 'performance': performance, 'independent_datasets': n,
            'pearson': float(pearson_result.statistic) if valid else None,
            'spearman': float(stats.spearmanr(x, y).statistic) if valid else None,
            'pearson_confidence_interval_95':pearson_interval,
            'spearman_confidence_interval_95':np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap)>=900 else None,
            'interval_method':'Pearson Fisher-z interval (n>=4); Spearman independent-dataset percentile bootstrap, 1000 draws, seed 42 (n>=6). Exploratory; no multiplicity correction.',
            'points': [{'dataset': g['dataset'], 'x': a, 'y': b, 'seeds': g['independent_seeds'], 'interval': g['confidence_interval_95']} for g, a, b in zip(values, x, y)],
            'interpretation': 'Exploratory association only; small sample and dataset confounding prevent causal conclusions.' if valid else 'At least three independent comparable datasets with varying measurements and performance are required.'})
        predictive_validity.append(predictive_comparison(values, signature, performance))
    summary = ['No downstream results are available.' if not rows else f'{len(rows)} evaluated runs across {len({r["dataset_sha256"] for r in rows})} dataset snapshots.']
    measured=[r for r in rows if r['metrics'].get('mAP50_95') is not None]
    if measured:
        best = max(measured, key=lambda r: r['metrics']['mAP50_95'])
        summary.append(f'Highest observed mAP@0.50:0.95: {best["metrics"]["mAP50_95"]:.4f}, model {best["model_id"]}, test dataset {best["test_dataset"]}. Rankings across unmatched conditions are descriptive only.')
    summary.extend(['Readiness decisions are requirement-based. Experimental weighted quality summaries are provisional and are not predicted accuracy.',
        'Seed repetitions are aggregated within dataset/configuration; artificial variants are excluded from natural-dataset correlations.',
        'Imported checkpoints with unverified training provenance are shown descriptively and excluded from controlled predictive correlations.',
        'Generalization gap is validation mAP minus external-test mAP; it is descriptive and depends on domain difficulty.',
        'No causal relationship is established. Independent domains, matched budgets, multiple seeds and representative datasets are needed.'])
    if any(not r['evaluation_settings'].get('ground_truth_protocol_version') for r in rows):
        summary.append('Historical evaluations without annotation-aware test hashes remain descriptive. Re-evaluate them under the current protocol before drawing controlled-comparison conclusions.')
    summary.extend(sorted(set(excluded)))
    ready=sorted({row['dataset'] for row in rows if row['readiness_status']=='READY'})
    if ready:summary.append('Datasets satisfying all confirmed requirements in observed runs: '+', '.join(ready)+'. This is not a numerical quality ranking.')
    valid=[c for c in correlations if c['spearman'] is not None]
    if valid:
        strongest=max(valid,key=lambda c:abs(c['spearman']))
        summary.append(f'The selected dimension {strongest["metric"]} has descriptive Spearman rho={strongest["spearman"]:.3f} with {performance} across {strongest["independent_datasets"]} comparable independent image sources; this is exploratory, not statistically established evidence.')
    else:
        summary.append('Available comparable independent image sources are insufficient to establish whether higher measured readiness predicts downstream performance.')
    stability=ranking_stability(rows,performance)
    summary.append(stability['interpretation'])
    matrix=[]
    for row in rows:
        condition_hash=hashlib.sha256(json.dumps(row['evaluation_settings'],sort_keys=True).encode()).hexdigest()[:6]
        matrix.append({'model_id':row['model_id'],'test_dataset':row['test_dataset']+' · '+row['test_sha256'][:6]+' / '+condition_hash,
                       'mAP50_95':row['metrics']['mAP50_95']})
    pairs=[]
    for variant in rows:
        if not variant['artificial'] or not variant['parent_dataset_id']:
            continue
        controls=[r for r in rows if r['dataset_id']==variant['parent_dataset_id'] and r['seed']==variant['seed'] and r['test_sha256']==variant['test_sha256']
                  and r['evaluation_settings']==variant['evaluation_settings']
                  and all(r['configuration'].get(k)==variant['configuration'].get(k) for k in ('architecture','initialization_sha256','epochs','batch','image_size','learning_rate'))]
        if controls:
            control=controls[0]
            keys=('profile_sha256','task','scenario','confirmed_overrides')
            policies_match=all(variant['quality_profile'].get(k)==control['quality_profile'].get(k) for k in keys)
            pairs.append({'control_model_id':control['model_id'],'variant_model_id':variant['model_id'],'intervention':variant['intervention'],
                'training_sizes':{'control':control['training_images'],'variant':variant['training_images']},
                'size_matched':control['training_images']==variant['training_images'],
                'performance_delta':variant['metrics'].get(performance)-control['metrics'].get(performance) if variant['metrics'].get(performance) is not None and control['metrics'].get(performance) is not None else None,
                'quality_policies_match':policies_match,
                'quality_deltas':{k:v-control['quality_metrics'][k] for k,v in variant['quality_metrics'].items() if k in control['quality_metrics']
                    and (policies_match or not k.startswith(('dimension.','adaptive_arithmetic.','equal_arithmetic.','baseline.adaptive','baseline.equal')))},
                'interpretation':'Paired same-seed observation on related data, not an independent dataset or universal causal validation.'})
    return {'rows': rows, 'aggregates': aggregated, 'metric_options': metric_options, 'selected_metric': selected_metric, 'correlations': correlations,
            'summary': summary, 'performance_matrix':matrix, 'predictive_validity':predictive_validity, 'intervention_pairs':pairs,
            'ranking_stability':stability,
            'empirical_validation':{'status':'recorded_outcomes_available' if rows else 'pending',
                'notice':'Empirical validation is pending until real training and independent held-out evaluations are recorded.' if not rows else 'Recorded outcomes do not alone establish predictive validity or causation.'}}


def ranking_stability(rows,performance):
    protocols=defaultdict(lambda:defaultdict(dict))
    for row in rows:  # rows are newest-first: retain the latest matching observation
        value=row['metrics'].get(performance)
        if value is None:
            continue
        signature=fingerprint_settings(row['evaluation_settings'])
        protocols[signature][row['test_sha256']].setdefault(row['model_id'],value)
    groups=[]
    for protocol,tests in protocols.items():
        if len(tests)<2:
            continue
        common=set.intersection(*(set(v) for v in tests.values()))
        if len(common)<2:
            continue
        # Tied performance has no strict ranking. Keep tied groups instead of
        # letting input/model ID order manufacture a ranking change.
        orders=[]
        for values in tests.values():
            levels=sorted({values[id] for id in common},reverse=True)
            orders.append(tuple(tuple(sorted(id for id in common if values[id]==level)) for level in levels))
        groups.append({'protocol_sha256':protocol,'test_manifests':len(tests),'shared_models':len(common),'ranking_changed':len(set(orders))>1})
    return {'groups':groups,'status':'descriptive' if groups else 'insufficient_evidence',
        'interpretation':('Observed model rankings change across test manifests within matched evaluation protocols; this is descriptive and does not establish a domain or quality effect.'
            if any(g['ranking_changed'] for g in groups) else 'Observed model rankings are unchanged across jointly evaluated test manifests within matched protocols.'
            if groups else 'Ranking stability requires the same multiple models on multiple test manifests under matching evaluation settings; available evidence is insufficient.')}


def fingerprint_settings(settings):
    return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()


def predictive_comparison(values, signature, performance):
    """Leave-one-independent-dataset-out OLS; scoring policies are never fitted here."""
    keys=sorted(set.intersection(*(set(v['quality_metrics']) for v in values))) if values else []
    keys=[k for k in keys if k.startswith(('dimension.','adaptive_arithmetic.','equal_arithmetic.','baseline.'))]
    output={'configuration_group':signature,'independent_datasets':len(values),'performance':performance,
            'method':'Leave-one-independent-dataset-out univariate ordinary least squares, with training-fold centering/scaling; comparison against training-fold mean.',
            'status':'insufficient_evidence','predictors':[],
            'limitations':'At least six independent matched dataset sources are required. Results remain exploratory and do not calibrate weights. No test-fold performance is used for fitting.'}
    if len(values)<6:
        return output
    y=np.asarray([v['mean'] for v in values],dtype=float)
    for key in keys:
        x=np.asarray([v['quality_metrics'][key] for v in values],dtype=float)
        if len(set(x))<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        predictions=[];baselines=[]
        for i in range(len(values)):
            mask=np.arange(len(values))!=i
            mean,sd=float(x[mask].mean()),float(x[mask].std())
            design=np.column_stack([np.ones(mask.sum()),(x[mask]-mean)/(sd or 1)])
            coefficients=np.linalg.lstsq(design,y[mask],rcond=None)[0]
            predictions.append(float(np.dot([1,(x[i]-mean)/(sd or 1)],coefficients)))
            baselines.append(float(y[mask].mean()))
        errors=np.asarray(predictions)-y
        output['predictors'].append({'metric':key,'mae':float(np.abs(errors).mean()),'rmse':float(np.sqrt((errors**2).mean())),
            'mean_baseline_mae':float(np.abs(np.asarray(baselines)-y).mean()),
            'held_out_predictions':[{'dataset':v['dataset'],'observed':float(a),'predicted':b} for v,a,b in zip(values,y,predictions)]})
    output['status']='exploratory' if output['predictors'] else 'insufficient_variation'
    return output
