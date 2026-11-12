"""Chart-ready views of stored evidence, shared by UI and report exports."""
from metrics.uav_metrics.research import VERSION


def evidence_sections(metrics):
    by_id = {m['metric_id']: m for m in metrics}
    sections = []
    def section(id, title, mid, raw, evidence_type, sample=None, coverage=None, denominator=None):
        metric = by_id.get(mid, {})
        item = {'id': id, 'title': title, 'metric_id': mid, 'status': raw.get('status', 'insufficient_evidence'),
                'evidence_type': evidence_type, 'sample_size': sample, 'coverage': coverage, 'denominator': denominator,
                'provenance': metric.get('provenance', {}), 'version': VERSION, 'limitations': raw.get('limitations', metric.get('limitations', [])),
                'methodology': {key:raw[key] for key in ('sampling','protocol','model_input','size_bins','complexity_cross_tab_definition','review_candidates','representativeness','independence_status') if key in raw},
                'plots': [], 'rows': []}
        sections.append(item)
        return item
    def rows(item, values, unit, denominator=None):
        item['rows'].extend({'label': key, 'value': value, 'unit': unit, 'denominator': denominator, 'status': 'measured' if value is not None else 'insufficient_evidence'} for key, value in values.items())
    def plot(item, title, values, unit, denominator, status='measured'):
        available=not (status in ('insufficient_evidence','unsupported','error') and not any(v for v in values.values() if isinstance(v,(int,float))))
        item['plots'].append({'title': title, 'unit': unit, 'denominator': denominator, 'status': status,
                              'rows': [{'label': str(key), 'value': value} for key, value in values.items() if value is not None] if available else []})
    def histogram(item, title, raw):
        plot(item, title, {f'{v["lower"]:.4g} to {v["upper"]:.4g}': v['count'] for v in raw.get('histogram', [])},
             'objects/bin; '+str(raw.get('unit', 'proxy')), raw.get('count'), raw.get('status', 'insufficient_evidence'))
    def value(mid):
        return by_id.get(mid, {}).get('value') or {}
    scale = value('UAV-08').get('scale_analysis', {})
    item = section('object_scale', 'Object-scale distributions', 'UAV-08', scale, 'observed annotation geometry', scale.get('object_count'), None, 'valid annotated objects')
    plot(item, 'Native-area object bins', scale.get('object_counts', {}), 'objects/bin', scale.get('object_count'), scale.get('status', 'insufficient_evidence'))
    plot(item, 'Images containing each bin (images may appear in multiple bins)', scale.get('image_counts', {}), 'images/bin', scale.get('image_population'), scale.get('status', 'insufficient_evidence'))
    for cls, counts in scale.get('per_class_bin_counts', {}).items():
        plot(item, 'Native-area bins: '+cls, counts, 'objects/bin', sum(counts.values()))
    for key, raw in scale.get('native_and_input_statistics', {}).items():
        rows(item, {key+' median': raw.get('median')}, raw.get('unit', ''), raw.get('count'))
    rows(item, {'Model-input scale available': 1 if scale.get('model_input', {}).get('status') == 'measured' else None}, 'availability', scale.get('object_count'))
    for mid, key, title in (('UAV-12', 'local_complexity', 'Object-local background complexity'), ('UAV-14', 'object_local', 'Object clarity and visibility proxies')):
        local = value(mid).get(key, {})
        sampling = local.get('sampling', {})
        item = section(key, title, mid, local, 'observed sampled pixel proxies', sampling.get('objects_inspected'), sampling.get('object_coverage'), 'inspected valid annotated objects; individual distributions may exclude unavailable crops/rings')
        fields = ['background_ring.entropy_bits', 'background_ring.edge_density', 'background_ring.contrast_std'] if mid == 'UAV-12' else ['normalized_local_contrast', 'luminance_difference', 'object.laplacian_variance', 'object.high_frequency_residual_mad', 'object.local_contrast_rms', 'object.underexposed_fraction', 'object.saturated_fraction']
        for field in fields:
            histogram(item, field, local.get('distributions', {}).get(field, {}))
        for group, values in local.get('per_class', {}).items():
            for field in fields:
                raw = values.get(field, {})
                rows(item, {group+' / '+field+' median': raw.get('median')}, raw.get('unit', ''), raw.get('count'))
        if mid == 'UAV-12':
            for size, values in local.get('size_local_complexity_cross_tab', {}).items():
                plot(item, 'Size / sample-relative entropy: '+size, values, 'objects/bin', sum(values.values()))
    global_proxies = value('UAV-14').get('global_degradation_proxies', {})
    item = section('global_degradation', 'Whole-image degradation indicators', 'UAV-14', {'status': 'measured' if any(v.get('count') for v in global_proxies.values()) else 'insufficient_evidence'}, 'observed cached ingestion proxies', by_id.get('UAV-14', {}).get('evidence', {}).get('sample_size'), None, 'readable images with each proxy')
    for key, raw in global_proxies.items():
        rows(item, {key+' mean': raw.get('mean'), key+' minimum': raw.get('minimum'), key+' maximum': raw.get('maximum')}, raw.get('unit', ''), raw.get('count'))
    audit = value('UAV-15').get('structured_audit', {})
    item = section('annotation_audit', 'Independent annotation audit categories', 'UAV-15', audit, audit.get('evidence_type', 'independent audit unavailable'), audit.get('images_audited'), audit.get('image_coverage'), 'category-specific reference objects, reviewed annotations or audited images')
    for key, raw in audit.get('error_rates', {}).items():
        rows(item, {key: raw.get('count')}, 'errors or reviewed uncertain objects', raw.get('denominator'))
        rows(item, {key+' fraction': raw.get('fraction')}, 'fraction', raw.get('denominator'))
    for level in ('per_class','per_size'):
        for group, statistics in audit.get(level,{}).items():
            for key,raw in statistics.get('rates',{}).items():
                rows(item,{level+' / '+group+' / '+key:raw['fraction']},'fraction',raw['denominator'])
    plot(item, 'Audit error counts (see each category denominator in table)', audit.get('error_counts', {}), 'category count', 'category-specific; see table', audit.get('status', 'insufficient_evidence'))
    neg = value('UAV-15').get('negative_coverage', {})
    item = section('negative_coverage', 'Target-negative image coverage', 'UAV-15', neg, 'observed labels plus optional independent audit', neg.get('denominator'), None, 'valid annotated images; unknown labels shown separately')
    count_keys = ('explicit_negative_images', 'unknown_label_images', 'other_class_only_images', 'verified_negative_images', 'candidate_count')
    plot(item, 'Negative evidence categories (overlap possible)', {k: neg.get(k) for k in count_keys}, 'images or candidates', neg.get('image_population'), neg.get('status', 'insufficient_evidence'))
    rows(item, {k: neg.get(k) for k in count_keys}, 'images or candidates', neg.get('image_population'))
    for cls, raw in neg.get('per_class', {}).items():
        rows(item, {cls+' annotated target-free': raw['annotated_target_free_images']}, 'images', raw['denominator'])
    conditional = value('UAV-17').get('conditional_coverage', {})
    item = section('conditional_scenarios', 'Confirmed conditional scenario coverage', 'UAV-17', conditional, 'declared metadata intersected with observed annotations', None, None, 'dataset images / valid matching objects')
    for scenario in conditional.get('scenarios', []):
        plot(item, scenario['name'], {k: scenario[k] for k in ('matching_image_count', 'known_nonmatching_count', 'unknown_count')}, 'images', scenario['population'], scenario['status'])
        rows(item, {scenario['name']+' matching objects': scenario['matching_object_count']}, 'objects', 'valid annotated objects')
        rows(item, {scenario['name']+' missing metadata': scenario['missing_metadata_count']}, 'images', scenario['population'])
    independent = value('UAV-18').get('independence_analysis', {})
    near = independent.get('near_duplicate_candidates', {})
    item = section('collection_independence', 'Collection groups and duplicate candidates', 'UAV-18', independent, 'observed hashes and user-declared collection IDs', near.get('images_inspected'), near.get('signature_coverage'), 'images / declared groups; no effective sample-size estimate')
    rows(item, {k: independent.get(k) for k in ('unique_exact_hashes', 'exact_duplicate_images', 'exact_duplicate_groups', 'cross_split_exact_duplicate_groups')}, 'images or groups', independent.get('image_count'))
    rows(item, {k: near.get(k) for k in ('retrieved_candidate_pairs', 'cross_split_candidate_pairs', 'group_count')}, 'candidate pairs or groups', near.get('images_inspected'))
    for column, raw in independent.get('collection_groups', {}).items():
        rows(item, {column+' groups': raw.get('group_count'), column+' missing': raw.get('missing'), column+' cross-split groups': raw.get('cross_split_group_count')}, 'declared groups / images', independent.get('image_count'))
        plot(item, column+' per-class group coverage', raw.get('per_class_group_count', {}), 'declared groups', raw.get('group_count'), raw.get('status', 'insufficient_evidence'))
    return {'version': VERSION, 'sections': sections, 'notice': 'Descriptive evidence; no added scoring weights or empirical performance claim. Unavailable measurements remain unavailable.'}
