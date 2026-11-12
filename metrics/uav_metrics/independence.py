"""Bounded duplicate candidates and declared collection evidence, never ESS."""
from collections import Counter, defaultdict
import pandas as pd
from metrics.uav_metrics.research import VERSION, bounded, distribution, object_records


def collection_evidence(ds, g):
    frame = ds.frame
    split_column = g.split_column or 'split'
    rows = frame.to_dict('records')
    split = {r['image']: str(r[split_column]) for r in rows if split_column in r and pd.notna(r[split_column])} if 'image' in frame else {}
    exact = defaultdict(list)
    for name, digest in ds.image_hashes.items():
        exact[digest].append(name)
    exact_groups = [v for v in exact.values() if len(v) > 1]
    cross_exact = [v for v in exact_groups if len({split[n] for n in v if n in split}) > 1]
    columns = g.uav_parameters.get('collection_group_columns', [g.group_column] if g.group_column else ['flight_id', 'session_id', 'recording_id', 'video_id', 'camera_id', 'location_id', 'collection_time_group'])
    if not isinstance(columns, list) or len(columns) > 12 or any(not isinstance(c, str) for c in columns):
        raise ValueError('collection_group_columns must contain at most 12 column names')
    objects = object_records(ds, g) if ds.modality == 'image' else []
    per_image = defaultdict(set)
    for obj in objects:
        per_image[obj['image']].add(obj['class'])
    groups = {}
    for column in dict.fromkeys(columns):
        if column not in frame:
            groups[column] = {'status': 'insufficient_evidence', 'missing': len(frame)}
            continue
        values = defaultdict(list)
        for row in rows:
            value = row.get(column)
            if value is not None and pd.notna(value) and str(value).strip():
                values[str(value)].append(row.get('image', str(len(values))))
        overlaps = [key for key, names in values.items() if len({split[n] for n in names if n in split}) > 1]
        per_class = defaultdict(set)
        for key, names in values.items():
            for name in names:
                for cls in per_image[name]:
                    per_class[cls].add(key)
        groups[column] = {'status': 'measured' if values else 'insufficient_evidence', 'evidence_type': 'user_declared_group_ids',
                          'group_count': len(values), 'missing': len(frame)-sum(map(len, values.values())),
                          'size_distribution': distribution(list(map(len, values.values())), 'images/group'),
                          'cross_split_group_count': len(overlaps) if split else None, 'cross_split_examples': overlaps[:20],
                          'per_class_group_count': {c: len(s) for c, s in per_class.items()}}
    config = g.uav_parameters.get('near_duplicate_analysis', {})
    if not isinstance(config, dict) or set(config)-{'max_images', 'max_tree_visits', 'distance', 'max_results', 'max_candidate_pairs'}:
        raise ValueError('Unknown near_duplicate_analysis configuration')
    maximum = bounded(config.get('max_images', 20000), 1, 50000, 'max_images')
    budget = bounded(config.get('max_tree_visits', 200000), 1, 1000000, 'max_tree_visits')
    distance = bounded(config.get('distance', 4), 0, 16, 'distance')
    max_results = bounded(config.get('max_results', 128), 1, 1024, 'max_results')
    pair_budget = bounded(config.get('max_candidate_pairs',100000),1,500000,'max_candidate_pairs')
    parent = {name: name for name in ds.image_hashes}
    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        parent[root(a)] = root(b)
    for names in exact.values():
        for name in names[1:]:
            union(names[0], name)
    from backend.workbench.splits import HashTree
    tree = HashTree()
    visited = inspected = pairs = cross_pairs = 0
    complete = True
    examples = []
    readable = sorted((r for r in rows if r.get('dhash64') and r.get('image') in parent), key=lambda r: r['image'])
    for row in readable[:maximum]:
        signature = int(row['dhash64'], 16)
        matches, visits, full_query = tree.find_bounded(signature, distance, max_visits=max(0, budget-visited), max_results=min(max_results,pair_budget-pairs))
        visited += visits
        inspected += 1
        complete &= full_query
        for other in matches:
            union(row['image'], other)
            pairs += 1
            cross = row['image'] in split and other in split and split[row['image']] != split[other]
            cross_pairs += cross
            if len(examples) < 20:
                examples.append({'images': [row['image'], other], 'cross_split': cross})
        tree.add(signature, row['image'])
        if visited >= budget or pairs>=pair_budget:
            complete = False
            break
    clusters = defaultdict(list)
    for name in parent:
        clusters[root(name)].append(name)
    candidate_groups = [names for names in clusters.values() if len(names) > 1]
    temporal = [key for key in groups if any(s in key.lower() for s in ('flight', 'session', 'recording', 'video', 'sequence')) and groups[key].get('group_count')]
    return {'version': VERSION, 'status': 'measured', 'population_unit':'images' if ds.modality=='image' else 'records',
            'record_count':len(frame),'image_count': len(frame) if ds.modality=='image' else None, 'annotated_objects': len(objects) if ds.modality=='image' else None,
            'unique_exact_hashes': len(exact) if ds.image_hashes else None, 'hash_coverage': len(ds.image_hashes)/len(frame) if len(frame) else None,
            'exact_duplicate_images': sum(len(v)-1 for v in exact_groups) if ds.image_hashes else None, 'exact_duplicate_groups': len(exact_groups) if ds.image_hashes else None,
            'cross_split_exact_duplicate_groups': len(cross_exact) if split and ds.image_hashes else None,
            'collection_groups': groups, 'independence_status': 'declared_collection_groups_available_not_statistically_validated' if temporal else 'uncertain; flight/session metadata absent',
            'near_duplicate_candidates': {'status': 'measured' if readable else 'insufficient_evidence', 'method': 'cached 64-bit dHash; bounded BK-tree Hamming retrieval; connected candidate components include exact duplicates',
                'distance': distance, 'tree_visits': visited, 'visit_budget': budget, 'images_inspected': inspected, 'signature_population': len(readable),
                'image_population': len(frame), 'signature_coverage': len(readable)/len(frame) if len(frame) else None,
                'complete_retrieval': complete and inspected == len(frame), 'retrieved_candidate_pairs': pairs if readable else None, 'cross_split_candidate_pairs': cross_pairs if split and readable else None,
                'group_count': len(candidate_groups) if readable else None, 'group_sizes': distribution([len(v) for v in candidate_groups], 'images/group'),
                'group_examples': [v[:20] for v in candidate_groups[:20]], 'pair_examples': examples,
                'configuration': {'max_images': maximum, 'max_tree_visits': budget, 'max_results': max_results, 'max_candidate_pairs':pair_budget}},
            'sample_sufficiency': 'Requires explicit task-specific count requirements; no universal minimum.',
            'limitations': ['Distinct filenames or exact hashes are not independent collections.', 'Candidate near-duplicate components are transitive, not guaranteed pairwise similarity.',
                           'dHash can collide on uniform/different images; candidates require review.', 'Incomplete retrieval does not prove absence of duplicates or leakage.',
                           'Group counts are not statistically established effective sample sizes.', 'Use existing strict split safeguards before model training.']}
