"""Deterministic split manifests, grouping exact/likely duplicate images together."""
from collections import defaultdict
import hashlib
import io
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from backend.workbench.datasets import load, image_source, file_sha


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, name):
        node = self.root
        if node is None:
            self.root = [value, [name], {}]
            return
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(name)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [name], {}]
                return
            node = node[2][distance]

    def find(self, value, threshold):
        queue = [self.root] if self.root else []
        result = []
        while queue:
            node = queue.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                result.extend(node[1])
            queue.extend(child for edge, child in node[2].items() if distance-threshold <= edge <= distance+threshold)
        return result

    def find_bounded(self, value, threshold, *, max_visits, max_results=128):
        """Read-only, explicitly budgeted BK-tree query for descriptive analysis.

        Split safeguards continue using exact find(); budget exhaustion here is
        incomplete evidence, never proof that two images are unrelated.
        """
        queue = [self.root] if self.root else []
        result, visits = [], 0
        while queue and visits < max_visits:
            node = queue.pop()
            visits += 1
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                remaining = max_results-len(result)
                result.extend(node[1][:remaining])
                if len(node[1]) > remaining:
                    return result, visits, False
            queue.extend(child for edge, child in node[2].items() if distance-threshold <= edge <= distance+threshold)
        return result, visits, not queue


def perceptual(raw):
    with Image.open(io.BytesIO(raw)) as image:
        pixels = np.asarray(image.convert('L').resize((9, 8)), dtype=int)
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).flat:
        value = value * 2 + int(bit)
    return value


def signatures(store, dataset_id):
    payload = store.get(dataset_id, 'dataset')['payload']
    if payload.get('perceptual_signatures'):
        return payload['perceptual_signatures']
    ds = load(store, dataset_id)
    output = {}
    with image_source(payload) as archive:
        for image, expected in ds.image_hashes.items():
            raw = archive.read(image)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Dataset changed since import; import a new version')
            output[image] = perceptual(raw)
    store.update(dataset_id,perceptual_signatures=output)
    return output


def assert_development_disjoint(a,b):
    """Combining sources must not move either source's training data into validation."""
    for left,right in ((a,b),(b,a)):
        validation=right['images']['val']
        hashes={right['hashes'][name] for name in validation}
        tree=HashTree()
        for name in validation:tree.add(right['perceptual'][name],name)
        threshold=max(left['config']['near_distance'],right['config']['near_distance'])
        for name in left['images']['train']:
            if left['hashes'][name] in hashes or tree.find(left['perceptual'][name],threshold):
                raise ValueError('Combined sources overlap training with validation by exact/likely duplicate images')
        if left.get('group_column') and left.get('group_column')==right.get('group_column'):
            if {left['groups'][name] for name in left['images']['train']} & {right['groups'][name] for name in validation}:
                raise ValueError('Combined sources overlap training/validation recording groups')


def create_split(store, dataset_id, config):
    ds = load(store, dataset_id)
    if ds.profile.get('source_split') in ('val', 'test'):
        raise ValueError('This is a validation-only/test-only dataset. It can be evaluated, but training requires a separate labeled training dataset and independent evaluation partitions.')
    if not store.get(dataset_id, 'dataset')['payload']['training']['eligible']:
        raise ValueError('Only supported labeled image datasets can have detector splits')
    if config.train_fraction + config.val_fraction >= 1:
        raise ValueError('Train and validation fractions must leave a positive test fraction')
    names = sorted(ds.image_hashes)
    parent = {name: name for name in names}
    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name
    def union(a, b):
        parent[root(a)] = root(b)
    exact = {}
    for name in names:
        digest = ds.image_hashes[name]
        if digest in exact:
            union(name, exact[digest])
        exact[digest] = name
    group_column = config.group_column
    if not group_column:
        group_column = next((c for c in ('flight_id', 'session_id', 'recording_id', 'video_id') if c in ds.frame), None)
    group_values = {}
    if group_column:
        if group_column not in ds.frame or ds.frame[group_column].isna().any():
            raise ValueError('Selected group column must have a value for every image')
        groups = {}
        for row in ds.frame.to_dict('records'):
            value = str(row[group_column])
            group_values[row['image']] = value
            if value in groups:
                union(row['image'], groups[value])
            groups[value] = row['image']
    elif not config.acknowledge_missing_groups:
        raise ValueError('Provide flight/session groups or acknowledge that recording independence is unverified')
    sig = signatures(store, dataset_id)
    tree = HashTree()
    near_pairs = 0
    if config.near_distance:
        for name in names:
            for other in tree.find(sig[name], config.near_distance):
                union(name, other)
                near_pairs += 1
            tree.add(sig[name], name)
    clusters = defaultdict(list)
    for name in names:
        clusters[root(name)].append(name)
    declared = {}
    aliases = {'training': 'train', 'validation': 'val', 'valid': 'val', 'testing': 'test'}
    for row in ds.frame.to_dict('records'):
        value = row.get('split')
        if value is None or pd.isna(value):
            parts = row['image'].split('/')
            value = next((p for p in parts[:-1] if p in ('train', 'val', 'valid', 'validation', 'test')), '')
        value = aliases.get(str(value).lower(), str(value).lower())
        if value in ('train', 'val', 'test'):
            declared[row['image']] = value
    split = {'train': [], 'val': [], 'test': []}
    if config.purpose == 'training_only':
        if any(value != 'train' for value in declared.values()):
            raise ValueError('Training-only manifests cannot repurpose declared validation or test images')
        split['train'] = names
    elif declared:
        if len(declared) != len(names):
            raise ValueError('Partial split assignments are ambiguous; supply a complete manifest')
        for cluster in clusters.values():
            if len({declared[name] for name in cluster}) != 1:
                raise ValueError('Exact/likely duplicates or recording groups overlap declared splits')
        for name in names:
            split[declared[name]].append(name)
        if not split['train']:
            raise ValueError('This validation-only/test-only dataset has no training partition. Use inference-only evaluation or supply a labeled training dataset.')
    else:
        units = sorted(clusters.values())
        random.Random(config.seed).shuffle(units)
        if len(units) < 3:
            raise ValueError('At least three independent image/recording groups are required for train/validation/test')
        target = {'train': len(names)*config.train_fraction, 'val': len(names)*config.val_fraction,
                  'test': len(names)*(1-config.train_fraction-config.val_fraction)}
        for i, cluster in enumerate(units):
            key = ('train', 'val', 'test')[i] if i < 3 else max(target, key=lambda k: target[k]-len(split[k]))
            split[key].extend(cluster)
    if config.purpose != 'training_only' and any(not images for images in split.values()):
        raise ValueError('Each split must contain images; adjust grouping or use a complete split manifest')
    for images in split.values():
        images.sort()
    payload = {'dataset_id': dataset_id, 'dataset_sha256': ds.sha256, 'config': config.model_dump(), 'images': split,
               'hashes': ds.image_hashes, 'perceptual': sig, 'groups': group_values, 'group_column': group_column,
               'near_duplicate_pairs': near_pairs, 'independence': 'grouped' if group_column else 'recording independence unverified'}
    payload['checksum'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return store.create('split', payload)


def get_split(store, id, dataset_id):
    payload = store.get(id, 'split')['payload']
    if payload['dataset_id'] != dataset_id or payload['dataset_sha256'] != store.get(dataset_id, 'dataset')['payload']['sha256']:
        raise ValueError('Split does not match dataset snapshot')
    return payload


def assert_independent(training_manifests, test, test_names, acknowledge=False):
    """Both training and validation are development data, excluded from final testing."""
    test_hashes = {test['hashes'][name] for name in test_names}
    tree = HashTree()
    for name in test_names:
        tree.add(test['perceptual'][name], name)
    for development in training_manifests:
        names = development['images']['train'] + development['images']['val']
        if {development['hashes'][name] for name in names} & test_hashes:
            raise ValueError('Final test images overlap training/validation by SHA256')
        threshold = max(development['config']['near_distance'], test.get('config', {}).get('near_distance', 4))
        if threshold and any(tree.find(development['perceptual'][name], threshold) for name in names):
            raise ValueError('Likely near-duplicate final test/development images detected')
        if development.get('group_column') and development.get('group_column') == test.get('group_column'):
            if {development['groups'][name] for name in names} & {test['groups'][name] for name in test_names}:
                raise ValueError('Final test recording groups overlap training/validation')
        elif not acknowledge:
            raise ValueError('Cross-dataset recording independence cannot be established; explicitly acknowledge missing comparable groups')
