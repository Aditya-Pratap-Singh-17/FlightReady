"""Canonical adapter reusing existing train-only research interventions."""
from pathlib import Path
import csv
import io
import json
import tempfile
import uuid
import zipfile
from backend.workbench.datasets import load, image_source, import_file
from backend.workbench.splits import get_split
from backend.research.variants import create_variant, VariantConfig


def variation(store, request):
    source = store.get(request.dataset_id, 'dataset')['payload']
    if not source['training']['eligible']:
        raise ValueError('A supported detection dataset is required')
    ds = load(store, request.dataset_id)
    split = get_split(store, request.split_id, request.dataset_id)
    classes = list(ds.profile['class_counts'])
    directory = store.directory('variations', uuid.uuid4().hex)
    canonical = directory / 'parent.zip'
    mapping = {}
    metadata = []
    frame = ds.frame.set_index('image').to_dict('index')
    # Copy only declared metadata columns; pixel/annotation evidence is recomputed.
    metadata_columns=set()
    with image_source(source) as archive:
        metadata_path=source['profile'].get('archive_root','')+'metadata.csv'
        if metadata_path in {info.filename for info in archive.infolist()}:
            with archive.open(metadata_path) as incoming:
                text=incoming.read(8*1024**2+1)
            if len(text)>8*1024**2:raise ValueError('Metadata exceeds limit')
            from backend.ingestion import decode_csv
            metadata_columns=set(next(csv.reader(io.StringIO(decode_csv(text)[0])),[]))
    if split['group_column']:
        metadata_columns.add(split['group_column'])
    with image_source(source) as archive, zipfile.ZipFile(canonical, 'x', compression=zipfile.ZIP_STORED) as output:
        output.writestr('classes.txt', '\n'.join(classes)+'\n')
        for split_name, names in split['images'].items():
            for i, name in enumerate(names):
                canonical_name = f'images/{split_name}/{i:08d}{Path(name).suffix.lower()}'
                mapping[canonical_name] = name
                raw = archive.read(name)
                import hashlib
                if hashlib.sha256(raw).hexdigest() != ds.image_hashes[name]:
                    raise ValueError('Source changed since import')
                output.writestr(canonical_name, raw)
                labels = [' '.join([str(classes.index(o['class']))]+[str(v) for v in o['bbox']]) for o in ds.objects if o['image']==name]
                output.writestr(canonical_name.replace('images/', 'labels/', 1).rsplit('.',1)[0]+'.txt', '\n'.join(labels)+('\n' if labels else ''))
                metadata.append({**{k:v for k,v in frame[name].items() if k in metadata_columns and k not in ('image','split')}, 'image': canonical_name, 'split': split_name})
        import pandas as pd
        output.writestr('metadata.csv', pd.DataFrame(metadata).to_csv(index=False))
    intervention = request.intervention
    parameters = {}
    target = None
    if intervention == 'blur':
        intervention = 'image_degradation'
        parameters = {'blur_radius': 10*request.severity}
    elif intervention in ('noise', 'exposure'):
        parameters = {'noise_sigma': 64*request.severity} if intervention=='noise' else {'brightness_factor': 1-.9*request.severity}
        intervention = 'image_degradation'
    elif intervention == 'wrong_class':
        intervention = 'annotation_noise'
        parameters = {'probability': request.severity}
    elif intervention == 'small_object_reduction':
        parameters = {'maximum_native_area': request.maximum_native_area, 'retain_fraction': 1-request.severity, 'classes': request.classes or classes}
    elif intervention == 'negative_coverage':
        if not request.hard_negative_column:
            raise ValueError('Hard-negative coverage selection requires an explicit candidate metadata column; candidate difficulty remains unverified')
        parameters = {'column': request.hard_negative_column, 'retain_fraction': 1-request.severity, 'classes': request.classes or classes}
    elif intervention == 'resolution':
        parameters = {'scale': max(.05, 1-request.severity)}
    elif intervention == 'size':
        intervention = 'control'
        target = max(1, round(len(split['images']['train'])*(1-request.severity)))
    elif intervention == 'imbalance':
        if not request.classes or not set(request.classes).issubset(classes):
            raise ValueError('Select retained class names from the dataset schema')
        intervention = 'class_representation'
        parameters = {'classes': request.classes,'retain_other_fraction':1-request.severity}
    elif intervention in ('label_omission','box_perturbation'):
        parameters = {'severity': request.severity}
    elif intervention == 'environment':
        if not request.environment_column or not request.allowed_values:
            raise ValueError('Environmental diversity selection requires a metadata column and retained values')
        parameters={'column':request.environment_column,'allowed_values':request.allowed_values}
    config = VariantConfig(seed=request.seed, intervention=intervention, parameters=parameters, target_train_images=target,
                           group_column=split['group_column'], acknowledge_missing_groups=not split['group_column'])
    try:
        manifest = create_variant(canonical, directory/'variant', config)
        with (directory/'variant'/'dataset.zip').open('rb') as incoming:
            record = import_file(store, 'controlled.zip', incoming)
        provenance = {'artificial': True, 'parent_dataset_id': request.dataset_id, 'parent_sha256': ds.sha256,
                      'parent_split_id': request.split_id, 'intervention': request.model_dump(), 'manifest': manifest,
                      'source_mapping': mapping, 'note': 'Validation/test image and annotation bytes are preserved. Reassess the new training subset before training.'}
        return store.update(record['id'], name=source['name']+' · '+request.intervention, provenance=provenance)
    finally:
        canonical.unlink(missing_ok=True)
