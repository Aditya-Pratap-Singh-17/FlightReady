"""Adapters around the existing parsers; persist parsed evidence without pickle."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import os
import shutil
import stat
import uuid
import zipfile
import pandas as pd
from PIL import Image
from backend.ingestion import Dataset, IngestionError
from backend.large_ingestion import load_dataset_stream
from backend.image_ingestion import validate_members, IMAGE_SUFFIXES
from metrics.dataProfiler import Dataprofiler


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


class DirectoryArchive:
    """Read-only archive interface; registering a local directory copies no images."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = {}
        total = 0
        for path in self.root.rglob('*'):
            if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                raise IngestionError('Directory symlinks/junctions are not supported')
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.root):
                raise IngestionError('Unsafe directory member')
            name = path.relative_to(self.root).as_posix()
            self.paths[name] = path
            total += path.stat().st_size
            if len(self.paths) > int(os.getenv('MAX_ZIP_FILES', '50000')) or total > int(os.getenv('MAX_ZIP_MB', '4096')) * 1024**2:
                raise IngestionError('Directory exceeds configured file/expanded-size limits')

    def infolist(self):
        result = []
        for name, path in self.paths.items():
            info = zipfile.ZipInfo(name)
            info.file_size = info.compress_size = path.stat().st_size
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            result.append(info)
        return result

    def open(self, name):
        path = self.paths[name].resolve()
        if not path.is_relative_to(self.root):
            raise IngestionError('Directory member changed to an unsafe link')
        return path.open('rb')

    def read(self, name):
        with self.open(name) as source:
            return source.read()


@contextmanager
def image_source(payload):
    if payload['source_kind']=='virtual_combined':
        from contextlib import ExitStack
        class CombinedArchive:
            def __init__(self):
                self.stack=ExitStack();self.archives={}
            def read(self,name):
                if name not in payload['virtual_sources']:
                    raise ValueError('Image is not in the combined training manifest')
                entry=payload['virtual_sources'][name]
                key=entry['source']['source_path']
                if key not in self.archives:
                    self.archives[key]=self.stack.enter_context(image_source(entry['source']))
                return self.archives[key].read(entry['image'])
            def infolist(self):
                import zipfile
                return [zipfile.ZipInfo(name) for name in payload['virtual_sources']]
        archive=CombinedArchive()
        try:
            yield archive
        finally:
            archive.stack.close()
    elif payload['source_kind'] == 'directory':
        yield DirectoryArchive(payload['source_path'])
    else:
        from backend.dataset_archives import open_archive
        with open_archive(payload['source_path'], payload['source_name']) as archive:
            yield archive


def training_eligibility(ds):
    reasons = []
    if ds.modality != 'image' or ds.profile.get('dataset_kind') != 'object_detection':
        reasons.append('Object detection requires labeled images; telemetry, tables, classification and unlabeled images remain eligible for readiness analysis only.')
    elif ds.profile.get('annotation_format') not in ('yolo', 'yolo_yaml', 'coco', 'visdrone_det', 'pascal_voc'):
        reasons.append('Detector execution requires supported YOLO, COCO, native VisDrone or VOC annotations.')
    if ds.profile.get('manifest_errors'):
        reasons.append('Repair invalid annotation-manifest references before detector training.')
    if ds.modality == 'image':
        for column in ('readable', 'annotation_present', 'annotation_valid'):
            if column not in ds.frame or not ds.frame[column].fillna(False).all():
                reasons.append(f'All images require {column.replace("_", " ")}.')
        if not ds.profile.get('class_counts'):
            reasons.append('No compatible object classes are available.')
    return {'eligible': not reasons, 'reasons': reasons}


def persist(store, ds, source_path, source_kind, source_name, name=None, provenance=None, id=None):
    id = id or uuid.uuid4().hex
    directory = store.directory('datasets', id)
    ds.frame.to_json(directory / 'frame.json', orient='table', index=False)
    details = {'objects': ds.objects, 'issues': ds.issues, 'image_hashes': ds.image_hashes,
               'near_pairs': ds.near_pairs, 'near_distance': ds.near_distance, 'metadata': ds.metadata}
    (directory / 'evidence.json').write_text(json.dumps(details, allow_nan=False), encoding='utf-8')
    disk_path = None
    if ds.disk_path:
        disk_path = str(directory / 'table.duckdb')
        shutil.copy2(ds.disk_path, disk_path)
    payload = {'name': name or ds.name, 'sha256': ds.sha256, 'version': ds.sha256[:12], 'modality': ds.modality,
               'profile': ds.profile, 'source_path': str(source_path) if source_path is not None else None, 'source_name': source_name,
               'source_kind': source_kind, 'directory': str(directory), 'disk_path': disk_path,
               'training': training_eligibility(ds), 'provenance': provenance or {}}
    return store.create('dataset', payload, id)


def load(store, id):
    payload = store.get(id, 'dataset')['payload']
    directory = Path(payload['directory'])
    details = json.loads((directory / 'evidence.json').read_text(encoding='utf-8'))
    frame = pd.read_json(io.StringIO((directory / 'frame.json').read_text(encoding='utf-8')), orient='table')
    return Dataset(payload['name'], payload['sha256'], payload['modality'], frame, payload['profile'],
                   **details, disk_path=payload.get('disk_path'), source_path=payload['source_path'],
                   image_reader=lambda: image_source(payload))


def import_file(store, name, stream, metadata=None):
    id = uuid.uuid4().hex
    directory = store.directory('datasets', id)
    source = directory / ('source' + Path(name).suffix.lower())
    total = 0
    try:
        with source.open('xb') as output:
            while block := stream.read(1024**2):
                total += len(block)
                if total > int(os.getenv('MAX_UPLOAD_MB', '2048')) * 1024**2:
                    raise IngestionError('Upload exceeds server limit')
                output.write(block)
        with source.open('rb') as incoming:
            ds = load_dataset_stream(name, incoming, metadata or {})
        try:
            if ds.profile.get('needs_selection'):
                raise IngestionError('Ambiguous dataset layout. Supply explicit parser metadata or import one labeled dataset at a time. Choices: ' + json.dumps(ds.profile.get('choices', [])))
            effective_name=ds.name
            image_archive=ds.modality=='image' and Path(effective_name).suffix.lower() not in IMAGE_SUFFIXES
            actual_source=source
            source_name=effective_name
            nested=bool(ds.profile.get('source_archive',{}).get('member') or ds.profile.get('compressed_source'))
            if ds.modality=='image' and ds.source_path and (nested or Path(ds.source_path).suffix!=source.suffix):
                actual_source=directory/('images-source'+Path(ds.source_path).suffix)
                shutil.copy2(ds.source_path,actual_source)
            return persist(store, ds, actual_source, 'archive' if image_archive else 'file', source_name, id=id)
        finally:
            ds.close()
    except Exception:
        shutil.rmtree(directory)
        raise


def register_directory(store, path, name='', metadata=None):
    from backend.universal_images import image_choices, load_universal_images
    path = Path(path).resolve()
    if not path.is_dir():
        raise IngestionError('Local dataset directory does not exist')
    archive = DirectoryArchive(path)
    names, _ = validate_members(archive)
    metadata = metadata or {}
    choices = image_choices(archive, names, metadata)
    if len(choices) != 1 and not metadata.get('annotation_format'):
        raise IngestionError('Directory requires explicit annotation_format/parser selection: ' + json.dumps(choices))
    options = {**(choices[0]['options'] if choices else {}), **metadata}
    # Hash all source files; external edits are detected before jobs run.
    digest = hashlib.sha256()
    for member in sorted(names):
        digest.update(member.encode())
        digest.update(bytes.fromhex(file_sha(archive.paths[member])))
    if options.get('legacy_layout'):
        from backend.image_ingestion import load_image_archive
        ds = load_image_archive(name or path.name, archive, digest.hexdigest(), options)
    else:
        ds = load_universal_images(name or path.name, archive, names, digest.hexdigest(), options)
    return persist(store, ds, path, 'directory', path.name, name)


def image_bytes(store, dataset_id, image):
    record = store.get(dataset_id, 'dataset')['payload']
    ds = load(store, dataset_id)
    if image not in ds.image_hashes:
        raise ValueError('Image is not in the dataset manifest')
    if record['source_kind'] == 'file':
        raw = Path(record['source_path']).read_bytes()
    else:
        with image_source(record) as archive:
            raw = archive.read(image)
    if hashlib.sha256(raw).hexdigest() != ds.image_hashes[image]:
        raise ValueError('Dataset image changed since import; import a new dataset version')
    return raw


def subset(ds, images):
    """Recompute train-only profiles so whole-dataset metrics never masquerade as training evidence."""
    import copy
    selected = copy.copy(ds)
    selected.frame = ds.frame[ds.frame.image.isin(images)].copy().reset_index(drop=True)
    selected.objects = [o for o in ds.objects if o['image'] in images]
    selected.image_hashes = {k: v for k, v in ds.image_hashes.items() if k in images}
    selected.near_pairs = [(a, b) for a, b in ds.near_pairs if a in images and b in images]
    selected.issues = [i for i in ds.issues if any(str(i.get('reference', '')).startswith(im) for im in images)]
    selected.sha256 = hashlib.sha256(json.dumps({'parent_sha256':ds.sha256,'images':sorted(selected.image_hashes.items()),
        'annotations':selected.objects,'rows':selected.frame.to_json(orient='records')},sort_keys=True,allow_nan=False).encode()).hexdigest()
    selected.metadata=copy.deepcopy(ds.metadata)
    audit=selected.metadata.get('annotation_audit')
    if isinstance(audit,dict):
        selected.metadata['annotation_audit_source_scope']={'dataset_sha256':ds.sha256,
            'audit_sha256':hashlib.sha256(json.dumps(audit,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'selection':'Only references for selected images; original records unchanged'}
        if isinstance(audit.get('images'),list) and audit.get('dataset_sha256') in (None,ds.sha256):
            audit['images']=[entry for entry in audit['images'] if isinstance(entry,dict) and entry.get('image') in selected.image_hashes]
            audit['parent_dataset_sha256']=ds.sha256
            audit['dataset_sha256']=selected.sha256
            # Whole-population count summaries cannot be reused for this subset.
            audit.pop('reviewed_objects',None);audit.pop('incorrect_objects',None)
        elif 'images' not in audit:
            selected.metadata.pop('annotation_audit')
            selected.metadata['annotation_audit_unavailable_reason']='Count-only audit has no image-level scope for this selected population.'
    if isinstance(selected.metadata.get('hard_negative_candidates'),list):
        selected.metadata['hard_negative_candidates']=[entry for entry in selected.metadata['hard_negative_candidates']
            if isinstance(entry,dict) and entry.get('image') in selected.image_hashes]
    counts = Counter(o['class'] for o in selected.objects)
    selected.profile = {**ds.profile, **Dataprofiler(selected.frame).profile().to_dict(), 'image_count': len(images),
                        'object_count': len(selected.objects), 'class_counts': {c: counts[c] for c in ds.profile.get('class_counts', {})},
                        'issues': selected.issues, 'assessment_scope': 'training manifest only'}
    return selected
