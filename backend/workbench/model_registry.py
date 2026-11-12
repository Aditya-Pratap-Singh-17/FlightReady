"""Architecture-checked Hub/local safetensors imports, without remote Python code."""
from pathlib import Path
import json
import re
import shutil
import uuid
from backend.workbench.datasets import file_sha

REVISION = '95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de'
# COCO names are authoritative here; synonyms need an explicit reviewed mapping.
COCO_CLASSES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')
FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')
ARCHITECTURES = {'yolos': ('YolosForObjectDetection', 'YolosImageProcessor'),
                 'detr': ('DetrForObjectDetection', 'DetrImageProcessor'),
                 'rt_detr': ('RTDetrForObjectDetection', 'RTDetrImageProcessor')}
CATALOG = {'yolos-tiny-coco': {'name': 'YOLOS Tiny (COCO)', 'architecture': 'YolosForObjectDetection',
    'adapter': 'transformers_yolos', 'task': 'object_detection', 'repository': 'hustvl/yolos-tiny', 'revision': REVISION,
    'license': 'apache-2.0', 'source_url': 'https://huggingface.co/hustvl/yolos-tiny',
    'download_size_bytes_approximate': 26_100_000, 'pretrained_classes': COCO_CLASSES,
    'operations': ['inference'], 'device_requirements': 'CPU supported; CUDA optional. Memory depends on image size.',
    'limitations': 'COCO-trained detector. No drone-target class. Training-set overlap with user data is unverified.'}}


def entry(key):
    if key not in CATALOG:
        raise ValueError('Choose a supported registry model; arbitrary remote repositories are not executable')
    return CATALOG[key]


def registry(store):
    acquired = store.list('acquired_model')
    return [{'id': key, **spec, 'cached_revision_available': all((store.root / 'hubcache' / ('models--'+spec['repository'].replace('/','--')) / 'snapshots' / spec['revision'] / name).is_file() for name in FILES),
             'local_available': any(m['payload'].get('registry_id') == key and verify_available(m['payload']) for m in acquired)}
            for key, spec in CATALOG.items()]


def verify_available(payload):
    root = Path(payload['directory'])
    try:
        safe_weight_files(payload['file_sha256'])
        return {'config.json','preprocessor_config.json'} <= set(payload['file_sha256']) and all((root / name).is_file() and file_sha(root / name) == digest for name, digest in payload['file_sha256'].items())
    except (ValueError,OSError): return False


def validate_snapshot(root, spec):
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    architecture = ARCHITECTURES.get(config.get('model_type'))
    if not architecture or config.get('architectures') != [architecture[0]] or config.get('auto_map'):
        raise ValueError('Unsupported model configuration. Choose a built-in YOLOS, DETR or RT-DETR detector saved with safetensors; custom remote code is not loaded.')
    classes = sorted(set(config.get('id2label', {}).values()) - {'N/A'})
    if not classes or any(not isinstance(c,str) or not c.strip() for c in classes) or set(classes) != set(spec['pretrained_classes']):
        raise ValueError('Model label schema does not match the pinned registry declaration')
    index=root/'model.safetensors.index.json'
    if index.is_file():
        if index.stat().st_size>1024**2: raise ValueError('Safetensors index exceeds the supported size')
        mapping=json.loads(index.read_text(encoding='utf-8')).get('weight_map',{})
        if not mapping or any(not re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors',name) or not (root/name).is_file() for name in mapping.values()):
            raise ValueError('Safetensors index must reference only supported local weight shards')
    return classes


def repository_id(value):
    value=value.strip().removeprefix('https://huggingface.co/').rstrip('/')
    if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*',value) or '..' in value:
        raise ValueError('Enter a Hugging Face model ID such as hustvl/yolos-tiny, or its model-page URL')
    return value


def safe_weight_files(filenames):
    if 'model.safetensors' in filenames: return ['model.safetensors']
    if 'model.safetensors.index.json' in filenames:
        shards=sorted(name for name in filenames if re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors',name))
        if shards: return ['model.safetensors.index.json',*shards]
    raise ValueError('This repository has no supported safetensors weights. Export it with safe_serialization=True or choose another detector.')


def inspect_repository(store, value, revision='main'):
    """A deliberate metadata-only Hub request resolves a branch to an immutable commit."""
    from huggingface_hub import HfApi, hf_hub_download
    repo=repository_id(value)
    try:
        info=HfApi(token=False).model_info(repo,revision=revision,files_metadata=True)
        if not re.fullmatch(r'[0-9a-f]{40}',info.sha or ''): raise ValueError('Hub did not return a full immutable revision')
        sizes={f.rfilename:f.size for f in info.siblings}
        for name in ('config.json','preprocessor_config.json'):
            if name not in sizes or sizes[name] is None or sizes[name]>1024**2:
                raise ValueError('The model needs config.json and preprocessor_config.json, each at most 1 MiB')
        weights=safe_weight_files(sizes)
        directory=Path(hf_hub_download(repo,'config.json',revision=info.sha,cache_dir=str(store.root/'hubcache'),token=False)).parent
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        architecture=ARCHITECTURES.get(config.get('model_type'))
        if not architecture or config.get('architectures') != [architecture[0]] or config.get('auto_map'):
            raise ValueError('This is not a supported built-in detector. Supported Transformers architectures: YOLOS, DETR and RT-DETR.')
        classes=sorted(set(config.get('id2label',{}).values())-{'N/A'})
        if not classes: raise ValueError('The model config does not declare its detection class names')
        card=info.card_data.to_dict() if hasattr(info.card_data,'to_dict') else info.card_data or {}
        license=card.get('license') or 'not declared; review the repository terms'
        files=['config.json','preprocessor_config.json',*weights]
        size=sum(sizes.get(f) or 0 for f in files)
        if size>8*1024**3: raise ValueError('This model exceeds the supported 8 GiB download limit; choose a smaller detector')
    except ValueError: raise
    except Exception as exc:
        raise ValueError('Could not inspect this Hugging Face model. Check the repository ID, revision and network access. Private/gated models require a local exported folder.') from exc
    return store.create('model_source',{'name':repo, 'repository':repo,'revision':info.sha,'architecture':architecture[0],
        'adapter':'transformers_detection','model_type':config['model_type'],'task':'object_detection','license':license,
        'source_url':'https://huggingface.co/'+repo,'pretrained_classes':classes,'download_size_bytes_approximate':size,
        'files':files,'operations':['inference'],'device_requirements':'CPU supported; CUDA optional.',
        'limitations':'Pretrained class names do not establish task suitability. Model development-data overlap is unverified.'})


def import_local(store, path, license_accepted):
    if not license_accepted: raise ValueError('Review and acknowledge the local model license before importing')
    root=Path(path).expanduser().resolve()
    if not root.is_dir(): raise ValueError('Choose a local save_pretrained folder containing config.json, preprocessor_config.json and safetensors weights')
    names={p.name for p in root.iterdir() if p.is_file()}
    files=['config.json','preprocessor_config.json',*safe_weight_files(names)]
    if any(not (root/name).is_file() for name in files): raise ValueError('Local folder needs config.json and preprocessor_config.json')
    if any((root/name).stat().st_size>1024**2 for name in files if name.endswith('.json')): raise ValueError('Model JSON exceeds the 1 MiB limit')
    if sum((root/name).stat().st_size for name in files)>8*1024**3: raise ValueError('Model exceeds the supported 8 GiB limit')
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    classes=sorted(set(config.get('id2label',{}).values())-{'N/A'})
    validate_snapshot(root,{'pretrained_classes':classes})
    destination=store.directory('importedmodels',uuid.uuid4().hex)
    for name in files: shutil.copyfile(root/name,destination/name)
    return store.create('acquired_model',{'name':root.name,'repository':'local export','revision':'sha256-identified local files',
        'architecture':config['architectures'][0],'model_type':config['model_type'],'adapter':'transformers_detection',
        'classes':classes,'directory':str(destination),'file_sha256':{name:file_sha(destination/name) for name in files},
        'license':'user-reviewed local model terms','license_accepted':True,'operations':['inference'],'source':'local_folder'})


def acquire(store, job_id, config):
    key = config.get('registry_id')
    spec = store.get(config['source_id'],'model_source')['payload'] if config.get('source_id') else entry(key)
    files=spec.get('files',list(FILES))
    if not config.get('local_only') and not config.get('download_consent') or not config.get('license_accepted'):
        raise ValueError('Explicit download consent and license acceptance are required')
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError('Install requirements/requirements-inference.txt before acquiring models') from exc
    store.update(job_id, stage='Loading pinned local cache' if config.get('local_only') else 'Downloading allowlisted JSON and safetensors at pinned revision')
    try:
        directory = Path(snapshot_download(repo_id=spec['repository'], revision=spec['revision'],
            allow_patterns=files, cache_dir=str(store.root / 'hubcache'), token=False, local_files_only=config.get('local_only',False)))
    except Exception as exc:
        raise ValueError('Pinned model is absent from the local cache. Review and consent to a download.' if config.get('local_only') else
                         'Model download failed. Check network access and available disk space, then retry. No model was activated.') from exc
    if any(not (directory / name).is_file() for name in files):
        raise ValueError('The pinned revision is missing a required safe model file')
    classes = validate_snapshot(directory, spec)
    payload = {**spec, 'registry_id': key, 'directory': str(directory), 'classes': classes,
               'file_sha256': {name: file_sha(directory / name) for name in files},
               'download_consent': config.get('download_consent',False), 'local_only':config.get('local_only',False), 'license_accepted': True, 'job_id': job_id}
    record = store.create('acquired_model', payload)
    return {'acquired_model_id': record['id']}
