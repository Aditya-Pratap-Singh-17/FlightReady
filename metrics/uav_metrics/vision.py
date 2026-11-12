"""Opt-in, local-only ONNX image encoders/classifiers.

Only a server-owned UAV_VISION_CONFIG file can enable inference. Upload metadata
cannot select models, files or code. No model download or network APIs exist in
this module. Models must output one finite N x D tensor from one NCHW RGB tensor.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from collections import defaultdict
import hashlib
import io
import json
import os
import numpy as np
from PIL import Image
from metrics.uav_metrics.assessment import summary, categories, valid_objects

_job=ContextVar('uav_visual_job',default=None)

@contextmanager
def vision_session(progress=lambda stage:None):
    token=_job.set({'progress':progress,'results':{}})
    try:yield
    finally:_job.reset(token)

def configuration():
    path=os.getenv('UAV_VISION_CONFIG')
    if not path:return None
    p=Path(path)
    if not p.is_file():return None
    if p.stat().st_size>65536:raise ValueError('Vision configuration is too large')
    config=json.loads(p.read_text(encoding='utf-8'))
    if config.get('enabled') is not True:return None
    return config

def fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024**2):digest.update(block)
    return digest.hexdigest()

def _source_images(ds,records):
    from backend.dataset_archives import open_archive,ARCHIVE_SUFFIXES
    from backend.image_ingestion import inspect_image
    if not ds.source_path:return
    is_archive=Path(ds.name).suffix.lower() in ARCHIVE_SUFFIXES
    @contextmanager
    def source():
        if is_archive:
            with open_archive(ds.source_path,ds.name) as archive:yield archive.read
        else:yield lambda name:Path(ds.source_path).read_bytes()
    with source() as read:
        for record in records:
            raw=read(record['image'])
            if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Staged image content changed')
            # Reapply decoder safety limits before this optional second decode.
            row,_=inspect_image(raw,record['image'])
            if not row['readable']:raise ValueError('Visual sample cannot be decoded')
            with Image.open(io.BytesIO(raw)) as im:
                rgb=im.convert('RGB')
                if record.get('bbox'):
                    x,y,w,h=record['bbox'];width,height=rgb.size
                    left=max(0,int(np.floor((x-w/2)*width)));top=max(0,int(np.floor((y-h/2)*height)))
                    right=min(width,int(np.ceil((x+w/2)*width)));bottom=min(height,int(np.ceil((y+h/2)*height)))
                    rgb=rgb.crop((left,top,right,bottom))
                yield record,rgb

def infer(ds,g,kind,objects=False):
    config=configuration()
    if not config or ds.source_path is None:return None
    spec=config.get('models',{}).get(kind)
    if not spec:return None
    # An absent optional dependency/model is missing evidence, not an API crash.
    try:import onnxruntime as ort
    except ImportError:return None
    path=Path(spec.get('path',''))
    if not path.is_file():return None
    if not spec.get('name') or not spec.get('version') or not spec.get('sha256'):
        raise ValueError('Local model name, version and SHA256 required')
    model_hash=fingerprint(path)
    if model_hash!=spec['sha256']:raise ValueError('Configured model hash mismatch')
    size=int(spec.get('image_size',224));batch=int(config.get('batch_size',8))
    limit=int(config.get('max_objects',512) if objects else config.get('max_images',256))
    if not 8<=size<=1024 or not 1<=batch<=32 or not 1<=limit<=2048:raise ValueError('Vision resource bounds exceeded')
    seed=int(config.get('seed',0));rng=np.random.default_rng(seed)
    records=[]
    if objects:
        population=valid_objects(ds)
        indices=sorted(rng.choice(len(population),min(len(population),limit),replace=False).tolist())
        hashes=ds.image_hashes or ds.frame.set_index('image').sha256.to_dict()
        records=[{**population[i],'sha256':hashes[population[i]['image']]} for i in indices]
    else:
        population=ds.frame.loc[ds.frame.readable,['image','sha256']].to_dict('records')
        indices=sorted(rng.choice(len(population),min(len(population),limit),replace=False).tolist())
        records=[population[i] for i in indices]
    if not records:return None
    public_config={k:v for k,v in spec.items() if k!='path'}
    runtime=getattr(ort,'__version__','test-double')
    available=ort.get_available_providers();providers=['CPUExecutionProvider']
    if config.get('device')=='cuda' and 'CUDAExecutionProvider' in available:providers.insert(0,'CUDAExecutionProvider')
    signature=hashlib.sha256(json.dumps({'spec':public_config,'seed':seed,'limit':limit,'batch':batch,'objects':objects,
        'onnxruntime':runtime,'providers':providers,'preprocessing_version':'rgb-resize-bicubic-nchw-1'},sort_keys=True).encode()).hexdigest()
    job=_job.get();jobkey=(ds.sha256,signature,kind)
    if job and jobkey in job['results']:return job['results'][jobkey]
    session=ort.InferenceSession(str(path),providers=providers)
    inputs=session.get_inputs()
    if len(inputs)!=1:raise ValueError('Encoder/classifier must accept one NCHW tensor')
    mean=np.asarray(spec.get('mean',[0,0,0]),dtype=np.float32).reshape(1,1,3)
    std=np.asarray(spec.get('std',[1,1,1]),dtype=np.float32).reshape(1,1,3)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid normalization')
    cache=Path(config.get('cache_dir','.runtime/vision-cache'));cache.mkdir(parents=True,exist_ok=True)
    vectors=[None]*len(records);pending=[];tensor=[]
    def flush():
        if not pending:return
        if job:job['progress']('Optional local vision batch')
        out=np.asarray(session.run(None,{inputs[0].name:np.stack(tensor)})[int(spec.get('output_index',0))],dtype=np.float32)
        if out.ndim!=2 or out.shape[0]!=len(pending) or not 1<=out.shape[1]<=4096 or not np.isfinite(out).all():
            raise ValueError('Expected finite NxD model output (D<=4096)')
        for (index,target),v in zip(pending,out):
            vectors[index]=v
            # Unique atomic temporary file supports concurrent assessment jobs.
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(dir=cache,suffix='.npy',delete=False) as temporary:
                np.save(temporary,v,allow_pickle=False);temp=Path(temporary.name)
            temp.replace(target)
        pending.clear();tensor.clear()
    to_decode=[]
    for i,record in enumerate(records):
        key=hashlib.sha256((record['sha256']+signature+json.dumps(record.get('bbox'))).encode()).hexdigest()
        target=cache/(key+'.npy')
        if target.is_file() and target.stat().st_size<=32768:
            v=np.load(target,allow_pickle=False)
            if v.ndim==1 and 1<=len(v)<=4096 and np.isfinite(v).all():vectors[i]=v;continue
        to_decode.append({**record,'index':i,'target':target})
    for record,im in _source_images(ds,to_decode):
        pixels=np.asarray(im.resize((size,size),Image.Resampling.BICUBIC),dtype=np.float32)/255
        tensor.append(((pixels-mean)/std).transpose(2,0,1));pending.append((record['index'],record['target']))
        if len(pending)>=batch:flush()
    flush()
    result={'vectors':np.stack(vectors),'records':records,'population':len(population),'count':len(records),
            'model':spec['name']+'@'+spec['version'],'model_sha256':model_hash,'configuration_sha256':signature,
            'providers':session.get_providers(),'runtime_version':runtime,'preprocessing':public_config,
            'sampling':'uniform without replacement; configured seed','seed':seed}
    if job:job['results'][jobkey]=result
    return result

def public(data):
    return {k:v for k,v in data.items() if k not in ('vectors','records')}

def normalized(v):
    norm=np.linalg.norm(v,axis=1,keepdims=True)
    if (norm==0).any():raise ValueError('Zero vector cannot define cosine similarity')
    return v/norm

def dispersion(v):
    v=normalized(v);center=v.mean(axis=0);norm=np.linalg.norm(center)
    return float((1-v@(center/norm)).mean()) if norm>0 else 1.

def nearest_within(vectors,indices):
    """One-query working memory, exact neighbors within the bounded sample."""
    if len(indices)<2:return [],[]
    scores=[];pairs=[]
    for i in indices:
        candidates=[j for j in indices if j!=i]
        similarities=vectors[candidates]@vectors[i]
        nearest=int(np.argmax(similarities));j=candidates[nearest]
        scores.append(float(similarities[nearest]));pairs.append((i,j,float(similarities[nearest])))
    return scores,pairs

def visual_summary(ds,g,kind):
    data=infer(ds,g,kind)
    if data:
        labels=configuration()['models'][kind].get('labels')
        logits=data['vectors']
        if not labels or len(labels)!=logits.shape[1] or len(labels)!=len(set(labels)):raise ValueError('Classifier label map mismatch')
        exp=np.exp(logits-logits.max(axis=1,keepdims=True));prob=exp/exp.sum(axis=1,keepdims=True)
        indices=prob.argmax(axis=1)
        return {**public(data),'estimated_categories':categories([labels[i] for i in indices]),
            'max_softmax_probability':summary(prob.max(axis=1),'uncalibrated probability'),
            'predictive_entropy':summary(-(prob*np.log2(np.maximum(prob,1e-30))).sum(axis=1),'bits'),
            'uncertainty_limit':'Softmax confidence is not calibrated correctness; independent validation required.'}
    if kind=='scene':
        data=infer(ds,g,'encoder')
        if data:
            v=normalized(data['vectors']);scores,_=nearest_within(v,list(range(len(v))))
            return {**public(data),'mean_cosine_distance_to_centroid':dispersion(v),
                    'nearest_neighbor_cosine_similarity':summary(scores,'cosine similarity'),
                    'scene_categories':'unavailable; global image embeddings include objects and backgrounds'}
    return None

def object_analysis(ds,g,discriminative=False):
    data=infer(ds,g,'encoder',objects=True)
    if not data:return None
    vectors=normalized(data['vectors']);records=data['records'];by_class=defaultdict(list)
    for i,o in enumerate(records):by_class[o['class']].append(i)
    job=_job.get();basekey=(ds.sha256,data['configuration_sha256'],'appearance_summary')
    base=job['results'].get(basekey) if job else None
    if base is None:
        within={};positive_pairs=[]
        for cls,indices in by_class.items():
            scores,pairs=nearest_within(vectors,indices)
            within[cls]={'count':len(indices),'mean_cosine_distance_to_centroid':dispersion(vectors[indices]),
                'nearest_within_class_similarity':summary(scores,'cosine similarity'),
                'redundancy_interpretation':'Nearest-neighbor similarity is a review proxy, not semantic duplicate ground truth.'}
            positive_pairs.extend(pairs)
        base={**public(data),'per_class':within,
            'within_class_review_candidates':[{'a':records[i]['reference'],'b':records[j]['reference'],'cosine_similarity':score}
                for i,j,score in sorted(positive_pairs,key=lambda x:x[2])[:30]]}
        if job:job['results'][basekey]=base
    out=dict(base)
    if not discriminative:return out
    # O(sample_size² * dimension), at most 2048 sampled objects; use one vector
    # against the sample at a time (O(sample_size) working memory), never N² RAM.
    pairs=[]
    labels=np.array([o['class'] for o in records])
    nearest=[]
    for i,v in enumerate(vectors):
        candidates=np.flatnonzero(labels!=labels[i])
        if not len(candidates):continue
        sim=vectors[candidates]@v;j=int(candidates[int(sim.argmax())]);score=float(sim.max());nearest.append(score)
        pairs.append({'a':records[i]['reference'],'b':records[j]['reference'],'cosine_similarity':score})
    out['cross_class_nearest_similarity']=summary(nearest,'cosine similarity')
    out['review_candidates']=sorted(pairs,key=lambda x:x['cosine_similarity'],reverse=True)[:30]
    out['cross_class_status']='estimated' if nearest else 'insufficient_evidence: at least two annotated classes needed in the sample'
    out['pair_scope']='nearest cross-class neighbors within configured object sample, not full dataset'
    out['semantic_correctness']='unverified; candidates require independent review'
    return out
