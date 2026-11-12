"""Assess the actual union of combined training manifests using cached evidence."""
from collections import Counter
import copy
import hashlib
import pandas as pd
from backend.context import DatasetContext
from backend.ingestion import Dataset
from metrics.dataProfiler import Dataprofiler
from backend.readiness_engine import assess
from metrics.quality_profiles import fingerprint
from metrics.quality_aggregation import aggregate_quality
from backend.workbench.datasets import load,subset,persist
from backend.workbench.splits import get_split


def combined_readiness(store,config,classes):
    frames=[];objects=[];hashes={};sources={};parts=[];issues=[]
    source_specs=[(config['dataset_id'],config['split_id'],{}),
                  (config['reference_dataset_id'],config['reference_split_id'],config['class_mapping'])]
    for id,split_id,mapping in source_specs:
        source=store.get(id,'dataset')['payload']
        manifest=get_split(store,split_id,id)
        part=subset(load(store,id),manifest['images']['train'])
        frame=part.frame.copy()
        renamed={name:id+'/'+name for name in part.image_hashes}
        frame['image']=frame['image'].map(renamed)
        frames.append(frame)
        for obj in part.objects:
            objects.append({**obj,'image':renamed[obj['image']],'class':mapping.get(obj['class'],obj['class'])})
        for name,digest in part.image_hashes.items():
            virtual=renamed[name]
            hashes[virtual]=digest
            sources[virtual]={'source':{k:source[k] for k in ('source_kind','source_path','source_name')},'image':name}
        issues.extend({**i,'source_dataset_id':id} for i in part.issues)
        parts.append({'dataset_id':id,'split_id':split_id,'training_sha256':part.sha256,'metadata':part.metadata,'name':part.name})
    frame=pd.concat(frames,ignore_index=True,sort=False)
    counts=Counter(o['class'] for o in objects)
    # Only common metadata conventions survive. Independent audits are not
    # silently reinterpreted as a combined-population audit.
    metadata={k:copy.deepcopy(v) for k,v in parts[0]['metadata'].items()
              if k not in ('annotation_audit','hard_negative_candidates','dataset_version','source','collection_date')
              and all(p['metadata'].get(k)==v for p in parts[1:])}
    identity=fingerprint({'training_sources':[{k:v for k,v in p.items() if k!='metadata'} for p in parts],
                          'reference_mapping':config['class_mapping'],'classes':classes})
    profile={**Dataprofiler(frame).profile().to_dict(),'dataset_kind':'object_detection','annotation_format':'combined_cached',
        'image_count':len(frame),'object_count':len(objects),'class_counts':dict(counts),'declared_classes':classes,
        'assessment_scope':'combined training manifests only','near_duplicate_analysis':'Full cross-source perceptual pair counts not assessed; source split manifests retain their leakage checks.',
        'source_dataset_ids':[p['dataset_id'] for p in parts],'issues':issues}
    ds=Dataset('Combined training: '+' + '.join(p['name'] for p in parts),identity,'image',frame,profile,
               metadata=metadata,objects=objects,issues=issues,image_hashes=hashes,near_distance=None,source_path=None)
    record=persist(store,ds,None,'virtual_combined','combined',provenance={'combined_sources':parts,'cached_measurements':True})
    store.update(record['id'],archived=True,virtual_sources=sources)
    from backend.workbench.datasets import image_source
    virtual_payload=store.get(record['id'],'dataset')['payload']
    ds.image_reader=lambda:image_source(virtual_payload)
    source=store.get(config['readiness_id'],'job')['payload']['result']
    context=DatasetContext.model_validate(source['confirmed_context'])
    certificate=assess(ds,context)
    selected=(store.get(config['aggregation_id'],'aggregation')['payload']['result'] if config.get('aggregation_id') else source.get('quality_profile'))
    if selected:
        overrides=selected.get('confirmed_overrides',{})
        certificate.quality_profile=aggregate_quality(certificate.model_dump(mode='json'),scenario=selected['scenario'],profile=selected['profile'],
            dimension_weights=overrides.get('dimensions',{}),metric_weights=overrides.get('metrics',{}),overrides_confirmed=True)
        certificate.reproducibility['configuration']['adaptive_quality_profile_sha256']=certificate.quality_profile['profile_sha256']
        certificate.reproducibility['assessment_fingerprint']=hashlib.sha256((certificate.reproducibility['assessment_fingerprint']+certificate.quality_profile['calculation_sha256']).encode()).hexdigest()
    certificate.limitations.append('Combined cached evidence reuses original decoded pixel/annotation measurements. Optional visual-model inference and full cross-source perceptual pair counts remain unassessed; original source manifests remain available.')
    job=store.create('job',{'task':'readiness','status':'completed','stage':'Completed','config':{'dataset_id':record['id'],'split_id':None,
        'context':context.model_dump(mode='json'),'source_dataset_ids':[p['dataset_id'] for p in parts]},'result':certificate.model_dump(mode='json'),'error':None})
    return {'combined_dataset_id':record['id'],'combined_readiness_id':job['id']}
