"""Derived analytical views: references to measured properties, never scores."""
import math
from collections import defaultdict
import pandas as pd

def associations(ds,g):
    """Image-level label PRESENCE vs nuisance category normalized mutual information.

    Each class is a binary presence label. Explicit annotated negatives contribute
    zeros; missing/invalid annotations never become absent class labels. An image
    contributes once per class, avoiding crowded frames dominating the table.
    """
    if ds.modality!='image' or ds.disk_path:return {'status':'not_applicable'}
    frame=ds.frame[ds.frame.readable & ds.frame.annotation_present & ds.frame.annotation_valid]
    labels=defaultdict(set)
    for o in ds.objects:labels[o['image']].add(o['class'])
    columns=g.uav_parameters.get('nuisance_columns',['scene','lighting','source'])
    output={}
    classes=g.required_classes or sorted({x for im in frame.image for x in labels[im]})
    if len(classes)>100:
        return {'status':'unsupported','reason':'Select at most 100 target classes for conditional nuisance analysis; full class frequencies remain available.'}
    for c in columns:
        if c not in frame:continue
        subset=frame[frame[c].notna()]
        values=subset[c].astype(str)
        if values.nunique()>100:
            output[c]={'status':'unsupported','reason':'More than 100 nuisance categories; declare a coarser category mapping'};continue
        per_class={}
        for cls in classes:
            if subset.empty:continue
            table=pd.crosstab(values,pd.Series([int(cls in labels[im]) for im in subset.image],index=subset.index))
            n=table.to_numpy().sum();joint=table.to_numpy()/n
            rows=joint.sum(axis=1);cols=joint.sum(axis=0)
            entropy=lambda a:-sum(float(x)*math.log2(float(x)) for x in a if x>0)
            hx,hy=entropy(rows),entropy(cols)
            mi=sum(float(joint[i,j])*math.log2(float(joint[i,j]/(rows[i]*cols[j]))) for i in range(len(rows)) for j in range(len(cols)) if joint[i,j]>0)
            per_class[cls]={'normalized_mutual_information':mi/math.sqrt(hx*hy) if hx>0 and hy>0 else None,
                'mutual_information_bits':mi,'denominator':int(n),'contingency':{str(k):{str(label):int(count) for label,count in row.items()} for k,row in table.to_dict('index').items()},
                'degenerate':not(hx>0 and hy>0)}
        output[c]={'per_class':per_class,'excluded_missing_nuisance':len(frame)-len(subset)}
    return {'status':'measured' if output else 'insufficient_evidence','associations':output,
        'definition':'MI / sqrt(H(nuisance)*H(binary class presence)); undefined for constant variables',
        'limitations':'Descriptive annotation associations; class co-occurrence and collection design confound results. Correlation does not establish a trained model uses a shortcut.'}

def learning_views(ds,g,results):
    by_id={m.metric_id:m for m in results}
    def view(ids,interpretation):
        return {'supporting_metric_ids':ids,'evidence_status':{mid:by_id[mid].assessment_status for mid in ids if mid in by_id},
                'interpretation':interpretation}
    out={
        'concept_coverage':view(['UAV-08','UAV-09','UAV-10','UAV-11','UAV-15'],'Observed target-concept variation only; unannotated concepts and semantic annotation correctness require audit.'),
        'invariance_support':view(['UAV-01','UAV-05','UAV-06','UAV-08','UAV-10','UAV-14'],'Marginal condition coverage does not establish label-preserving invariance; within-class condition coverage is supporting evidence only.'),
        'shortcut_vulnerability':view(['UAV-01','UAV-05','UAV-18'],'Conditional label/nuisance associations identify candidates for controlled downstream tests.'),
        'discriminative_information':view(['UAV-09','UAV-16'],'Encoder similarities identify review candidates; discriminability of the target model remains unverified.'),
        'independent_evidence_and_redundancy':view(['UAV-18'],'Collection groups and duplicates constrain evidence interpretation; sample adequacy remains task/model-specific.'),
        'potential_deployment_domain_shift':view(['UAV-17'],'Descriptive reference differences; downstream degradation requires independent model evaluation.'),
    }
    out['shortcut_vulnerability']['conditional_associations']=associations(ds,g)
    # Observe class-conditional nuisance coverage without manufacturing transforms.
    coverage={};columns=g.uav_parameters.get('invariance_columns',['lighting','weather','viewpoint'])
    if ds.modality=='image' and not ds.disk_path:
        indexed=ds.frame.set_index('image')
        for c in columns:
            if c not in indexed:continue
            by_class=defaultdict(lambda:defaultdict(set))
            from metrics.uav_metrics.assessment import valid_objects
            for o in valid_objects(ds):
                value=indexed.loc[o['image'],c]
                if pd.notna(value):by_class[o['class']][str(value)].add(o['image'])
            coverage[c]={cls:{k:len(images) for k,images in groups.items()} for cls,groups in by_class.items()}
    out['invariance_support']['class_condition_image_counts']=coverage
    out['invariance_support']['label_preserving_transformations']='unverified; no generated transformation is assumed label-preserving'
    out['invariance_support']['required_validation']='Independent visibility and label-preservation review for each transformation and target class.'
    return out

def recommendations(results):
    from backend.context import RemediationRecommendation
    out=[]
    for m in results:
        if not m.metric_id.startswith('UAV-'):continue
        if m.assessment_status in ('insufficient_evidence','unsupported'):
            out.append(RemediationRecommendation(description='Supply or independently review evidence for '+m.metric_id+': '+m.reason,
                supporting_metric_ids=[m.metric_id],observed_evidence='Required evidence unavailable; no numerical absence was inferred.',
                hypothesized_effect='This gap limits assessment of the intended task. Its model-performance effect has not been established.',conditional=True))
        if m.metric_id=='UAV-17' and isinstance(m.value,dict):
            missing={c:v['unobserved_reference_categories'] for c,v in m.value.items() if isinstance(v,dict) and v.get('unobserved_reference_categories')}
            if missing:
                out.append(RemediationRecommendation(description='Review deployment-reference relevance and acquire independent examples for unobserved declared reference categories.',
                    supporting_metric_ids=['UAV-17','UAV-18'],observed_evidence=str(missing),conditional=True,
                    hypothesized_effect='Additional coverage may support robustness; no accuracy improvement has been experimentally established.'))
    return out
