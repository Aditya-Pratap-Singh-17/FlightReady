"""Deterministic properties and optional visual evidence; no readiness score.

Metadata columns are explicitly mapped through Goal.uav_parameters.columns.
Default names are a documented input convention, not inferred sensor truth.
Image measurements use all ingested rows unless a result states a sampled scope.
"""
from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from backend.context import MetricResult, AssessmentEvidence

VERSION='1.0'

def summary(values, unit=None):
    a=np.asarray(list(values),dtype=float)
    a=a[np.isfinite(a)]
    if not len(a): return {'count':0,'unit':unit}
    q=np.quantile(a,[0,.25,.5,.75,1])
    return dict(count=len(a),unit=unit,minimum=float(q[0]),q25=float(q[1]),median=float(q[2]),
                q75=float(q[3]),maximum=float(q[4]),mean=float(a.mean()))

def numeric(frame,column,unit=None):
    if column not in frame: return {'count':0,'missing':len(frame),'invalid':0,'unit':unit}
    s=frame[column];parsed=pd.to_numeric(s,errors='coerce')
    finite=parsed.notna() & np.isfinite(parsed)
    return {**summary(parsed[finite],unit),'missing':int(s.isna().sum()),
            'invalid':int((s.notna() & ~finite).sum())}

def categories(values,total=None):
    s=pd.Series(list(values),dtype=object)
    s=s[s.notna() & s.astype(str).str.strip().ne('')].astype(str)
    counts={str(k):int(v) for k,v in s.value_counts().items()}
    n=len(s);total=n if total is None else total
    return {'counts':counts,'fractions':{k:v/n for k,v in counts.items()} if n else {},
            'observed':n,'missing':total-n,'denominator':n,'population':total}

def column(g,key,default=None):
    return g.uav_parameters.get('columns',{}).get(key,default or key)

def emit(mid,ds,value=None,status='measured',reason=None,n=None,scope='full_ingested_dataset',details=None,estimation=None):
    from metrics.core import REGISTRY
    d=REGISTRY[mid]
    state={'measured':'warning','estimated':'warning','unsupported':'insufficient_evidence'}.get(status,status)
    if status in ('measured','estimated'):
        reason=reason or 'Descriptive evidence; no acceptance decision without a confirmed criterion.'
    return MetricResult(metric_id=mid,dimension=d.dimension,state=state,value=value,
        reason=reason or 'Required evidence is unavailable.',assessment_status=status,unit=d.unit,
        dataset_scope=scope,provenance={'dataset_sha256':ds.sha256,
            'pixel_evidence':'decoded image data' if ds.modality=='image' else None,
            'metadata_evidence':'user-declared metadata; authenticity not independently verified'},
        estimation_method=estimation,limitations=d.limitations,
        evidence=AssessmentEvidence(method=d.calculation_method,sample_size=n,
            details=details or {}),reproducibility={'implementation_version':d.implementation_version})

def unavailable(mid,ds,reason,status='insufficient_evidence'):
    return emit(mid,ds,status=status,reason=reason)

def observed_category(mid,ds,g,key):
    c=column(g,key)
    if c not in ds.frame or ds.frame[c].dropna().empty:
        return unavailable(mid,ds,f'Supply recorded {key} metadata (column {c}) or an explicitly configured local classifier where supported.')
    value=categories(ds.frame[c],len(ds.frame))
    required=g.target_environment.get(c,g.target_environment.get(key))
    if isinstance(required,list):value['unobserved_declared_categories']=[str(x) for x in required if str(x) not in value['counts']]
    return emit(mid,ds,value,n=value['observed'],details={'column':c,'evidence_type':'declared_metadata'})

def valid_objects(ds):
    from metrics.uav_metrics.research import object_records
    # Default descriptive bins do not alter validity or historical geometry.
    from backend.context import Goal
    return object_records(ds,Goal())

def bins(values,bounds):
    """Explicit [lower,upper) bins; null bounds represent open ends."""
    output={}
    for name,edge in bounds.items():
        if len(edge)!=2: raise ValueError('Bins require two bounds')
        lo,hi=edge
        if lo is not None and hi is not None and lo>=hi:raise ValueError('Invalid bin bounds')
        output[name]=sum((lo is None or v>=lo) and (hi is None or v<hi) for v in values)
    return output

def geometry(ds,g):
    dimensions=ds.frame.set_index('image')[['width','height']].to_dict('index')
    fields=defaultdict(lambda:defaultdict(list))
    for o in valid_objects(ds):
        x,y,w,h=o['bbox'];size=dimensions[o['image']]
        width,height=w*size['width'],h*size['height']
        for key,value in dict(width_pixels=width,height_pixels=height,area_pixels=width*height,
                              aspect_ratio=w/h*size['width']/size['height'],relative_occupancy=w*h).items():
            fields[o['class']][key].append(value)
    out={}
    for cls,fs in fields.items():
        out[cls]={k:summary(v,'pixel' if k.endswith('_pixels') and k!='area_pixels' else 'pixel²' if k=='area_pixels' else 'ratio') for k,v in fs.items()}
        for key,b in g.uav_parameters.get('scale_bins',{}).items():
            if key not in fs:raise ValueError('Unknown scale bin measurement')
            out[cls][key]['declared_bin_counts']=bins(fs[key],b)
    return out

def density(ds,g):
    objects=valid_objects(ds);by_image=defaultdict(list)
    for o in objects:by_image[o['image']].append(o)
    annotated=ds.frame[ds.frame.readable & ds.frame.annotation_present & ds.frame.annotation_valid]
    per_image=[];per_mp=[];by_class=defaultdict(list);ious=[];distances=[]
    limit=int(g.uav_parameters.get('max_spatial_pairs',10000))
    if not 1<=limit<=100000:raise ValueError('max_spatial_pairs must be 1..100000')
    possible=0
    classes=sorted({o['class'] for o in objects})
    # Ordered deterministic prefix pairs, explicitly reported as such, not random.
    for row in annotated.itertuples():
        boxes=by_image[row.image];n=len(boxes);per_image.append(n)
        per_mp.append(n/(row.width*row.height/1e6))
        count=Counter(o['class'] for o in boxes)
        for cls in classes:by_class[cls].append(count[cls])
        possible+=n*(n-1)//2
        if len(ious)>=limit:continue
        for i,a in enumerate(boxes):
            if len(ious)>=limit:break
            for b in boxes[i+1:i+1+limit-len(ious)]:
                x,y,w,h=a['bbox'];xx,yy,ww,hh=b['bbox']
                iw=max(0,min(x+w/2,xx+ww/2)-max(x-w/2,xx-ww/2))
                ih=max(0,min(y+h/2,yy+hh/2)-max(y-h/2,yy-hh/2))
                intersection=iw*ih
                ious.append(intersection/(w*h+ww*hh-intersection))
                distances.append(math.hypot((x-xx)*row.width,(y-yy)*row.height)/math.hypot(row.width,row.height))
    counts=bins(per_image,g.uav_parameters.get('density_bins',{}))
    return {'objects_per_image':summary(per_image,'objects/image'),
        'objects_per_megapixel':summary(per_mp,'objects/megapixel'),
        'per_class_objects_per_image':{k:summary(v,'objects/image') for k,v in by_class.items()},
        'declared_density_bins':counts,'pair_iou':summary(ious,'ratio'),
        'pair_center_distance':summary(distances,'fraction of image diagonal'),
        'pair_population':possible,'pairs_examined':len(ious),'pair_policy':'ordered prefix up to max_spatial_pairs',
        'excluded_images':len(ds.frame)-len(annotated)}

def legacy(ds,g,mid,r=None,p=None):
    from metrics.core import run_metric
    m=run_metric(mid,ds,g,p,reference=r)
    return {'metric_id':mid,'state':m.state,'value':m.value,'denominator':m.evidence.sample_size,
            'reason':m.reason,'details':m.evidence.details}

def calculate(mid,ds,g,p,r):
    # Never calculate full-dataset results from the disk backend's 100-row preview.
    if ds.disk_path:
        return unavailable(mid,ds,'UAV metadata/temporal summaries do not yet query disk-backed tables; existing full-data tabular metrics remain available.','unsupported')
    n=len(ds.frame)
    images=ds.modality=='image'
    image_only={'UAV-01','UAV-05','UAV-08','UAV-09','UAV-10','UAV-11','UAV-12','UAV-13','UAV-14','UAV-15','UAV-16'}
    if mid in image_only and not images:
        return unavailable(mid,ds,'This UAV image property does not apply to the tabular modality.','not_applicable')
    detection={'UAV-08','UAV-09','UAV-10','UAV-11','UAV-13','UAV-15','UAV-16'}
    if mid in detection and ds.profile.get('dataset_kind') in ('image_classification','unlabelled_images'):
        return unavailable(mid,ds,'Object annotations are not provided by this collection.','not_applicable' if ds.profile.get('dataset_kind')=='image_classification' else 'insufficient_evidence')
    if mid=='UAV-01':
        v={'luminance':numeric(ds.frame,'brightness','8-bit gray level'),'contrast':numeric(ds.frame,'contrast','gray standard deviation')}
        c=column(g,'lighting')
        if c in ds.frame:v['declared_lighting']=categories(ds.frame[c],n)
        from metrics.uav_metrics.vision import visual_summary
        predicted=visual_summary(ds,g,'lighting')
        if predicted:v['estimated_lighting']=predicted
        return emit(mid,ds,v,n=v['luminance']['count']) if v['luminance']['count'] else unavailable(mid,ds,'No readable pixels.')
    if mid in ('UAV-02','UAV-05'):
        key='weather' if mid=='UAV-02' else 'scene'
        m=observed_category(mid,ds,g,key)
        if m.assessment_status=='measured':return m
        from metrics.uav_metrics.vision import visual_summary
        predicted=visual_summary(ds,g,key)
        if predicted:return emit(mid,ds,predicted,status='estimated',n=predicted['count'],scope='configured_visual_sample',estimation=predicted['model'])
        return m
    if mid=='UAV-03':
        c=column(g,'time_of_day')
        if c in ds.frame:return observed_category(mid,ds,g,'time_of_day')
        c=column(g,'timestamp',g.date_column or 'timestamp')
        if c not in ds.frame:return unavailable(mid,ds,'Recorded timestamps or time_of_day metadata required; appearance is not capture time.')
        times=pd.to_datetime(ds.frame[c],errors='coerce',utc=True,format='mixed')
        if not times.notna().any():return unavailable(mid,ds,'No parseable recorded timestamps.')
        return emit(mid,ds,{'utc_hour':categories(times.dropna().dt.hour,n),'missing':int(ds.frame[c].isna().sum()),
            'invalid':int((ds.frame[c].notna() & times.isna()).sum()),'timezone_policy':'parse explicit offsets; naive values interpreted as UTC, verify source'},n=int(times.notna().sum()))
    if mid=='UAV-04':
        if column(g,'season') in ds.frame:return observed_category(mid,ds,g,'season')
        c=column(g,'timestamp',g.date_column or 'timestamp');hemisphere=g.uav_parameters.get('hemisphere')
        if c not in ds.frame or hemisphere not in ('north','south'):
            return unavailable(mid,ds,'Supply declared seasons, or recorded timestamps and explicit north/south hemisphere for meteorological seasons.')
        months=pd.to_datetime(ds.frame[c],errors='coerce',utc=True,format='mixed').dropna().dt.month
        if months.empty:return unavailable(mid,ds,'No parseable collection dates.')
        names=['winter','spring','summer','autumn']
        values=[names[((m%12)//3+(2 if hemisphere=='south' else 0))%4] for m in months]
        return emit(mid,ds,{'meteorological_seasons':categories(values,n),'hemisphere':hemisphere},n=len(months),details={'assumption':'UTC month calendar, explicit hemisphere; not visual inference'})
    if mid=='UAV-06':
        if column(g,'viewpoint') in ds.frame:return observed_category(mid,ds,g,'viewpoint')
        c=column(g,'camera_pitch')
        unit=ds.metadata.get('units',{}).get(c)
        if c not in ds.frame or unit not in ('degree','degrees','deg','radian','rad'):
            return unavailable(mid,ds,'Supply viewpoint categories or calibrated camera_pitch metadata with explicit angular units.')
        pitch=numeric(ds.frame,c,unit)
        if not pitch['count']:return unavailable(mid,ds,'No finite camera orientation records.')
        return emit(mid,ds,{'camera_pitch':pitch,'calibration_declared':bool(ds.metadata.get('calibration'))},n=pitch['count'])
    if mid=='UAV-07':
        c=column(g,'altitude','altitude_meters');unit=ds.metadata.get('units',{}).get(c)
        if c not in ds.frame or not unit:return unavailable(mid,ds,'Flight altitude column and explicit unit/datum metadata required; apparent object scale is not altitude.')
        v=numeric(ds.frame,c,unit)
        if not v['count']:return unavailable(mid,ds,'No finite altitude measurements.')
        v['datum']=ds.metadata.get('altitude_datum','unspecified')
        finite=pd.to_numeric(ds.frame[c],errors='coerce').dropna()
        v['declared_bin_counts']=bins(finite[np.isfinite(finite)],g.uav_parameters.get('altitude_bins',{}))
        return emit(mid,ds,v,n=v['count'])
    if mid=='UAV-08':
        v=geometry(ds,g)
        from metrics.uav_metrics.research import scale_analysis
        enhanced=scale_analysis(ds,g)
        return emit(mid,ds,{'per_class':v,'scale_analysis':enhanced},n=enhanced['object_count'],
            status='measured' if v else 'insufficient_evidence',reason=None if v else 'No valid annotated objects on readable images; counts and missing scope retained.')
    if mid in ('UAV-09','UAV-16'):
        from metrics.uav_metrics.vision import object_analysis
        v=object_analysis(ds,g,discriminative=mid=='UAV-16')
        return emit(mid,ds,v,status='estimated',n=v['count'],scope='configured_object_sample',estimation=v['model']) if v else unavailable(mid,ds,'Configure a local visual encoder; valid crop embeddings and model/version are required. No model is downloaded.')
    if mid in ('UAV-10','UAV-11'):
        objs=valid_objects(ds);key='occlusion' if mid=='UAV-10' else 'truncation'
        by_class={cls:categories([o.get(key) for o in objs if o['class']==cls],sum(o['class']==cls for o in objs)) for cls in sorted({o['class'] for o in objs})}
        observed=sum(v['observed'] for v in by_class.values())
        v={'per_class':by_class,'annotation_format':ds.profile.get('annotation_format'),'total_objects':len(objs)}
        if mid=='UAV-11':
            if not objs:return unavailable(mid,ds,'No valid boxes for border-contact diagnostics.')
            border=Counter()
            for o in objs:
                x,y,w,h=o['bbox']
                if min(x-w/2,y-h/2,1-x-w/2,1-y-h/2)<=1e-9:border[o['class']]+=1
            counts=Counter(o['class'] for o in objs)
            v['border_contact_proxy']={c:{'count':border[c],'denominator':counts[c],'fraction':border[c]/counts[c]} for c in counts}
            v['true_truncation_status']='declared_annotations' if observed else 'insufficient_evidence'
            return emit(mid,ds,v,n=len(objs),details={'border_contact_is_proxy':True})
        if not observed:return emit(mid,ds,v,status='insufficient_evidence',n=0,reason='Occlusion annotations are absent; missing values are not zero occlusion.')
        return emit(mid,ds,v,n=observed,details={'code_semantics':'annotation-format-specific; VisDrone 0=none, 1=partial, 2=heavy'})
    if mid=='UAV-12':
        v={'luminance_entropy':numeric(ds.frame,'entropy_bits','bits/pixel histogram'),
           'semantic_clutter_status':'unverified','annotated_density_source':'UAV-13'}
        from metrics.uav_metrics.pixels import local_measurements
        v.update(global_edge_density=numeric(ds.frame,'research_edge_density','fraction of thumbnail pixels'),
                 annotated_density=density(ds,g),local_complexity=local_measurements(ds,g))
        return emit(mid,ds,v,n=v['luminance_entropy']['count'],details={'proxy_scale':'aspect-preserving thumbnail, longest edge <=256; local settings explicitly recorded'}) if v['luminance_entropy']['count'] else unavailable(mid,ds,'No readable pixels for clutter proxies.')
    if mid=='UAV-13':
        v=density(ds,g)
        if not v['objects_per_image']['count']:return unavailable(mid,ds,'No readable, syntactically valid annotated images (explicit negatives are included).')
        return emit(mid,ds,v,n=v['objects_per_image']['count'],details={'max_spatial_pairs':g.uav_parameters.get('max_spatial_pairs',10000)})
    if mid=='UAV-14':
        readable=ds.frame[ds.frame.readable]
        if readable.empty:return unavailable(mid,ds,'No readable images for visual-information measurements.')
        v={k:numeric(readable,k,u) for k,u in [('width','pixel'),('height','pixel'),('brightness','gray level'),('contrast','gray standard deviation'),
            ('laplacian_variance','squared gray level'),('encoded_bytes_per_pixel','bytes/pixel')]}
        v['excluded_unreadable_images']=len(ds.frame)-len(readable)
        from metrics.uav_metrics.pixels import local_measurements
        v['global_degradation_proxies']={key:numeric(readable,'research_'+key,unit) for key,unit in (
            ('edge_density','fraction'),('high_frequency_residual_mad','normalized gray level'),('underexposed_fraction','fraction'),
            ('saturated_fraction','fraction'),('local_contrast_rms','normalized gray level'))}
        v['object_local']=local_measurements(ds,g)
        return emit(mid,ds,v,n=int(ds.frame.readable.sum()),details={'laplacian_kernel':'4-neighbor discrete Laplacian; interior pixels; population variance','proxy_scale':'thumbnail <=256 pixels longest edge, Pillow default resize'})
    if mid=='UAV-15':
        v={k:legacy(ds,g,k) for k in ['bbox_invalid_rate','missing_label_rate']}
        v['manifest_errors']=ds.profile.get('manifest_errors')
        v['annotation_policy']=ds.profile.get('annotation_policy','Not independently reviewed')
        audit=ds.metadata.get('annotation_audit')
        v['semantic_correctness']='unverified';v['visible_object_completeness']='unverified'
        if audit:
            if not isinstance(audit,dict) or not audit.get('independent') or not audit.get('reviewer') or not audit.get('protocol'):
                v['audit_status']='insufficient_evidence: independent reviewer and protocol required'
            else:
                reviewed=audit.get('reviewed_objects');incorrect=audit.get('incorrect_objects')
                if isinstance(reviewed,int) and not isinstance(reviewed,bool) and reviewed>0 and isinstance(incorrect,int) and not isinstance(incorrect,bool) and 0<=incorrect<=reviewed:
                    v['independent_audit']={'reviewed_objects':reviewed,'incorrect_objects':incorrect,'error_fraction':incorrect/reviewed,
                        'protocol':audit['protocol'],'scope':'declared reviewed objects only'}
                else:v['audit_status']='insufficient_evidence: valid independent counts required'
        from metrics.uav_metrics.audit import annotation_audit,negative_coverage
        v['structured_audit']=annotation_audit(ds,g)
        v['negative_coverage']=negative_coverage(ds,g,v['structured_audit'])
        return emit(mid,ds,v,n=n)
    if mid=='UAV-17':
        v={};cols=g.uav_parameters.get('reference_columns',[])
        from metrics.uav_metrics.research import scenarios
        conditional=scenarios(ds,g,r)
        if r is not None and cols:v['reference_dataset']=legacy(ds,g,'reference_comparison',r,{'columns':cols})
        if g.uav_parameters.get('compare_reference_embeddings') and r is not None and images and r.modality=='image':
            from metrics.uav_metrics.vision import infer,normalized
            a,b=infer(ds,g,'encoder'),infer(r,g,'encoder')
            if a and b and a['model_sha256']==b['model_sha256']:
                ca=normalized(a['vectors']).mean(axis=0);cb=normalized(b['vectors']).mean(axis=0)
                v['estimated_embedding_shift']={'centroid_euclidean_distance':float(np.linalg.norm(ca-cb)),
                    'dataset_sample':a['count'],'reference_sample':b['count'],'denominator':a['count'],
                    'unit':'normalized embedding Euclidean distance','model':a['model'],
                    'model_sha256':a['model_sha256'],'dataset_configuration_sha256':a['configuration_sha256'],
                    'reference_configuration_sha256':b['configuration_sha256'],'status':'estimated',
                    'limitations':'Mean normalized-embedding shift is a model-dependent summary; equal centroids do not establish equal distributions.'}
            else:v['estimated_embedding_shift']={'status':'insufficient_evidence','reason':'Matching configured local encoders and readable reference images required'}
        for c,reference in g.uav_parameters.get('reference_distributions',{}).items():
            if c not in ds.frame: v[c]={'status':'insufficient_evidence'};continue
            observed=categories(ds.frame[c],n)
            if not observed['observed']:v[c]={'status':'insufficient_evidence'};continue
            if not reference or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<0 for x in reference.values()) or sum(reference.values())<=0:
                raise ValueError('Reference weights must be finite, nonnegative and nonempty')
            keys=sorted(set(observed['counts'])|set(reference));total=sum(reference.values())
            a=[observed['fractions'].get(k,0) for k in keys];b=[reference.get(k,0)/total for k in keys]
            v[c]={'js_divergence_bits':float(jensenshannon(a,b,base=2)**2),'denominator':observed['observed'],
                  'missing':observed['missing'],'reference':'user-declared weights','unobserved_reference_categories':[k for k in reference if reference[k]>0 and k not in observed['counts']]}
        if not v:
            return emit(mid,ds,{'conditional_coverage':conditional},n=n,status='measured' if conditional['scenarios'] else 'insufficient_evidence',
                reason='Conditional coverage is descriptive; deployment representativeness requires an explicit meaningful reference.')
        has_measurement=any('js_divergence_bits' in x or x.get('value') is not None for x in v.values())
        has_estimate='centroid_euclidean_distance' in v.get('estimated_embedding_shift',{})
        v['conditional_coverage']=conditional
        return emit(mid,ds,v,n=n,status='measured' if has_measurement or conditional['scenarios'] else 'estimated' if has_estimate else 'insufficient_evidence',
            estimation=v.get('estimated_embedding_shift',{}).get('model') if has_estimate else None)
    if mid=='UAV-18':
        ids=['class_distribution','duplicate_rate','near_duplicates','split_overlap','group_leakage','near_split_overlap','evidence_summary']
        v={k:legacy(ds,g,k) for k in ids};v['sample_adequacy']='unverified; no universal minimum'
        if g.group_column and g.group_column in ds.frame:
            v['collection_groups']={'count':int(ds.frame[g.group_column].nunique()),'missing':int(ds.frame[g.group_column].isna().sum()),'column':g.group_column}
        from metrics.uav_metrics.independence import collection_evidence
        v['independence_analysis']=collection_evidence(ds,g)
        return emit(mid,ds,v,n=n)
    if mid=='UAV-19':return sensor(mid,ds,g)
    if mid=='UAV-20':return temporal(mid,ds,g)
    raise ValueError('Unknown UAV property')

def sensor(mid,ds,g):
    types=ds.metadata.get('sensor_types')
    if not isinstance(types,dict) or not types:return unavailable(mid,ds,'Declare sensor_types mappings, units, and calibration/alignment evidence for relevant modalities.')
    v={'sensors':{},'calibration_status':'declared_present_unverified' if ds.metadata.get('calibration') else 'insufficient_evidence',
       'alignment_status':'declared_present_unverified' if ds.metadata.get('alignment') else 'insufficient_evidence','synchronization_status':'insufficient_evidence'}
    units=ds.metadata.get('units',{})
    for name,fields in types.items():
        if name.lower() not in ('rgb','thermal','gps','imu','multisensor'):
            v['sensors'][name]={'status':'unsupported','reason':'No modality-specific validation is implemented for this sensor'};continue
        if not isinstance(fields,list) or not fields or any(not isinstance(c,str) for c in fields):
            v['sensors'][name]={'status':'insufficient_evidence','reason':'mapping must be a nonempty column list'};continue
        v['sensors'][name]={'missing_fields':[c for c in fields if c not in ds.frame],
            'units':{c:units.get(c) for c in fields},'unit_mismatches':{c:{'actual':units.get(c),'expected':g.expected_units[c]} for c in fields if c in g.expected_units and units.get(c)!=g.expected_units[c]},
            'missing_records':{c:int(ds.frame[c].isna().sum()) for c in fields if c in ds.frame}}
    sync=g.uav_parameters.get('synchronization')
    if sync:
        a,b=sync['columns']
        if a in ds.frame and b in ds.frame:
            ta=pd.to_datetime(ds.frame[a],errors='coerce',utc=True,format='mixed');tb=pd.to_datetime(ds.frame[b],errors='coerce',utc=True,format='mixed')
            delta=(ta-tb).dt.total_seconds().abs().dropna();v['timestamp_delta_seconds']=summary(delta,'seconds');v['missing_pairs']=len(ds.frame)-len(delta)
            if len(delta):v['synchronization_status']='paired_timestamp_deltas_measured'
            if 'tolerance_seconds' in sync and len(delta):
                tolerance=float(sync['tolerance_seconds'])
                if not math.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid synchronization tolerance')
                v['pairs_within_declared_tolerance']=int((delta<=tolerance).sum());v['tolerance_seconds']=tolerance
    return emit(mid,ds,v,n=len(ds.frame),reason='Schema, metadata presence and available timestamp evidence only; sensor physics/calibration validity remain unverified.')

def temporal(mid,ds,g):
    kind=ds.metadata.get('collection_kind','static_images' if ds.modality=='image' else None)
    c=column(g,'timestamp',g.date_column or 'timestamp')
    if kind=='static_images' and g.task_type!='tracking':
        return unavailable(mid,ds,'Static-image collection; temporal continuity is not applicable. Declare collection_kind=video/telemetry and sequence metadata when appropriate.','not_applicable')
    if c not in ds.frame:return unavailable(mid,ds,'Recorded timestamps and sequence identifiers required for temporal integrity.')
    group=column(g,'sequence',g.group_column or 'sequence_id')
    data=ds.frame.copy();data['_time']=pd.to_datetime(data[c],errors='coerce',utc=True,format='mixed')
    if not data['_time'].notna().any():return unavailable(mid,ds,'No parseable recorded timestamps for temporal checks.')
    groups=data.groupby(group,sort=False,dropna=False) if group in data else [(None,data)]
    deltas=[];repeated=0;backward=0;frame_gaps=0;frame_backwards=0;track_repeats=0;intervals=0
    frame=column(g,'frame','frame_index');track=column(g,'track','track_id');frame_intervals=0
    for _,rows in groups:
        # diff before dropna prevents bridging a missing timestamp as continuity.
        d=rows['_time'].diff().dt.total_seconds().dropna();deltas.extend(d.tolist());intervals+=len(d)
        repeated+=int(rows['_time'].dropna().duplicated().sum());backward+=int((d<0).sum())
        if frame in rows:
            f=pd.to_numeric(rows[frame],errors='coerce').diff().dropna();frame_intervals+=len(f)
            frame_gaps+=int((f>1).sum());frame_backwards+=int((f<=0).sum())
        if track in rows and frame in rows:
            track_repeats+=int(rows[[frame,track]].dropna().duplicated().sum())
    v={'timestamp_records':int(data['_time'].notna().sum()),'missing_timestamps':int(data[c].isna().sum()),
       'invalid_timestamps':int((data[c].notna() & data['_time'].isna()).sum()),
       'sequence_count':int(data[group].nunique()) if group in data else None,
       'missing_sequence_ids':int(data[group].isna().sum()) if group in data else len(data),
       'cross_sequence_intervals_excluded':group in data,'repeated_timestamps':repeated,'backwards_intervals':backward,
       'adjacent_timestamp_intervals':intervals,'intervals_seconds':summary(deltas,'seconds'),
       'frame_gap_events':frame_gaps if frame in data else None,'frame_order_violations':frame_backwards if frame in data else None,
       'adjacent_frame_intervals':frame_intervals,'duplicate_track_frame_records':track_repeats if track in data and frame in data else None,
       'complete_track_frame_records':int(data[[track,frame]].dropna().shape[0]) if track in data and frame in data else None,
       'missing_track_ids':int(data[track].isna().sum()) if track in data else None,
       'known_track_id_count':int(data[track].nunique()) if track in data else None,
       'tracking_identity_correctness':'unverified','physical_motion':'unsupported without calibrated positions/flow',
       'ordering':'record order within declared sequence; no sorting before validation'}
    return emit(mid,ds,v,n=len(data),scope='recorded_sequence_order' if group in data else 'record_order_sequence_unknown')
