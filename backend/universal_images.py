"""Adapters for COCO, Pascal VOC, ImageFolder, YAML YOLO and unlabelled images.

All labels retain their source. Folder-derived labels are explicitly inferred;
unlabelled images never become assumed negatives for an object-detection task.
"""
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import PurePosixPath
import csv
import hashlib
import io
import json
import math
import os
import ijson
import numpy as np
import pandas as pd
import yaml
from defusedxml import ElementTree as SafeXML
from backend.ingestion import Dataset,IngestionError,decode_csv
from metrics.dataProfiler import Dataprofiler
from backend.image_ingestion import IMAGE_SUFFIXES,find_layouts,inspect_image,parse_yolo

GENERIC_FOLDERS={'','data','dataset','datasets','images','image','imgs','pictures','photos','jpegimages','train','training','test','testing','val','valid','validation'}
SPLITS={'train','training','test','testing','val','valid','validation'}

class GuardedJSONReader:
    """Bound COCO token/container growth before the streaming parser allocates it."""
    def __init__(self,source):
        self.source=source;self.quoted=False;self.escaped=False;self.length=0
        self.position=0;self.depth=0;self.stack=[];self.bounded=[];self.first=True
    def read(self,size=-1):
        if size==0:return b''
        data=self.source.read(min(size,65536) if size>0 else 65536)
        if self.first:
            self.first=False
            if data.startswith(b'\xef\xbb\xbf'):data=data[3:]
        for char in data:
            self.position+=1
            if self.bounded and self.position-self.bounded[0]>8*1024**2:raise IngestionError('A COCO record exceeds 8 MiB')
            if self.quoted:
                self.length+=1
                if self.length>8*1024**2:raise IngestionError('A COCO string token exceeds 8 MiB')
                if self.escaped:self.escaped=False
                elif char==92:self.escaped=True
                elif char==34:self.quoted=False
            elif char==34:self.quoted=True;self.length=0
            elif char in (91,123):
                self.depth+=1
                if self.depth>64:raise IngestionError('COCO nesting exceeds 64 levels')
                limited=self.depth>=3;self.stack.append(limited)
                if limited:self.bounded.append(self.position)
            elif char in (93,125):
                if self.stack and self.stack.pop():self.bounded.pop()
                self.depth-=1
        return data

@contextmanager
def coco_source(archive,name):
    with archive.open(name) as source:yield GuardedJSONReader(source)

def folder_labels(images):
    paths=[list(PurePosixPath(name).parts[:-1]) for name in images]
    shared=0
    while paths and all(len(p)>shared and p[shared]==paths[0][shared] for p in paths):shared+=1
    result={}
    for name,parts in zip(images,paths):
        split=next((i for i,part in enumerate(parts) if part.lower() in SPLITS),None)
        index=split+1 if split is not None else shared if shared<len(parts) else len(parts)-1
        label=parts[index] if 0<=index<len(parts) else ''
        result[name]=None if label.lower() in GENERIC_FOLDERS else label
    return result

def _small_read(archive,name,maximum=8*1024**2):
    with archive.open(name) as source:data=source.read(maximum+1)
    if len(data)>maximum:raise IngestionError('Annotation/config record exceeds its safe size limit: '+name)
    return data

def coco_keys(archive,name):
    keys=set()
    try:
        with coco_source(archive,name) as source:
            for prefix,event,value in ijson.parse(source):
                if prefix=='' and event=='map_key':keys.add(value)
                if {'images','categories'}<=keys and 'annotations' in keys:return True
    except (ValueError,ijson.JSONError,UnicodeError):return False
    return False

def image_choices(archive,names,metadata):
    images=sorted(n for n in names if PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES)
    if not images:return []
    choices=[{'label':format.upper()+': '+(root or 'archive root'),'options':{'annotation_format':format,'image_root':root,'legacy_layout':True}}
             for format,root in find_layouts(names)]
    for name in sorted(names):
        if PurePosixPath(name).suffix.lower()=='.json' and coco_keys(archive,name):
            choices.append({'label':'COCO detection: '+name,'options':{'annotation_format':'coco','annotation_file':name}})
    xmls=[n for n in sorted(names) if n.lower().endswith('.xml')]
    voc=[]
    for name in xmls:
        try:
            root=SafeXML.fromstring(_small_read(archive,name),forbid_dtd=True,forbid_entities=True,forbid_external=True)
            if root.tag=='annotation':voc.append(name)
        except Exception as exc:
            # An XML candidate with a security violation is not reinterpreted as
            # benign unlabelled data. Reject it with a useful format explanation.
            if 'Forbidden' in type(exc).__name__:raise IngestionError('Unsafe XML annotation: '+name) from exc
    if voc:choices.append({'label':f'Pascal VOC detection ({len(voc)} XML files)','options':{'annotation_format':'pascal_voc'}})
    if not choices:
        yamls=[n for n in sorted(names) if PurePosixPath(n).name.lower() in {'data.yaml','dataset.yaml','data.yml','dataset.yml'} or n.lower().endswith('.names')]
        # Match side-by-side annotations for PNG/WebP/etc., not only JPEG.
        image_stems={str(PurePosixPath(n).with_suffix('')) for n in images}
        labels=[n for n in names if '/labels/' in '/'+n or PurePosixPath(n).suffix.lower()=='.txt' and str(PurePosixPath(n).with_suffix('')) in image_stems]
        if yamls and labels:
            choices.append({'label':'YOLO with YAML class names: '+yamls[0],'options':{'annotation_format':'yolo_yaml','class_file':yamls[0]}})
        elif labels:
            choices.append({'label':'YOLO boxes (class-name mapping unavailable)','options':{'annotation_format':'yolo_unmapped'}})
    if not choices:
        mapping=folder_labels(images);parents=set(mapping.values())
        labelled=any(parent is not None for parent in parents)
        if labelled:choices.append({'label':f'Image classification from folders ({len(images)} files)','options':{'annotation_format':'imagefolder'}})
        if not labelled or None in parents or len(parents)==1:
            choices.append({'label':f'Unlabelled image collection ({len(images)} files)','options':{'annotation_format':'unlabelled'}})
    return choices

def resolver(image_names):
    images=set(image_names);basenames=defaultdict(list)
    for name in image_names:basenames[PurePosixPath(name).name].append(name)
    def resolve(value,parent=''):
        if not isinstance(value,str):raise IngestionError('Image filename must be text')
        path=PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or '\\' in value:raise IngestionError('Unsafe annotation image path')
        for candidate in (value,str(PurePosixPath(parent)/value) if parent else value):
            if candidate in images:return candidate
        matches=basenames.get(path.name,[])
        if len(matches)==1:return matches[0]
        if len(matches)>1:raise IngestionError('Ambiguous image filename; select one image_root or use full annotation paths: '+value)
        return None
    return resolve

def _coco(archive,name,resolve):
    records={};by_id={};categories={};objects=defaultdict(list);errors=[]
    with coco_source(archive,name) as source:
        for item in ijson.items(source,'categories.item'):
            if not isinstance(item,dict):raise IngestionError('COCO category entries must be objects')
            id=item.get('id');label=item.get('name')
            if not isinstance(id,int) or not isinstance(label,str) or not label.strip() or id in categories:
                raise IngestionError('COCO categories need unique integer IDs and nonempty names')
            categories[id]=label
    if not categories:raise IngestionError('COCO has no category mapping')
    with coco_source(archive,name) as source:
        for item in ijson.items(source,'images.item'):
            if not isinstance(item,dict):raise IngestionError('COCO image entries must be objects')
            id=item.get('id');filename=item.get('file_name')
            if not isinstance(id,int) or id in by_id:raise IngestionError('COCO image IDs must be unique integers')
            resolved=resolve(filename,str(PurePosixPath(name).parent))
            key=resolved or 'missing:'+str(filename)
            by_id[id]=key
            if key in records:raise IngestionError('COCO maps multiple image IDs to the same file; resolve duplicate image entries')
            records[key]={'declared_width':item.get('width'),'declared_height':item.get('height'),'missing':resolved is None,'image_id':id,'filename':filename}
    if not records:raise IngestionError('COCO has no image records')
    count=0;annotation_ids=set()
    with coco_source(archive,name) as source:
        for item in ijson.items(source,'annotations.item',use_float=True):
            count+=1
            if count>int(os.getenv('MAX_ANNOTATED_OBJECTS','1000000')):raise IngestionError('COCO exceeds MAX_ANNOTATED_OBJECTS')
            if not isinstance(item,dict):
                errors.append({'reference':f'{name}:annotation:{count}','kind':'orphan_annotation','reason':'Annotation entry is not an object'});continue
            annotation_id=item.get('id')
            if annotation_id is not None:
                if not isinstance(annotation_id,int) or isinstance(annotation_id,bool) or annotation_id in annotation_ids:
                    errors.append({'reference':f'{name}:annotation:{count}','kind':'orphan_annotation','reason':'Annotation IDs must be unique integers'})
                else:annotation_ids.add(annotation_id)
            image_id=item.get('image_id')
            image=by_id.get(image_id) if isinstance(image_id,int) and not isinstance(image_id,bool) else None
            reference=f'{name}:annotation:{item.get("id",count)}'
            if image is None:
                errors.append({'reference':reference,'kind':'orphan_annotation','reason':'Annotation image_id is absent from the COCO images manifest'});continue
            objects[image].append({'class':categories.get(item.get('category_id')),'category_id':item.get('category_id'),
                'pixel_bbox':item.get('bbox'),'iscrowd':item.get('iscrowd',0),'ignored':bool(item.get('ignore',False)),
                'invalid_flags':item.get('iscrowd',0) not in (0,1) or item.get('ignore',0) not in (0,1),
                'reference':reference,'annotation_id':item.get('id')})
    return records,objects,categories,errors

def _voc(archive,names,resolve):
    records={};objects=defaultdict(list);errors=[]
    for name in sorted(n for n in names if n.lower().endswith('.xml')):
        root=SafeXML.fromstring(_small_read(archive,name),forbid_dtd=True,forbid_entities=True,forbid_external=True)
        if root.tag!='annotation':continue
        filename=root.findtext('filename')
        if not filename:raise IngestionError('Pascal VOC annotation lacks filename: '+name)
        image=resolve(filename,str(PurePosixPath(name).parent)) or 'missing:'+filename
        if image in records:raise IngestionError('Multiple VOC files reference the same image; choose a single annotation set')
        size=root.find('size')
        records[image]={'declared_width':size.findtext('width') if size is not None else None,
            'declared_height':size.findtext('height') if size is not None else None,'missing':image.startswith('missing:'),'filename':filename,'annotation_file':name}
        for index,obj in enumerate(root.findall('object'),1):
            box=obj.find('bndbox')
            objects[image].append({'class':obj.findtext('name'),'voc_box':[box.findtext(k) if box is not None else None for k in ('xmin','ymin','xmax','ymax')],
                'difficult':obj.findtext('difficult','0'),'truncated':obj.findtext('truncated','0'),'reference':f'{name}:object:{index}'})
    return records,objects,{},errors

def _yolo_classes(archive,metadata):
    if metadata.get('class_names') is not None:classes=metadata['class_names']
    elif metadata.get('class_file'):
        raw=_small_read(archive,metadata['class_file'],1024**2)
        if PurePosixPath(metadata['class_file']).suffix.lower()=='.names':classes=raw.decode('utf-8-sig').splitlines()
        else:
            try:document=yaml.safe_load(raw)
            except yaml.YAMLError as exc:raise IngestionError('Invalid or unsafe YOLO dataset YAML') from exc
            classes=document.get('names') if isinstance(document,dict) else None
    else:return None
    if isinstance(classes,dict):
        try:normalized={int(key):value for key,value in classes.items()}
        except (TypeError,ValueError) as exc:raise IngestionError('YOLO class-name keys must be integer IDs') from exc
        if len(normalized)!=len(classes):raise IngestionError('Duplicate YOLO class IDs after normalization')
        classes=normalized
        if set(classes)!=set(range(len(classes))):raise IngestionError('YOLO names mapping must use contiguous zero-based IDs')
        classes=[classes[i] for i in range(len(classes))]
    if not isinstance(classes,list) or not classes or any(not isinstance(c,str) or not c.strip() for c in classes) or len(set(classes))!=len(classes):
        raise IngestionError('YOLO class_names / YAML names must be unique nonempty strings')
    return classes

def _normal_box(box,width,height):
    if not isinstance(box,(list,tuple)) or len(box)!=4:raise ValueError('Expected four pixel bbox coordinates')
    x,y,w,h=map(float,box)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or x<0 or y<0 or w<=0 or h<=0 or x+w>width+1e-9 or y+h>height+1e-9:
        raise ValueError('Bounding box has invalid area or lies outside the decoded image')
    return [(x+w/2)/width,(y+h/2)/height,w/width,h/height]

def load_universal_images(name,archive,names,digest,metadata):
    format=metadata['annotation_format'];prefix=metadata.get('image_root','')
    image_names=sorted(n for n in names if n.startswith(prefix) and PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES)
    image_set=set(image_names)
    resolve=resolver(image_names)
    records,annotations,categories,issues={},{},{},[]
    if format=='coco':records,annotations,categories,issues=_coco(archive,metadata['annotation_file'],resolve)
    elif format=='pascal_voc':records,annotations,categories,issues=_voc(archive,names,resolve)
    classes=_yolo_classes(archive,metadata) if format in ('yolo_yaml','yolo_unmapped') else None
    counts=Counter();rows=[];objects=[];hashes={};perceptual={}
    near=int(os.getenv('NEAR_DUPLICATE_DISTANCE','0'))
    if not 0<=near<=64:raise IngestionError('NEAR_DUPLICATE_DISTANCE must be between 0 and 64')
    use_near=bool(near) and len(image_names)<=int(os.getenv('MAX_NEAR_DUPLICATE_IMAGES','2000'))
    inferred_labels=folder_labels(image_names) if format=='imagefolder' else {}
    # Missing files declared by a manifest are real missing evidence, not
    # fabricated negative images. Include them in readability accounting.
    keys=image_names+sorted(key for key,value in records.items() if value.get('missing'))
    for image in keys:
        missing=image not in image_set
        if missing:row={'image':image,'readable':False,'width':None,'height':None};phash=None;row['sha256']=None
        else:
            raw=archive.read(image);row,phash=inspect_image(raw,image)
            hashes[image]=hashlib.sha256(raw).hexdigest();row['sha256']=hashes[image]
        if not row['readable']:issues.append({'reference':image,'kind':'unreadable_image'})
        if use_near and phash is not None:perceptual[image]=phash
        row.update(annotation_present=image in records,annotation_valid=True,label=None)
        for part in PurePosixPath(image).parts[:-1]:
            if part.lower() in SPLITS:row['split']=part
        if format=='imagefolder':
            row['label']=inferred_labels.get(image);row['annotation_present']=row['label'] is not None
        elif format in ('yolo_yaml','yolo_unmapped'):
            parts=list(PurePosixPath(image).with_suffix('.txt').parts)
            if 'images' in parts:parts[parts.index('images')]='labels'
            label='/'.join(parts)
            row['annotation_present']=label in names
            if label in names:
                try:
                    text=_small_read(archive,label).decode('utf-8-sig')
                    if classes is None:
                        ids={int(line.split()[0]) for line in text.splitlines() if line.strip()}
                        if any(i<0 or i>10000 for i in ids):raise ValueError('Unsupported YOLO class ID')
                        mapped=[f'class_id:{i}' for i in range(max(ids,default=-1)+1)]
                    else:mapped=classes
                    objects.extend(parse_yolo(text,mapped,image,label))
                except (ValueError,UnicodeError) as exc:
                    row['annotation_valid']=False;issues.append({'reference':label,'kind':'invalid_annotation','reason':str(exc)})
        elif format in ('coco','pascal_voc') and image in records:
            declared=records[image]
            for dimension in ('width','height'):
                value=declared.get('declared_'+dimension)
                if value is not None and row.get(dimension) is not None and float(value)!=row[dimension]:
                    row['annotation_valid']=False
                    issues.append({'reference':image,'kind':'invalid_annotation','reason':'Manifest '+dimension+' differs from decoded image'})
            for obj in annotations.get(image,[]):
                counts['annotation_records']+=1
                try:
                    if not obj.get('class'):raise ValueError('Unknown/missing category mapping')
                    if obj.get('invalid_flags'):raise ValueError('COCO iscrowd/ignore flags must be zero or one')
                    if obj.get('ignored'):
                        counts['ignored_annotations']+=1;continue
                    if row['width'] is None or row['height'] is None:raise ValueError('Decoded image dimensions unavailable')
                    if format=='pascal_voc':
                        base=metadata.get('voc_coordinate_base',1)
                        if base not in (0,1):raise IngestionError('voc_coordinate_base must be 0 or 1')
                        left,top,right,bottom=map(float,obj['voc_box'])
                        box=[left-base,top-base,right-left+base,bottom-top+base]
                        obj={**obj,'pixel_bbox':box}
                    normalized=_normal_box(obj['pixel_bbox'],row['width'],row['height'])
                    objects.append({**obj,'image':image,'bbox':normalized})
                    if obj.get('iscrowd'):counts['crowd_annotations']+=1
                except (ValueError,TypeError) as exc:
                    row['annotation_valid']=False;counts['invalid_target_records']+=1
                    issues.append({'reference':obj['reference'],'kind':'invalid_annotation','reason':str(exc)})
        if not row['annotation_present'] and format not in ('imagefolder','unlabelled'):
            issues.append({'reference':image,'kind':'missing_annotation_file'})
        if len(objects)>int(os.getenv('MAX_ANNOTATED_OBJECTS','1000000')):raise IngestionError('Dataset exceeds MAX_ANNOTATED_OBJECTS')
        rows.append(row)
    if not rows:raise IngestionError('No supported images found')
    from backend.image_ingestion import merge_image_metadata
    frame=merge_image_metadata(archive,names,pd.DataFrame(rows),prefix)
    pairs=[]
    if use_near:
        keys=list(perceptual)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if np.count_nonzero(perceptual[a]!=perceptual[b])<=near:pairs.append((a,b))
    profile=Dataprofiler(frame).profile().to_dict()
    labelled=format=='imagefolder'
    distribution=dict(Counter(frame['label'].dropna())) if labelled else dict(Counter(obj['class'] for obj in objects))
    profile.update(annotation_format=format,image_count=len(frame),object_count=len(objects),class_counts=distribution,issues=issues,
        declared_classes=list(categories.values()) if format=='coco' else classes or sorted(distribution),
        annotation_summary=dict(counts),manifest_errors=sum(i['kind']=='orphan_annotation' for i in issues),
        dataset_kind='image_classification' if labelled else 'unlabelled_images' if format=='unlabelled' else 'object_detection',
        label_source='folder_name_inferred' if labelled else 'unavailable' if format=='unlabelled' else 'annotation_file',
        class_name_mapping='unavailable' if format=='yolo_unmapped' and classes is None else 'available',
        suggested_goal={'task_type':'classification' if labelled else 'object_detection' if format!='unlabelled' else 'anomaly_detection',
            'prediction_target':'label' if labelled else 'object bounding boxes and classes','input_modalities':['image']},
        processing={'archive_access':'streamed_members','image_decode':'all_images','images_per_decode_batch':1},
        near_duplicate_analysis='enabled' if use_near else 'disabled or skipped above the configured pairwise image limit')
    if 'split' in frame:profile['suggested_goal']['split_column']='split'
    if format=='pascal_voc':profile['voc_coordinate_base']=metadata.get('voc_coordinate_base',1)
    return Dataset(name,digest,'image',frame,profile,metadata,objects,issues,hashes,pairs,near if use_near else None)

def load_single_image(name,path,digest,metadata):
    """A standalone image is inspectable without inventing annotations."""
    if path.stat().st_size>20*1024**2:raise IngestionError('A single image exceeds the 20 MiB decode limit')
    row,_=inspect_image(path.read_bytes(),name)
    row.update(sha256=digest,annotation_present=False,annotation_valid=True,label=None)
    frame=pd.DataFrame([row]);profile=Dataprofiler(frame).profile().to_dict()
    profile.update(annotation_format='unlabelled',dataset_kind='unlabelled_images',image_count=1,object_count=0,class_counts={},
        label_source='unavailable',issues=[],suggested_goal={'task_type':'anomaly_detection','input_modalities':['image']})
    return Dataset(name,digest,'image',frame,profile,metadata,image_hashes={name:digest})
