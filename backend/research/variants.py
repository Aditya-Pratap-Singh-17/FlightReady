"""Reproducible, train-only interventions for the documented YOLO ZIP format.

Selection restrictions operate at whole-image level so retained annotations stay
complete relative to their parent. Evaluation bytes and annotations are immutable.
No input is overwritten; each invocation requires a new output directory.
"""
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
import hashlib
import io
import json
import math
import random
import platform
import numpy as np
import PIL
import zipfile
import pandas as pd
from PIL import Image,ImageFilter,ImageEnhance
from backend.ingestion import load_dataset
from backend.image_ingestion import validate_members,parse_yolo
from backend.context import Model
from pydantic import Field

class VariantConfig(Model):
    seed: int
    intervention: str
    parameters: dict = Field(default_factory=dict)
    target_train_images: int | None = Field(default=None,ge=1)
    split_column: str = 'split'
    group_column: str | None = None
    acknowledge_missing_groups: bool = False

INTERVENTIONS={'annotation_noise','object_scale','redundancy','class_representation',
               'environment','viewpoint','shortcut','image_degradation','control',
               'resolution','label_omission','box_perturbation','small_object_reduction','negative_coverage'}

def sha(raw):return hashlib.sha256(raw).hexdigest()

def label_path(image):
    relative=PurePosixPath(image).relative_to('images').with_suffix('.txt')
    return 'labels/'+str(relative)

def assign_split(image,row,c):
    if c.split_column in row and pd.notna(row[c.split_column]):value=str(row[c.split_column]).lower()
    else:value=PurePosixPath(image).parts[1].lower() if len(PurePosixPath(image).parts)>2 else ''
    aliases={'training':'train','validation':'val','valid':'val','dev':'val','testing':'test'}
    value=aliases.get(value,value)
    if value not in ('train','val','test'):raise ValueError('Every image needs an explicit train/val/test assignment')
    return value

def create_variant(parent_path,output_dir,config:VariantConfig):
    if config.intervention not in INTERVENTIONS:raise ValueError('Unsupported intervention')
    parent_path=Path(parent_path).resolve();out=Path(output_dir).resolve()
    if out.exists():raise ValueError('Output directory already exists; choose a new directory')
    if out==parent_path or parent_path.is_relative_to(out):raise ValueError('Output must not contain the parent dataset')
    # This research utility deliberately supports only canonical YOLO ZIPs.
    with zipfile.ZipFile(parent_path) as source:
        names,_=validate_members(source)
        if 'classes.txt' not in names:raise ValueError('Research variants require canonical root classes.txt, images/, labels/ YOLO ZIP')
        def small_read(name,limit=8*1024**2):
            with source.open(name) as member:raw=member.read(limit+1)
            if len(raw)>limit:raise ValueError('Research text member exceeds its bounded read limit')
            return raw
        classes=small_read('classes.txt',1024**2).decode('utf-8-sig').splitlines()
        if not classes or len(classes)!=len(set(classes)) or any(not c.strip() for c in classes):raise ValueError('Invalid class mapping')
        images=sorted(n for n in names if n.startswith('images/') and PurePosixPath(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'})
        meta=pd.DataFrame()
        if 'metadata.csv' in names:
            if source.getinfo('metadata.csv').file_size>8*1024**2:raise ValueError('Metadata exceeds 8 MiB')
            meta=pd.read_csv(io.BytesIO(source.read('metadata.csv')))
            if 'image' not in meta or meta.image.duplicated().any() or not set(meta.image).issubset(set(images)):raise ValueError('Invalid image metadata')
        metadata=meta.set_index('image').to_dict('index') if len(meta) else {}
        records={};groups=defaultdict(set)
        for image in images:
            row=metadata.get(image,{})
            split=assign_split(image,row,config);label=label_path(image)
            if label not in names:raise ValueError('Variant parent requires explicit annotation files, including negatives')
            objects=parse_yolo(small_read(label).decode('utf-8-sig'),classes,image,label)
            records[image]={'split':split,'label':label,'objects':objects,'metadata':row}
            if config.group_column:
                group=row.get(config.group_column)
                if group is None or pd.isna(group):raise ValueError('Complete group IDs required')
                groups[str(group)].add(split)
        if not records:raise ValueError('No images in parent')
        if config.group_column and any(len(splits)>1 for splits in groups.values()):raise ValueError('Flight/session/source group occurs in multiple splits')
        if not config.group_column and not config.acknowledge_missing_groups:
            raise ValueError('Declare group_column, or explicitly acknowledge that group independence cannot be verified')
        if not any(v['split']=='val' for v in records.values()) or not any(v['split']=='test' for v in records.values()):
            raise ValueError('Independent validation and test partitions are required for research variants')
        # Reject known content overlap before creating any output.
        by_hash=defaultdict(set)
        for image,record in records.items():by_hash[sha(source.read(image))].add(record['split'])
        if any(len(splits)>1 for splits in by_hash.values()):raise ValueError('Exact image content overlaps training/evaluation partitions')
        rng=random.Random(config.seed);p=config.parameters;train=[im for im,v in records.items() if v['split']=='train']
        eligible=list(train)
        if config.intervention=='object_scale':
            low,high=p['relative_area_range']
            if not 0<=low<high<=1:raise ValueError('Invalid relative occupancy interval')
            eligible=[im for im in train if any(low<=o['bbox'][2]*o['bbox'][3]<high for o in records[im]['objects'])]
        if config.intervention=='class_representation':
            selected=p['classes']
            if not set(selected).issubset(classes):raise ValueError('Unknown class')
            eligible=[im for im in train if any(o['class'] in selected for o in records[im]['objects'])]
            retain_other=float(p.get('retain_other_fraction',0))
            if not math.isfinite(retain_other) or not 0<=retain_other<=1:raise ValueError('Retained other-class fraction must be 0..1')
            others=sorted(set(train)-set(eligible))
            eligible.extend(rng.sample(others,round(len(others)*retain_other)))
        if config.intervention in ('environment','viewpoint'):
            c=p['column'];allowed=p['allowed_values']
            if any(c not in records[im]['metadata'] or pd.isna(records[im]['metadata'][c]) for im in train):raise ValueError('Complete condition metadata required')
            eligible=[im for im in train if records[im]['metadata'][c] in allowed]
        if config.intervention=='shortcut':
            # Controlled selection, not a claim that the downstream model uses it.
            c=p['column'];mapping=p['class_condition_map'];eligible=[]
            if not set(mapping).issubset(classes):raise ValueError('Unknown shortcut class')
            for im in train:
                value=records[im]['metadata'].get(c)
                if value is None or pd.isna(value):raise ValueError('Complete nuisance metadata required')
                targets={o['class'] for o in records[im]['objects']}&set(mapping)
                if targets and all(value==mapping[cls] for cls in targets):eligible.append(im)
        if config.intervention in ('small_object_reduction','negative_coverage'):
            retain=float(p['retain_fraction']);selected_classes=p.get('classes',classes)
            if not math.isfinite(retain) or not 0<=retain<=1 or not set(selected_classes).issubset(classes) or not selected_classes:
                raise ValueError('Select known classes and a finite retained fraction in [0,1]')
            restricted=[]
            for im in train:
                objects=records[im]['objects']
                if config.intervention=='small_object_reduction':
                    maximum=float(p['maximum_native_area'])
                    if not math.isfinite(maximum) or maximum<=0:raise ValueError('Explicit positive native-area criterion required')
                    with Image.open(io.BytesIO(source.read(im))) as image:
                        if image.width*image.height>20_000_000:raise ValueError('Image decode limit exceeded')
                        qualifies=any(o['class'] in selected_classes and o['bbox'][2]*o['bbox'][3]*image.width*image.height<maximum for o in objects)
                else:
                    flag=records[im]['metadata'].get(p['column'])
                    if flag is None or pd.isna(flag) or str(flag).lower() not in ('true','false','1','0'):
                        raise ValueError('Complete explicit boolean hard-negative candidate metadata required')
                    qualifies=str(flag).lower() in ('true','1') and not any(o['class'] in selected_classes for o in objects)
                if qualifies:restricted.append(im)
            if not restricted:raise ValueError('No training images meet the explicit intervention criteria')
            eligible=sorted(set(train)-set(restricted))+rng.sample(sorted(restricted),round(len(restricted)*retain))
        target=config.target_train_images or len(eligible)
        if not eligible:raise ValueError('Intervention selected no training images')
        if config.intervention=='redundancy':
            unique=int(p['unique_images'])
            if not 1<=unique<=min(target,len(eligible)):raise ValueError('Invalid unique_images for redundancy')
            chosen=rng.sample(eligible,unique)
            selected=chosen+[rng.choice(chosen) for _ in range(target-unique)]
        else:
            if target>len(eligible):raise ValueError('Cannot match requested training size without implicit duplication')
            selected=rng.sample(eligible,target)
        selected.sort();counts=Counter();copies=[];mods=[]
        for im in selected:
            count=counts[im];counts[im]+=1
            if not count:dest=im
            else:
                path=PurePosixPath(im);dest=str(path.with_name(path.stem+f'__repeat_{count}'+path.suffix))
                if dest in names:raise ValueError('Generated duplicate name collides with parent')
            copies.append((im,dest))
        retained={im for im,v in records.items() if v['split']!='train'}|set(selected)
        dropped=sorted(set(train)-set(selected));evaluation_hashes={}
        training_labels={records[im]['label'] for im in train}
        training_images=set(train)
        evaluation_labels={v['label'] for v in records.values() if v['split']!='train'}
        # Exclusive output directory creation prevents clobbering research data.
        out.mkdir(parents=True,exist_ok=False)
        try:
            with zipfile.ZipFile(out/'dataset.zip','x',compression=zipfile.ZIP_STORED) as archive:
                def write(name,raw):
                    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
                    archive.writestr(info,raw)
                for name in sorted(names):
                    if name.endswith('/') or name=='metadata.csv':continue
                    if name in training_images or name in training_labels:continue
                    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
                    digest=hashlib.sha256()
                    with source.open(name) as member,archive.open(info,'w') as destination:
                        while block:=member.read(1024**2):digest.update(block);destination.write(block)
                    if name in records and records[name]['split']!='train' or name in evaluation_labels:
                        evaluation_hashes[name]=digest.hexdigest()
                new_metadata=[{'image':im,**records[im]['metadata'],config.split_column:records[im]['split']} for im in sorted(retained) if records[im]['split']!='train']
                evaluation_images={digest for name,digest in evaluation_hashes.items() if name in records}
                for original,dest in copies:
                    record=records[original];raw=source.read(original);labels=small_read(record['label']);old_image_hash=sha(raw);old_label_hash=sha(labels)
                    if config.intervention=='annotation_noise':
                        probability=float(p['probability'])
                        if not 0<=probability<=1 or len(classes)<2:raise ValueError('Label noise needs probability 0..1 and at least two classes')
                        lines=[]
                        for line in labels.decode('utf-8-sig').splitlines():
                            if not line.strip():continue
                            fields=line.split();cls=int(fields[0])
                            if rng.random()<probability:fields[0]=str(rng.choice([i for i in range(len(classes)) if i!=cls]))
                            lines.append(' '.join(fields))
                        labels=('\n'.join(lines)+'\n' if lines else '').encode()
                    if config.intervention=='image_degradation':
                        radius=float(p.get('blur_radius',0));brightness=float(p.get('brightness_factor',1))
                        sigma=float(p.get('noise_sigma',0))
                        if not math.isfinite(radius) or not math.isfinite(brightness) or not math.isfinite(sigma) or not 0<=radius<=20 or not 0<=brightness<=3 or not 0<=sigma<=64:raise ValueError('Invalid degradation parameters')
                        from backend.image_ingestion import inspect_image
                        inspected,_=inspect_image(raw,original)
                        if not inspected['readable']:raise ValueError('Cannot degrade an unreadable image')
                        with Image.open(io.BytesIO(raw)) as im:
                            image=ImageEnhance.Brightness(im.convert('RGB')).enhance(brightness).filter(ImageFilter.GaussianBlur(radius))
                            if sigma:
                                noise_seed=int(hashlib.sha256(f'{config.seed}:{original}'.encode()).hexdigest()[:16],16)
                                noise=np.random.default_rng(noise_seed).normal(0,sigma,(*np.asarray(image).shape[:2],1))
                                image=Image.fromarray(np.clip(np.rint(np.asarray(image,dtype=float)+noise),0,255).astype(np.uint8))
                            encoded=io.BytesIO();image.save(encoded,format=im.format);raw=encoded.getvalue()
                    if config.intervention=='resolution':
                        scale=float(p.get('scale',1))
                        if not math.isfinite(scale) or not .05<=scale<=1:raise ValueError('Resolution scale must be .05..1')
                        with Image.open(io.BytesIO(raw)) as im:
                            reduced=im.convert('RGB').resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
                            encoded=io.BytesIO();reduced.save(encoded,format=im.format);raw=encoded.getvalue()
                    if config.intervention in ('label_omission','box_perturbation'):
                        severity=float(p.get('severity',0))
                        if not math.isfinite(severity) or not 0<=severity<=1:raise ValueError('Severity must be 0..1')
                        lines=[]
                        for line in labels.decode('utf-8-sig').splitlines():
                            if not line.strip():continue
                            if config.intervention=='label_omission':
                                if rng.random()>=severity:lines.append(line)
                            else:
                                fields=line.split();cx,cy,w,h=map(float,fields[1:])
                                cx=min(1-w/2,max(w/2,cx+rng.uniform(-1,1)*severity*w))
                                cy=min(1-h/2,max(h/2,cy+rng.uniform(-1,1)*severity*h))
                                lines.append(' '.join([fields[0],str(cx),str(cy),str(w),str(h)]))
                        labels=('\n'.join(lines)+'\n' if lines else '').encode()
                    if sha(raw) in evaluation_images:raise ValueError('Intervention created exact image overlap with immutable evaluation images')
                    write(dest,raw);write(label_path(dest),labels)
                    mods.append({'source':original,'destination':dest,'parent_image_sha256':old_image_hash,'image_sha256':sha(raw),
                        'parent_annotation_sha256':old_label_hash,'annotation_sha256':sha(labels),'intervention':config.intervention,
                        'parent_object_count':len(record['objects']),'output_object_count':len([line for line in labels.decode('utf-8-sig').splitlines() if line.strip()])})
                    new_metadata.append({'image':dest,**record['metadata'],config.split_column:'train'})
                write('metadata.csv',pd.DataFrame(new_metadata).to_csv(index=False).encode())
            parent_hash=_file_sha(parent_path);variant_hash=_file_sha(out/'dataset.zip')
            manifest={'schema_version':'1.0','variant_id':variant_hash,'parent_dataset_id':parent_hash,
                'implementation_version':'interventions-2.0','software':{'python':platform.python_version(),'numpy':np.__version__,'pillow':PIL.__version__,'pandas':pd.__version__},
                'config':config.model_dump(mode='json'),'modified_property':config.intervention,'modifications':mods,'removed_training_images':dropped,
                'evaluation_sha256':evaluation_hashes,'training_images':len(selected),'unique_training_source_images':len(counts),
                'group_independence':'verified declared IDs' if groups else 'unverified, explicitly acknowledged',
                'matched_budget_note':'Use equal training steps/epochs and target_train_images across comparisons; identical counts do not imply equal independent evidence.'}
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            return manifest
        except Exception:
            # Keep a partial directory for inspection; never publish a manifest
            # representing incomplete work as a successful experiment.
            raise

def _file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        while block:=source.read(1024**2):h.update(block)
    return h.hexdigest()
