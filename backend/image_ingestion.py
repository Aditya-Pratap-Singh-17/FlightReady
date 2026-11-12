"""Stream YOLO and native VisDrone DET ZIPs one member at a time.

The archive is never extracted, and an uploaded path is never executed. Pixel
coordinates from VisDrone are validated before conversion to normalized boxes.
Ignored regions are recorded separately from trainable target annotations.
"""
from collections import Counter
from pathlib import PurePosixPath
import csv
import hashlib
import io
import logging
import math
import os
import stat
import warnings
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, UnidentifiedImageError
from metrics.dataProfiler import Dataprofiler
from backend.ingestion import Dataset, IngestionError, decode_csv

VISDRONE_CLASSES = {1:'pedestrian',2:'people',3:'bicycle',4:'car',5:'van',6:'truck',7:'tricycle',8:'awning-tricycle',9:'bus',10:'motor'}
IMAGE_SUFFIXES = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif','.jfif','.ppm','.pgm','.pnm','.avif'}
logger = logging.getLogger(__name__)

def validate_members(archive):
    members=archive.infolist()
    maximum=int(os.getenv('MAX_ZIP_MB','4096'))*1024**2
    if len(members)>int(os.getenv('MAX_ZIP_FILES','50000')):
        raise IngestionError('ZIP contains too many entries; raise MAX_ZIP_FILES for a trusted larger dataset')
    total,seen=0,set()
    for member in members:
        path=PurePosixPath(member.filename)
        mode=member.external_attr>>16
        if (path.is_absolute() or '..' in path.parts or '\\' in member.orig_filename or '\x00' in member.orig_filename
                or ':' in member.filename or stat.S_ISLNK(mode) or member.flag_bits & 1):
            raise IngestionError('Unsafe ZIP path, link or encrypted member')
        if member.filename in seen: raise IngestionError('Duplicate ZIP member names')
        seen.add(member.filename)
        total+=member.file_size
        member_limit=20*1024**2 if path.suffix.lower() in IMAGE_SUFFIXES else int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2
        if total>maximum or member.file_size>member_limit:
            raise IngestionError('Archive exceeds decompressed size limits (MAX_ZIP_MB, 20 MiB/image or the record-file limit)')
        if member.file_size>max(member.compress_size,1)*200:
            raise IngestionError('ZIP compression ratio exceeds safety limit')
    return seen,total

def find_layouts(names):
    """Use structural folders rather than trusting the uploaded filename."""
    yolo_roots=sorted(n[:-len('classes.txt')] for n in names if PurePosixPath(n).name=='classes.txt')
    image_roots,annotation_roots=set(),set()
    for name in names:
        parts=PurePosixPath(name).parts
        for folder,collection in [('images',image_roots),('annotations',annotation_roots)]:
            if folder not in parts: continue
            if folder=='images' and PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES: continue
            if folder=='annotations' and not name.lower().endswith('.txt'): continue
            index=parts.index(folder)
            root='/'.join(parts[:index])
            collection.add(root+'/' if root else '')
    roots=[]
    for root in yolo_roots:
        if root in image_roots: roots.append(('yolo',root))
    roots.extend(('visdrone_det',root) for root in image_roots & annotation_roots if root not in yolo_roots)
    roots=sorted(set(roots))
    return roots

def identify_layout(names):
    roots=find_layouts(names)
    if len(roots)!=1:
        raise IngestionError('Expected one YOLO dataset (classes.txt, images/, labels/) or native VisDrone DET dataset (images/, annotations/). Multiple datasets must be uploaded separately; COCO/VID/MOT are not supported.')
    return roots[0]

def inspect_image(raw, image_name):
    row={'image':image_name,'readable':False,'width':None,'height':None}
    perceptual=None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if getattr(im,'n_frames',1)>1:
                    raise IngestionError('Multi-frame/animated images require an explicit frame export; upload static images for this version')
                if im.width*im.height>20_000_000:
                    raise IngestionError('An image exceeds the 20-megapixel decode limit')
                row.update(width=im.width,height=im.height,format=im.format)
                im.load()
                gray=im.convert('L')
                small=np.asarray(gray.resize((9,8)),dtype=int)
                perceptual=small[:,1:]>small[:,:-1]
                row['dhash64']=f'{int("".join("1" if bit else "0" for bit in perceptual.flat),2):016x}'
                row.update(readable=True,brightness=float(ImageStat.Stat(gray).mean[0]),contrast=float(ImageStat.Stat(gray).stddev[0]))
                # Fixed bounded thumbnail: these are scale-dependent proxies,
                # never semantic clutter or a universal blur threshold.
                proxy=gray.copy()
                proxy.thumbnail((256,256))
                pixels=np.asarray(proxy,dtype=np.float64)
                from metrics.uav_metrics.pixels import pixel_features
                row.update({'research_'+key:value for key,value in pixel_features(pixels).items()})
                hist=np.bincount(pixels.astype(np.uint8).ravel(),minlength=256)
                probability=hist[hist>0]/pixels.size
                lap=(-4*pixels[1:-1,1:-1]+pixels[:-2,1:-1]+pixels[2:,1:-1]
                     +pixels[1:-1,:-2]+pixels[1:-1,2:])
                row.update(entropy_bits=float(-(probability*np.log2(probability)).sum()),
                    laplacian_variance=float(lap.var()) if lap.size else None,
                    encoded_bytes=len(raw),encoded_bytes_per_pixel=len(raw)/(im.width*im.height))
    except IngestionError:
        raise
    except (Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise IngestionError('Image decompression bomb rejected') from exc
    except (OSError,UnidentifiedImageError,ValueError):
        pass
    return row,perceptual

def parse_yolo(text,classes,image_name,label_path):
    objects=[]
    for number,line in enumerate(text.splitlines(),1):
        if not line.strip(): continue
        fields=line.split()
        if len(fields)!=5: raise ValueError(f'{label_path}:{number}: expected class and four normalized coordinates')
        cls=int(fields[0]);x,y,w,h=map(float,fields[1:])
        if not (0<=cls<len(classes) and all(math.isfinite(v) for v in (x,y,w,h)) and 0<w<=1 and 0<h<=1
                and x-w/2>=-1e-9 and y-h/2>=-1e-9 and x+w/2<=1+1e-9 and y+h/2<=1+1e-9):
            raise ValueError(f'{label_path}:{number}: invalid class or bounding box')
        objects.append({'image':image_name,'class':classes[cls],'bbox':[x,y,w,h],'reference':f'{label_path}:{number}'})
    return objects


def merge_image_metadata(archive,names,frame,root=''):
    """One metadata convention for all image annotation adapters."""
    path=root+'metadata.csv'
    if path not in names:return frame
    with archive.open(path) as stream:raw=stream.read(8*1024**2+1)
    if len(raw)>8*1024**2:raise IngestionError('Image metadata.csv exceeds 8 MiB')
    text,_=decode_csv(raw)
    extra=pd.read_csv(io.StringIO(text))
    if 'image' not in extra or extra.image.isna().any() or extra.image.duplicated().any() or not set(extra.image).issubset(set(frame.image)):
        raise IngestionError('metadata.csv requires unique image paths matching full archive image paths')
    # Metadata may never replace parser, hash, pixel or annotation evidence.
    overlap=(set(extra.columns)&set(frame.columns))-{'image'}
    if 'split' in overlap:
        # A path-derived split may be restated in metadata, but never contradicted.
        aliases={'training':'train','validation':'val','valid':'val','testing':'test'}
        declared=extra.set_index('image')['split']
        mapped=frame.image.map(declared)
        normalize=lambda value:aliases.get(str(value).lower(),str(value).lower())
        known=frame['split'].notna() & mapped.notna()
        if any(normalize(a)!=normalize(b) for a,b in zip(frame.loc[known,'split'],mapped[known])):
            raise IngestionError('metadata.csv split contradicts the image directory split')
        frame=frame.copy();frame['split']=frame['split'].fillna(mapped)
        extra=extra.drop(columns=['split']);overlap.remove('split')
    if overlap:raise IngestionError('metadata.csv attempts to override computed image evidence')
    return frame.merge(extra,on='image',how='left',validate='one_to_one')

def parse_visdrone(text,image_name,label_path,width,height):
    """Native eight-field ground truth; predictions and ten-field MOT are rejected.

    Official score=0, category=0 and category=11 are excluded from training
    counts. Zero-area ignored regions are diagnostics, not target-box failures.
    Valid target rows remain inspectable even when a separate target row fails.
    """
    objects,issues,counts=[],[],Counter()
    for number,line in enumerate(text.splitlines(),1):
        if not line.strip(): continue
        reference=f'{label_path}:{number}'
        counts['annotation_records']+=1
        try:
            fields=[v.strip() for v in next(csv.reader([line]))]
            if fields and fields[-1]=='': fields.pop() # Some official rows have a trailing comma.
            if len(fields)!=8: raise ValueError('Expected eight DET fields; tracking and prediction files are not ground truth')
            x,y,w,h=map(float,fields[:4])
            score,category,truncation,occlusion=map(int,fields[4:])
            if not all(math.isfinite(v) for v in (x,y,w,h)): raise ValueError('Non-finite box coordinates')
            if category not in range(12) or score not in (0,1) or truncation not in (0,1) or occlusion not in (0,1,2):
                raise ValueError('Invalid ground-truth class, score, truncation or occlusion code')
            ignored=score==0 or category in (0,11)
            if ignored:
                counts['ignored_annotations']+=1
                counts['ignored_regions' if category==0 else 'other_category' if category==11 else 'score_zero_targets']+=1
            geometry=(x>=0 and y>=0 and w>0 and h>0 and (width is None or x+w<=width+1e-9) and (height is None or y+h<=height+1e-9))
            if not geometry:
                if ignored:
                    counts['invalid_ignored_boxes']+=1
                    issues.append({'reference':reference,'kind':'invalid_ignored_box','reason':'Ignored region has invalid bounds; excluded from target counts'})
                    continue
                raise ValueError('Target box has zero/negative area or lies outside the image')
            if ignored: continue
            if width is None or height is None:
                raise ValueError('Image dimensions are unavailable for target box validation')
            objects.append({'image':image_name,'class':VISDRONE_CLASSES[category],
                'bbox':[(x+w/2)/width,(y+h/2)/height,w/width,h/height],
                'pixel_bbox':[x,y,w,h],'score':score,'category_id':category,'truncation':truncation,'occlusion':occlusion,'reference':reference})
        except (ValueError,UnicodeError,csv.Error) as exc:
            counts['invalid_target_records']+=1
            issues.append({'reference':reference,'kind':'invalid_annotation','reason':str(exc)})
    return objects,issues,counts

def load_image_archive(name,source,digest,metadata):
    # source is a staged filesystem path for large uploads, not a bytes copy.
    from backend.dataset_archives import open_archive
    from contextlib import nullcontext
    with (nullcontext(source) if hasattr(source,'infolist') else open_archive(source,name)) as archive:
        names,expanded=validate_members(archive)
        selected=[(fmt,root) for fmt,root in find_layouts(names)
            if metadata.get('image_root')==root and metadata.get('annotation_format')==fmt]
        format,root=selected[0] if len(selected)==1 else identify_layout(names)
        classes=list(VISDRONE_CLASSES.values()) if format=='visdrone_det' else archive.read(root+'classes.txt').decode('utf-8-sig').splitlines()
        if not classes or any(not c.strip() for c in classes) or len(set(classes))!=len(classes):
            raise IngestionError('classes.txt must list unique nonempty class names')
        image_prefix=root+'images/'
        image_names=sorted(n for n in names if n.startswith(image_prefix) and PurePosixPath(n).suffix.lower() in IMAGE_SUFFIXES)
        stems=[str(PurePosixPath(n).relative_to(image_prefix).with_suffix('')) for n in image_names]
        if len(stems)!=len(set(stems)):
            raise IngestionError('Different images map to the same annotation file; image stems must be unique')
        extra=pd.DataFrame()
        metadata_path=root+'metadata.csv'
        if metadata_path in names:
            with archive.open(metadata_path) as stream:raw_metadata=stream.read(8*1024**2+1)
            if len(raw_metadata)>8*1024**2:raise IngestionError('Image metadata.csv exceeds 8 MiB')
            text,_=decode_csv(raw_metadata)
            extra=pd.read_csv(io.StringIO(text))
            if 'image' not in extra or extra['image'].duplicated().any() or not set(extra['image']).issubset(set(image_names)):
                raise IngestionError('metadata.csv requires unique image paths matching full archive image paths')
            reserved={'readable','width','height','format','brightness','contrast','entropy_bits','laplacian_variance','encoded_bytes','encoded_bytes_per_pixel','annotation_present','annotation_valid','sha256','ignored_annotations'}
            if reserved & set(extra.columns): raise IngestionError('metadata.csv attempts to override computed image evidence')
        rows,objects,issues=[],[],[]
        hashes,perceptual={},{}
        totals=Counter()
        near_distance=int(os.getenv('NEAR_DUPLICATE_DISTANCE','0'))
        if not 0<=near_distance<=64: raise IngestionError('NEAR_DUPLICATE_DISTANCE must be between 0 and 64')
        near_enabled=bool(near_distance) and len(image_names)<=int(os.getenv('MAX_NEAR_DUPLICATE_IMAGES','2000'))
        for index,image_name in enumerate(image_names,1):
            raw=archive.read(image_name)
            hashes[image_name]=hashlib.sha256(raw).hexdigest()
            row,perceptual_hash=inspect_image(raw,image_name)
            row['sha256']=hashes[image_name]
            if near_enabled and perceptual_hash is not None: perceptual[image_name]=perceptual_hash
            if not row['readable']: issues.append({'reference':image_name,'kind':'unreadable_image'})
            relative=PurePosixPath(image_name).relative_to(image_prefix).with_suffix('.txt')
            label_path=root+('annotations/' if format=='visdrone_det' else 'labels/')+str(relative)
            row.update(annotation_present=label_path in names,annotation_valid=True)
            if label_path not in names:
                issues.append({'reference':image_name,'kind':'missing_annotation_file'})
            else:
                try:
                    text=archive.read(label_path).decode('utf-8-sig')
                    if format=='yolo':
                        objects.extend(parse_yolo(text,classes,image_name,label_path))
                    else:
                        parsed,annotation_issues,counts=parse_visdrone(text,image_name,label_path,row['width'],row['height'])
                        objects.extend(parsed);issues.extend(annotation_issues);totals.update(counts)
                        row['annotation_valid']=counts['invalid_target_records']==0
                        row['ignored_annotations']=counts['ignored_annotations']
                except (ValueError,UnicodeError,csv.Error) as exc:
                    row['annotation_valid']=False
                    issues.append({'reference':label_path,'kind':'invalid_annotation','reason':str(exc)})
            if len(objects)>int(os.getenv('MAX_ANNOTATED_OBJECTS','1000000')):
                raise IngestionError('Dataset exceeds MAX_ANNOTATED_OBJECTS; increase deliberately if enough RAM is available')
            rows.append(row)
            if index%500==0 or index==len(image_names): logger.info('Validated %s / %s images',index,len(image_names))
        frame=pd.DataFrame(rows)
        # Official native wrapper declares partition role, not capture conditions.
        # Preserve it so a validation-only download can never become training data.
        if format == 'visdrone_det':
            wrapper = PurePosixPath(root.rstrip('/')).name.lower()
            role = next((role for suffix, role in (('-train', 'train'), ('-val', 'val'), ('-test-dev', 'test'), ('-test-challenge', 'test'))
                         if wrapper.endswith(suffix)), None)
            if role:
                frame['split'] = role
        if len(extra):
            if 'split' in frame and 'split' in extra:
                declared = extra.set_index('image')['split'].dropna()
                if any(str(v).lower() not in (role, 'validation' if role == 'val' else role, 'training' if role == 'train' else role) for v in declared):
                    raise IngestionError('metadata.csv split contradicts the official dataset wrapper role')
                extra = extra.drop(columns=['split'])
            frame=frame.merge(extra,on='image',how='left',validate='one_to_one')
        pairs=[]
        if near_enabled:
            keys=list(perceptual)
            for i,a in enumerate(keys):
                for b in keys[i+1:]:
                    if np.count_nonzero(perceptual[a]!=perceptual[b])<=near_distance: pairs.append((a,b))
        profile=Dataprofiler(frame).profile().to_dict()
        counts=Counter(o['class'] for o in objects)
        profile.update(image_count=len(frame),object_count=len(objects),class_counts={c:counts[c] for c in classes},issues=issues,
            annotation_format=format,archive_root=root,expanded_bytes=expanded,annotation_summary=dict(totals),
            dataset_kind='object_detection',class_name_mapping='available',
            suggested_goal={'task_type':'object_detection','prediction_target':'object bounding boxes and classes','input_modalities':['image']},
            processing={'archive_access':'streamed_members','image_decode':'all_images','images_per_decode_batch':1},
            near_duplicate_analysis='enabled' if near_enabled else 'skipped: image count exceeds configured pairwise limit' if near_distance else 'disabled')
        if format=='visdrone_det':
            if 'split' in frame:
                profile['source_split'] = str(frame['split'].iloc[0])
            profile['annotation_policy']='Only score=1 categories 1–10 contribute target objects. Ignored regions/category 11 are separate; valid target rows are retained when another row is invalid. Pedestrian and people remain separate classes.'
        return Dataset(name,digest,'image',frame,profile,metadata,objects,issues,hashes,pairs,near_distance if near_enabled else None)
