"""Safely discover datasets inside downloaded archives without running code.

Ambiguous collections stay staged until the user chooses a dataset. Train/test
files are not silently concatenated. Tar members are read, never extracted.
"""
from contextlib import contextmanager
from pathlib import Path,PurePosixPath
from tempfile import TemporaryDirectory
import bz2
import gzip
import hashlib
import io
import lzma
import os
import stat
import tarfile
import zipfile
import pandas as pd
from backend.ingestion import Dataset,IngestionError,SelectionRequired,decode_csv

ARCHIVE_SUFFIXES={'.zip','.tar','.tgz','.gz','.bz2','.xz','.tbz2','.txz'}

class TarAdapter:
    def __init__(self,source):
        self.archive=tarfile.open(fileobj=source,mode='r:*') if hasattr(source,'read') else tarfile.open(source,mode='r:*')
        self.members={};self.infos=[]
        try:
            for member in self.archive:
                if len(self.infos)>=int(os.getenv('MAX_ZIP_FILES','50000')):raise IngestionError('Archive has too many entries')
                if not (member.isfile() or member.isdir()):raise IngestionError('TAR links, devices and special files are not supported')
                if member.name in self.members:raise IngestionError('Duplicate TAR member names')
                info=zipfile.ZipInfo('placeholder');info.filename=member.name;info.orig_filename=member.name
                info.file_size=member.size;info.compress_size=member.size
                info.external_attr=((stat.S_IFDIR if member.isdir() else stat.S_IFREG)|member.mode)<<16
                self.members[member.name]=member;self.infos.append(info)
        except Exception:self.archive.close();raise
    def infolist(self):return self.infos
    def open(self,name):
        stream=self.archive.extractfile(self.members[name])
        if stream is None:raise IngestionError('Requested archive member is not a file')
        return stream
    def read(self,name):
        with self.open(name) as stream:return stream.read()
    def close(self):self.archive.close()

@contextmanager
def open_archive(source,name='dataset.zip'):
    archive=zipfile.ZipFile(source) if Path(name).suffix.lower()=='.zip' else TarAdapter(source)
    try:yield archive
    finally:archive.close()

def is_auxiliary(name):
    path=PurePosixPath(name)
    if any(part in {'.git','__MACOSX','node_modules','.venv','__pycache__'} or part.startswith('.') for part in path.parts):return True
    lower=path.name.lower()
    return lower.startswith(('readme','license','copying','notice')) or lower in {'package.json','package-lock.json','tsconfig.json','metadata.csv','classes.txt'}

def discover(archive,names,metadata):
    from backend.record_formats import RECORD_SUFFIXES,TEXT_SUFFIXES
    from backend.universal_images import image_choices
    images=image_choices(archive,names,metadata)
    choices=list(images)
    texts=[]
    image_annotations={choice['options'].get('annotation_file') for choice in images}
    from backend.image_ingestion import IMAGE_SUFFIXES
    image_stems={str(PurePosixPath(name).with_suffix('')) for name in names if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES}
    has_yolo=any(choice['options'].get('annotation_format','').startswith('yolo') for choice in images)
    for name in sorted(names):
        path=PurePosixPath(name)
        if is_auxiliary(name) or path.suffix.lower() not in RECORD_SUFFIXES or name.endswith('/'):continue
        if images and (name in image_annotations or any(p.lower() in {'annotations','labels','annotations_xml'} for p in path.parts)):continue
        # A side-by-side YOLO annotation is part of the image dataset, not a
        # competing text corpus. Keep unrelated text files available to select.
        if has_yolo and path.suffix.lower()=='.txt' and str(path.with_suffix('')) in image_stems:continue
        if path.suffix.lower() in TEXT_SUFFIXES:texts.append(name)
        else:choices.append({'label':'Records: '+name,'options':{'dataset_member':name}})
    if len(texts)>1:
        choices.append({'label':f'Text corpus ({len(texts)} files; one document per file)','options':{'collection_type':'text_corpus'}})
    elif len(texts)==1:choices.append({'label':'Text: '+texts[0],'options':{'dataset_member':texts[0]}})
    return choices,texts

def _expanded_compressed(name,path,metadata,digest):
    from backend.large_ingestion import load_dataset_stream,ROOT,CHUNK_BYTES
    suffix=Path(name).suffix.lower()
    opener={'.gz':gzip.open,'.bz2':bz2.open,'.xz':lzma.open}.get(suffix)
    if not opener:raise IngestionError('Unsupported or corrupt TAR archive')
    inner=name[:-len(suffix)]
    if not Path(inner).suffix:inner+='.txt'
    # load_dataset_stream enforces the same expanded upload limit while reading.
    with opener(path,'rb') as data:
        ds=load_dataset_stream(inner,data,metadata)
    ds.profile['compressed_source']={'name':name,'format':suffix.lstrip('.'),'sha256':digest}
    return ds

def load_container(name,path,digest,metadata):
    from backend.image_ingestion import validate_members,load_image_archive
    from backend.large_ingestion import ROOT,load_dataset_stream,load_record_table,CHUNK_BYTES
    from backend.universal_images import load_universal_images
    depth=metadata.get('_archive_depth',0)
    if type(depth) is not int or depth<0 or depth>=3:raise IngestionError('Nested archive depth exceeds the supported limit; upload the actual dataset files')
    child_metadata={key:value for key,value in metadata.items() if key not in {'dataset_member','annotation_file','image_root','annotation_format','class_file','legacy_layout','collection_type'}}
    child_metadata['_archive_depth']=depth+1
    if Path(name).suffix.lower()!='.zip':
        try:
            if not tarfile.is_tarfile(path):return _expanded_compressed(name,path,child_metadata,digest)
        except (tarfile.TarError,OSError):return _expanded_compressed(name,path,child_metadata,digest)
    with open_archive(path,name) as archive:
        names,expanded=validate_members(archive)
        if expanded>max(Path(path).stat().st_size,1)*200:raise IngestionError('Archive expansion ratio exceeds safety limit')
        member=metadata.get('dataset_member')
        if member:
            if member not in names or member.endswith('/'):raise IngestionError('dataset_member must be an existing archive file path')
            with archive.open(member) as data:ds=load_dataset_stream(member,data,child_metadata)
            ds.profile['source_archive']={'name':name,'sha256':digest,'member':member}
            return ds
        choices,texts=discover(archive,names,metadata)
        selected=None
        if metadata.get('collection_type')=='text_corpus':
            selected=next((c for c in choices if c['options'].get('collection_type')=='text_corpus'),None)
        elif metadata.get('annotation_format'):
            wanted=metadata['annotation_format']
            matches=[c for c in choices if c['options'].get('annotation_format')==wanted
                and (not metadata.get('annotation_file') or c['options'].get('annotation_file')==metadata['annotation_file'])
                and (metadata.get('image_root') is None or c['options'].get('image_root','')==metadata['image_root'])]
            if len(matches)==1:selected=matches[0]
        elif len(choices)==1:selected=choices[0]
        if selected is None:
            if not choices:raise IngestionError('No supported dataset found in this archive. Source code is not executed; upload data files or a documented image layout.')
            raise SelectionRequired('This archive contains multiple datasets. Choose what to assess; train/test files are kept separate.',choices[:500])
        options={**metadata,**selected['options']}
        if options.get('dataset_member'):
            member=options['dataset_member']
            with archive.open(member) as data:ds=load_dataset_stream(member,data,child_metadata)
            ds.profile['source_archive']={'name':name,'sha256':digest,'member':member}
            return ds
        if options.get('collection_type')=='text_corpus':
            owner=TemporaryDirectory(prefix='corpus-',dir=ROOT)
            parsing={'format':'text_corpus','record_unit':'document','warnings':['Folder labels, where present, are inferred from file organization; verify their meaning.'],'json_encoded_fields':[]}
            def records():
                for member in texts:
                    # Bound the read before allocating a document, including for
                    # compressed archive members with very large decoded sizes.
                    with archive.open(member) as stream:raw=stream.read(8*1024**2+1)
                    if len(raw)>8*1024**2:raise IngestionError('A corpus document exceeds 8 MiB; select it individually with line records')
                    text,_=decode_csv(raw,metadata.get('csv_encoding'))
                    parent=PurePosixPath(member).parent.name
                    label=None if parent.lower() in {'','data','dataset','texts','documents','train','test','val','validation'} else parent
                    yield {'text':text or None,'label':label,'source_file':member}
            try:
                ds=load_record_table(name,Path(owner.name)/'corpus.txt',digest,options,owner,Path(path).stat().st_size,
                    prepared=(records(),parsing))
            except Exception:owner.cleanup();raise
            ds.profile['source_archive']={'name':name,'sha256':digest,'collection':'text_corpus'}
            return ds
        if options.get('legacy_layout'):
            return load_image_archive(name,path,digest,options)
        return load_universal_images(name,archive,names,digest,options)
