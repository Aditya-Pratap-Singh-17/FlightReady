"""Bounded-memory upload staging and disk-backed tabular ingestion.

Only a preview becomes a pandas DataFrame. All decision metrics query the full
DuckDB table; they must never evaluate this preview as the full dataset.
"""
from pathlib import Path
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from typing import Protocol, cast
import codecs
import csv
import hashlib
import io
import math
import os
import re
import shutil
import zipfile
import tarfile
import ijson
from defusedxml.common import DefusedXmlException
from openpyxl.utils.exceptions import InvalidFileException
from xml.parsers.expat import ExpatError
import duckdb
from sql.datasets import connect
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from backend.ingestion import Dataset, IngestionError, SelectionRequired, decode_csv, load_dataset

ROOT = Path(__file__).resolve().parents[1] / '.runtime' / 'datasets'
CHUNK_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
csv.field_size_limit(MAX_RECORD_BYTES)
# Match pandas' default CSV NA tokens so small and disk-backed inputs agree.
NA_VALUES = {'', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'}
CONTROLS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

class ByteSource(Protocol):
    def read(self,size:int=-1,/)->bytes: ...

def detect_encoding(path: Path, metadata: dict) -> dict:
    with path.open('rb') as file:
        head = file.read(4)
    declared = metadata.get('csv_encoding')
    has_bom = head.startswith((codecs.BOM_UTF8, codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE))
    if declared is not None or has_bom:
        # Decode just a BOM, avoiding splitting a multibyte character in a probe.
        probe = head[:4] if head.startswith((codecs.BOM_UTF32_LE,codecs.BOM_UTF32_BE)) else head[:2] if head.startswith((codecs.BOM_UTF16_LE,codecs.BOM_UTF16_BE)) else codecs.BOM_UTF8 if has_bom else b''
        _, info = decode_csv(probe if has_bom else b'', declared)
        return info
    try:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')
        with path.open('rb') as file:
            while block := file.read(CHUNK_BYTES):
                decoder.decode(block)
            decoder.decode(b'', final=True)
        return {'encoding':'utf-8','encoding_source':'strict_utf8','delimiter':',','warnings':[]}
    except UnicodeDecodeError:
        return {'encoding':'cp1252','encoding_source':'legacy_fallback','delimiter':',',
                'warnings':['Strict UTF-8 decoding failed; Windows-1252 fallback used. Confirm text or declare csv_encoding.']}

class BoundedLines:
    """Limit physical lines and the total decoded size of a multiline record."""
    def __init__(self, text):
        self.text, self.record_size = text, 0
    def __iter__(self): return self
    def __next__(self):
        line = self.text.readline(MAX_RECORD_BYTES + 1)
        if not line: raise StopIteration
        self.record_size += len(line.encode('utf-8'))
        if self.record_size > MAX_RECORD_BYTES:
            raise IngestionError('A CSV record exceeds the 8 MiB decoded record limit')
        if CONTROLS.search(line):
            raise IngestionError('CSV contains binary NUL bytes or unsupported control characters')
        return line

def csv_batches(path: Path, metadata: dict):
    from backend.record_formats import delimiter_for
    info = detect_encoding(path, metadata)
    with path.open('r', encoding=info['encoding'], errors='strict', newline='') as text:
        delimiter,origin=delimiter_for(text.read(65536),metadata,path.suffix.lower())
        text.seek(0)
        info.update(delimiter=delimiter,delimiter_source=origin,has_header=metadata.get('has_header',True))
        if not isinstance(info['has_header'],bool):raise IngestionError('has_header must be true or false')
        lines = BoundedLines(text)
        reader = csv.reader(lines, strict=True,delimiter=delimiter)
        first = next(reader, None)
        header = first if info['has_header'] else metadata.get('column_names') or [f'column_{i+1}' for i in range(len(first or []))]
        lines.record_size = 0
        if not header or len(header) > 512 or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise IngestionError('CSV needs 1–512 nonempty, unique column names')
        yield header, info
        rows, size = [], 0
        if not info['has_header'] and first:
            if len(first)!=len(header):raise IngestionError('column_names does not match the first record width')
            rows.append([None if value in NA_VALUES else value for value in first])
        for row in reader:
            record_size = lines.record_size
            lines.record_size = 0
            if not row: continue
            if len(row) != len(header):
                raise IngestionError(f'CSV record ending at line {reader.line_num} has the wrong number of fields')
            # NA normalization is explicit; no malformed record is skipped.
            rows.append([None if value in NA_VALUES else value for value in row])
            size += record_size
            if len(rows) >= MAX_BATCH_ROWS or size >= MAX_RECORD_BYTES:
                yield rows
                rows, size = [], 0
        if rows: yield rows

def forced_text_fields(parsing):
    typed=parsing.get('format') in ('json','jsonl','xlsx','feather','arrow')
    forced={c for c,types in parsing.get('source_type_counts',{}).items() if typed and set(types)-{'null'}=={'string'}}
    forced.update(parsing.get('json_encoded_fields',[]));forced.update(parsing.get('exact_integer_fields',[]))
    if parsing.get('format') in ('plain_text','text_corpus'):forced.add('text')
    return forced

def load_record_table(name,path,digest,metadata,owner,source_bytes,*,prepared=None):
    """Evolving JSON/XML fields become nullable columns, never dropped keys."""
    from backend.record_formats import prepare_records,MAX_RECORD
    iterator,parsing=prepared if prepared is not None else prepare_records(path,metadata)
    database=str(Path(owner.name)/'dataset.duckdb')
    columns=[];count=0;batch=[];batch_bytes=0
    try:
        with connect(database,False) as connection:
            connection.execute('CREATE TABLE raw_records (__rowid BIGINT)')
            def flush():
                nonlocal count,batch,batch_bytes
                if not batch:return
                arrays=[pa.array(range(count,count+len(batch)),type=pa.int64())]
                for column in columns:
                    arrays.append(pa.array([None if record.get(column) is None else str(record[column]) for record in batch],type=pa.string()))
                table=pa.Table.from_arrays(arrays,names=['__rowid']+[f'c{i}' for i in range(len(columns))])
                connection.register('incoming_batch',table)
                try:connection.execute('INSERT INTO raw_records SELECT * FROM incoming_batch')
                finally:connection.unregister('incoming_batch')
                count+=len(batch);batch=[];batch_bytes=0
            for record in iterator:
                for column in record:
                    if column not in columns:
                        if len(columns)>=512:raise IngestionError('Dataset schema exceeds 512 distinct fields')
                        connection.execute(f'ALTER TABLE raw_records ADD COLUMN c{len(columns)} VARCHAR')
                        columns.append(column)
                batch.append(record)
                batch_bytes+=sum(len(str(v).encode('utf-8')) for v in record.values() if v is not None)
                if len(batch)>=MAX_BATCH_ROWS or batch_bytes>=MAX_RECORD:flush()
            flush()
            if not count:raise IngestionError('Dataset contains no data records')
            profile=full_profile(connection,columns,count,forced_text_fields(parsing))
            profile.update(parsing=parsing,source_bytes=source_bytes,
                suggested_goal={'prediction_target':'label' if 'label' in columns else '',
                    'input_modalities':['text'] if 'text' in columns else ['tabular']})
            for column in parsing.get('json_encoded_fields',[]):profile['data_types'][column]='json'
            preview=connection.execute('SELECT '+','.join(f'c{i}' for i in range(len(columns)))+' FROM records ORDER BY __rowid LIMIT 100').fetchdf()
            preview.columns=columns
            connection.execute('CHECKPOINT')
        return Dataset(name,digest,'tabular',preview,profile,metadata,disk_path=database,storage_owner=owner)
    finally:
        close=getattr(iterator,'close',None)
        if close:close()

def full_profile(connection, columns: list[str], count: int,forced_text=None) -> dict:
    profile = {'row_count':count,'column_count':len(columns),'columns':columns,
        'data_types':{},'missing_count':{},'missing_fraction':{},'non_finite_count':{},
        'mean':{},'std':{},'minimum':{},'maximum':{},'unique_count':{},'categorical_counts':{},
        'datetime_ranges':{},'suspicious_identifiers':[], 'duplicate_rows':None,
        'row_reference':'Zero-based source data row index, excluding header and blank records',
        'storage_backend':'duckdb','scope':'full_dataset',
        'deferred_checks':['distinct counts and exact duplicates are evaluated on demand'],
        'preview_rows':min(100,count)}
    expressions = []
    for i in range(len(columns)):
        c=f'c{i}'
        expressions.extend([f'count({c})', f'count(try_cast({c} as double))',
            f'count(*) filter(where isinf(try_cast({c} as double)))',
            f"count(*) filter(where try_cast({c} as bigint) is not null and regexp_full_match(trim({c}), '[+-]?[0-9]+'))",
            f"count(*) filter(where lower({c}) in ('true','false'))"])
    stats = connection.execute('SELECT '+','.join(expressions)+' FROM raw_records').fetchone()
    selections=['__rowid']
    for i,column in enumerate(columns):
        observed,numeric,infinite,integer,boolean = stats[5*i:5*i+5]
        kind = 'int64' if observed == count and integer == observed else 'float64' if observed and numeric == observed else 'bool' if observed and boolean == observed else 'object'
        if column in (forced_text or set()):kind='object'
        profile['data_types'][column] = kind
        profile['missing_count'][column] = count-observed
        profile['missing_fraction'][column] = (count-observed)/count
        if kind != 'object':
            if kind != 'bool': profile['non_finite_count'][column] = infinite
            cast = 'BIGINT' if kind == 'int64' else 'BOOLEAN' if kind == 'bool' else 'DOUBLE'
            selections.append(f'try_cast(c{i} as {cast}) as c{i}')
        else:
            selections.append(f'c{i}')
        if column.lower() in ('id','uuid') or column.lower().endswith('_id'):
            profile['suspicious_identifiers'].append(column)
    connection.execute('CREATE VIEW records AS SELECT '+','.join(selections)+' FROM raw_records')
    numeric_indices=[i for i,c in enumerate(columns) if profile['data_types'][c] in ('int64','float64')]
    if numeric_indices:
        aggregates=[]
        for i in numeric_indices:
            aggregates.extend(f'{fn}(c{i}) filter(where isfinite(c{i}))' for fn in ('avg','stddev_samp','min','max'))
        values=connection.execute('SELECT '+','.join(aggregates)+' FROM records').fetchone()
        for j,i in enumerate(numeric_indices):
            for name,value in zip(('mean','std','minimum','maximum'),values[4*j:4*j+4]):
                profile[name][columns[i]]=float(value) if value is not None and math.isfinite(value) else None
    return profile

def load_dataset_stream(name: str, source: ByteSource, metadata: dict | None=None, *, force_disk=False) -> Dataset:
    """Copy at most 1 MiB per read, hash all bytes, then ingest bounded batches."""
    metadata=metadata or {}
    suffix=Path(name).suffix.lower()
    from backend.dataset_archives import ARCHIVE_SUFFIXES,load_container
    override=metadata.get('format')
    if override and suffix not in ARCHIVE_SUFFIXES:
        allowed={'csv':'.csv','tsv':'.tsv','psv':'.psv','text':'.txt','json':'.json','jsonl':'.jsonl','xml':'.xml','xlsx':'.xlsx','parquet':'.parquet','feather':'.feather','arrow':'.arrow'}
        if override not in allowed:raise IngestionError('Unsupported explicit format choice')
        suffix=allowed[override]
    from backend.record_formats import RECORD_SUFFIXES
    from backend.image_ingestion import IMAGE_SUFFIXES
    if suffix not in RECORD_SUFFIXES|ARCHIVE_SUFFIXES|IMAGE_SUFFIXES:
        raise IngestionError('Unsupported extension. Use a supported text/table, image, ZIP or TAR archive; executable/pickle formats are not loaded.')
    ROOT.mkdir(parents=True,exist_ok=True)
    owner=TemporaryDirectory(prefix='dataset-',dir=ROOT)
    directory=Path(owner.name)
    path=directory/('source'+suffix)
    limit=int(os.getenv('MAX_UPLOAD_MB','2048'))*1024**2
    if suffix=='.zip': limit=min(limit,int(os.getenv('MAX_IMAGE_UPLOAD_MB','2048'))*1024**2)
    digest,size=hashlib.sha256(),0
    source_handles=ExitStack()
    try:
        with path.open('wb') as destination:
            while block:=source.read(CHUNK_BYTES):
                size+=len(block)
                if size>limit: raise IngestionError('Upload exceeds the configured size limit')
                if shutil.disk_usage(directory).free < len(block)+64*1024**2:
                    raise IngestionError('Insufficient free disk space for the upload')
                digest.update(block)
                destination.write(block)
        threshold=int(os.getenv('LARGE_DATASET_THRESHOLD_MB','16'))*1024**2
        if suffix in ARCHIVE_SUFFIXES:
            ds=load_container(name,path,digest.hexdigest(),metadata)
            if ds.modality=='pending':
                ds.storage_owner=owner;ds.source_path=str(path)
            elif ds.modality=='image' and ds.source_path is None:
                # Keep the safe staged container for opt-in worker-side crops.
                # Nested image datasets already own their staged source.
                ds.storage_owner=owner;ds.source_path=str(path)
            else:owner.cleanup()
            return ds
        if suffix in IMAGE_SUFFIXES:
            from backend.universal_images import load_single_image
            ds=load_single_image(name,path,digest.hexdigest(),metadata)
            ds.storage_owner=owner;ds.source_path=str(path)
            return ds
        if suffix not in ('.csv','.parquet'):
            ds=load_record_table(name,path,digest.hexdigest(),metadata,owner,size)
            path.unlink();return ds
        if not force_disk and size<=threshold:
            effective_name=str(Path(name).with_suffix(suffix))
            ds=load_dataset(effective_name,path.read_bytes(),metadata)
            ds.name=name
            owner.cleanup()
            return ds
        # Reserve headroom for the database and external sorts. Compressed
        # Parquet can expand substantially; disk exhaustion remains a safe error.
        if shutil.disk_usage(directory).free < max(size*2,128*1024**2):
            raise IngestionError('Insufficient disk space for database and temporary checks; allow at least 3 times the upload size')
        database=str(directory/'dataset.duckdb')
        with connect(database,False) as connection:
            if suffix=='.csv':
                generator=csv_batches(path,metadata)
                source_handles.callback(generator.close)
                columns,parsing=next(generator)
                batches=generator
            else:
                parquet=pq.ParquetFile(path)
                source_handles.callback(parquet.close)
                columns=parquet.schema_arrow.names
                if not columns or len(columns)>512 or len(set(columns))!=len(columns) or any(not c.strip() for c in columns):
                    raise IngestionError('Parquet needs 1–512 nonempty unique columns')
                for field in parquet.schema_arrow:
                    if pa.types.is_nested(field.type) or pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
                        raise IngestionError('Large Parquet ingestion supports scalar text, numeric, boolean and timestamp columns')
                parsing={'format':'parquet','warnings':[]}
                batches=parquet.iter_batches(batch_size=MAX_BATCH_ROWS)
                source_handles.callback(batches.close)
            connection.execute('CREATE TABLE raw_records (__rowid BIGINT,'+','.join(f'c{i} VARCHAR' for i in range(len(columns)))+')')
            count=0
            for batch in batches:
                if not isinstance(batch,pa.RecordBatch):
                    # Construct Arrow arrays directly: no whole-file DataFrame.
                    rows=cast(list[list[str | None]],batch)
                    arrays=[pa.array([row[i] for row in rows],type=pa.string()) for i in range(len(columns))]
                    length=len(rows)
                else:
                    record_batch=cast(pa.RecordBatch,batch)
                    arrays=[]
                    for array in record_batch.columns:
                        strings=pc.cast(array,pa.string())
                        # Treat floating NaNs as missing, consistent with pandas.
                        if pa.types.is_floating(array.type):
                            strings=pc.call_function('if_else',[pc.call_function('is_nan',[array]),pa.scalar(None,type=pa.string()),strings])
                        arrays.append(strings)
                    length=record_batch.num_rows
                table=pa.Table.from_arrays([pa.array(range(count,count+length),type=pa.int64())]+arrays,
                    names=['__rowid']+[f'c{i}' for i in range(len(columns))])
                connection.register('incoming_batch',table)
                try: connection.execute('INSERT INTO raw_records SELECT * FROM incoming_batch')
                finally: connection.unregister('incoming_batch')
                count+=length
            if count==0: raise IngestionError('Dataset must contain at least one data row')
            profile=full_profile(connection,columns,count,forced_text_fields(parsing))
            profile['parsing']=parsing
            profile['source_bytes']=size
            profile['processing']={'database_memory_mb':int(os.getenv('DATASET_MEMORY_MB','256')),'database_threads':2,
                'batch_rows':MAX_BATCH_ROWS,'csv_batch_target_bytes':MAX_RECORD_BYTES,'preview_rows':100}
            preview=connection.execute('SELECT '+','.join(f'c{i}' for i in range(len(columns)))+' FROM records ORDER BY __rowid LIMIT 100').fetchdf()
            preview.columns=columns
            connection.execute('CHECKPOINT')
        # Windows will not remove a source still held by a Parquet reader or
        # suspended CSV generator. Close all readers before deleting staging.
        source_handles.close()
        path.unlink()
        return Dataset(name,digest.hexdigest(),'tabular',preview,profile,metadata,disk_path=database,storage_owner=owner)
    except SelectionRequired as exc:
        source_handles.close()
        return Dataset(name,digest.hexdigest(),'pending',pd.DataFrame(),
            {'row_count':0,'column_count':0,'needs_selection':True,'selection_message':str(exc),'choices':exc.choices,
             'columns':[],'source_bytes':size},metadata,storage_owner=owner,source_path=str(path))
    except Exception as exc:
        source_handles.close()
        owner.cleanup()
        if isinstance(exc,IngestionError): raise
        if isinstance(exc,zipfile.BadZipFile):
            raise IngestionError('Invalid or corrupt ZIP archive; verify the download is complete') from exc
        if isinstance(exc,(UnicodeError,csv.Error)):
            raise IngestionError('CSV decoding or structure error: '+str(exc)) from exc
        if isinstance(exc,(ijson.JSONError,DefusedXmlException,tarfile.TarError,InvalidFileException,RecursionError,ExpatError)):
            raise IngestionError('The declared data format is malformed or unsafe: '+str(exc)[:200]) from exc
        if isinstance(exc,(OSError,duckdb.Error,pa.ArrowException,ValueError,KeyError,TypeError)):
            raise IngestionError('Large dataset could not be processed; check format, disk space and DATASET_MEMORY_MB. '+str(exc)[:200]) from exc
        raise
