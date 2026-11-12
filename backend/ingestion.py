"""Bounded offline ingestion; large ZIPs are streamed, never extracted.

YOLO ZIP contract: classes.txt, images/<path>.<ext>, labels/<path>.txt,
optional metadata.csv keyed by image (the full images/... archive path).
An empty label file means an annotated negative image; no file means unknown.
"""
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
import codecs
import csv
import hashlib
import io
import os
import zipfile
import pandas as pd
from metrics.dataProfiler import Dataprofiler

class IngestionError(ValueError):
    pass

class SelectionRequired(IngestionError):
    """A supported container has several plausible datasets/record layouts."""
    def __init__(self,message,choices):
        super().__init__(message)
        self.choices=choices


def decode_csv(data: bytes, declared_encoding: Any = None) -> tuple[str, dict[str, Any]]:
    """Decode text without dropping bytes or silently inserting replacements.

    A BOM identifies Unicode encodings reliably. Without a BOM, prefer UTF-8
    and try Windows-1252 only if strict UTF-8 decoding fails. The fallback is
    recorded as an inference, since bytes alone cannot prove a legacy encoding.
    A user-declared encoding is authoritative and is never silently overridden.
    """
    notes: list[str] = []
    if declared_encoding is not None:
        if not isinstance(declared_encoding, str):
            raise IngestionError('CSV encoding must be a supported encoding name')
        try:
            encoding = codecs.lookup(declared_encoding).name
        except LookupError as exc:
            raise IngestionError(f'Unsupported CSV encoding: {declared_encoding}') from exc
        supported = {'utf-8', 'utf-8-sig', 'cp1252', 'ascii', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be'}
        if encoding not in supported:
            raise IngestionError(f'Unsupported CSV encoding: {declared_encoding}; use UTF-8, UTF-16, UTF-32 or Windows-1252')
        source = 'user_declared'
    elif data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        # UTF-32 LE shares the first two bytes of the UTF-16 LE BOM.
        encoding, source = 'utf-32', 'byte_order_mark'
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding, source = 'utf-16', 'byte_order_mark'
    elif data.startswith(codecs.BOM_UTF8):
        encoding, source = 'utf-8-sig', 'byte_order_mark'
    else:
        encoding, source = 'utf-8', 'strict_utf8'
        try:
            data.decode(encoding, errors='strict')
        except UnicodeDecodeError:
            encoding, source = 'cp1252', 'legacy_fallback'
            notes.append('Strict UTF-8 decoding failed; Windows-1252 was used as a fallback. Confirm the text looks correct or declare csv_encoding in upload metadata.')
    try:
        text = data.decode(encoding, errors='strict')
    except UnicodeError as exc:
        raise IngestionError(f'CSV cannot be decoded using {encoding}. Export it as UTF-8 or supply the correct csv_encoding in upload metadata.') from exc
    if any((ord(c) < 32 and c not in '\t\r\n') or 127 <= ord(c) <= 159 for c in text):
        raise IngestionError('CSV contains binary NUL bytes or unsupported control characters')
    return text, {'encoding': encoding, 'encoding_source': source, 'delimiter': ',', 'warnings': notes}

@dataclass
class Dataset:
    name: str
    sha256: str
    modality: str
    frame: pd.DataFrame
    profile: dict
    metadata: dict = field(default_factory=dict)
    objects: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    image_hashes: dict[str, str] = field(default_factory=dict)
    near_pairs: list[tuple[str, str]] = field(default_factory=list)
    near_distance: int | None = None
    # Large tabular data lives in a private on-disk database. frame contains
    # only a bounded preview; assessment code must use disk_path when present.
    disk_path: str | None = None
    storage_owner: Any = field(default=None, repr=False)
    source_path: str | None = None
    # Runtime-only read capability; never serialized or supplied by uploads.
    image_reader: Any = field(default=None, repr=False)

    def close(self):
        if self.storage_owner is not None:
            self.storage_owner.cleanup()
            self.storage_owner = None

def load_dataset(name: str, data: bytes, metadata: dict | None = None) -> Dataset:
    if len(data) > int(os.getenv('MAX_UPLOAD_MB', '2048')) * 1024**2:
        raise IngestionError('Upload exceeds the configured size limit')
    metadata = metadata or {}
    suffix = PurePosixPath(name.lower()).suffix
    if suffix not in ('.csv','.parquet','.zip'):
        from backend.large_ingestion import load_dataset_stream
        return load_dataset_stream(name,io.BytesIO(data),metadata)
    if suffix == '.zip' and len(data) > int(os.getenv('MAX_IMAGE_UPLOAD_MB', '2048')) * 1024**2:
        raise IngestionError('Image ZIP exceeds MAX_IMAGE_UPLOAD_MB')
    if suffix in ('.csv', '.parquet') and len(data) > int(os.getenv('LARGE_DATASET_THRESHOLD_MB', '16')) * 1024**2:
        from backend.large_ingestion import load_dataset_stream
        return load_dataset_stream(name, io.BytesIO(data), metadata, force_disk=True)
    digest = hashlib.sha256(data).hexdigest()
    parsing = None
    try:
        if suffix == '.zip':
            return load_images(name, data, digest, metadata)
        if suffix == '.csv':
            text, parsing = decode_csv(data, metadata.get('csv_encoding'))
            from backend.record_formats import delimiter_for
            delimiter,origin=delimiter_for(text[:65536],metadata,'.csv')
            parsing.update(delimiter=delimiter,delimiter_source=origin,has_header=metadata.get('has_header',True))
            if not isinstance(parsing['has_header'],bool):raise IngestionError('has_header must be true or false')
            first = next(csv.reader(io.StringIO(text),delimiter=delimiter))
            header = first if parsing['has_header'] else metadata.get('column_names') or [f'column_{i+1}' for i in range(len(first))]
            if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
                raise IngestionError('CSV needs nonempty, unique column names')
            # Check row lengths before pandas can silently infer an index.
            for i, row in enumerate(csv.reader(io.StringIO(text),delimiter=delimiter,strict=True)):
                if row and len(row) != len(header):
                    raise IngestionError(f'CSV row {i+1} has the wrong number of fields')
            df = pd.read_csv(io.StringIO(text), on_bad_lines='error',sep=delimiter,header=0 if parsing['has_header'] else None,names=None if parsing['has_header'] else header)
        elif suffix == '.parquet':
            if not data.startswith(b'PAR1') or not data.endswith(b'PAR1'):
                raise IngestionError('Invalid Parquet signature')
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(io.BytesIO(data))
            # Reject huge declared decompressed buffers before materializing them.
            decoded_size = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.metadata.num_row_groups))
            if decoded_size > 128 * 1024**2 or parquet.metadata.num_rows > 500_000:
                from backend.large_ingestion import load_dataset_stream
                return load_dataset_stream(name, io.BytesIO(data), metadata, force_disk=True)
            df = parquet.read().to_pandas()
            if int(df.memory_usage(deep=True).sum()) > 256 * 1024**2:
                raise IngestionError('Decoded table exceeds the memory limit')
        else:
            raise IngestionError('Supported uploads: CSV, Parquet, or YOLO/VisDrone DET image ZIP')
        if len(df) > 500_000:
            from backend.large_ingestion import load_dataset_stream
            return load_dataset_stream(name, io.BytesIO(data), metadata, force_disk=True)
        if any(not isinstance(c, str) for c in df.columns):
            raise IngestionError('Column names must be strings')
        profile = Dataprofiler(df).profile().to_dict()
        if parsing is not None:
            profile['parsing'] = parsing
        profile['row_reference'] = 'Zero-based data row index (CSV header excluded)'
        return Dataset(name, digest, 'tabular', df, profile, metadata)
    except IngestionError:
        raise
    except (ValueError, UnicodeError, StopIteration, OSError, zipfile.BadZipFile, KeyError, TypeError) as exc:
        raise IngestionError('File could not be parsed as the declared supported format') from exc

def load_images(name, data, digest, metadata):
    """Compatibility wrapper; large uploads pass their staged path directly."""
    from backend.large_ingestion import load_dataset_stream
    return load_dataset_stream(name,io.BytesIO(data),metadata)
