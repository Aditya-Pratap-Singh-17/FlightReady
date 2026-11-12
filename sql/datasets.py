"""Bounded DuckDB connections for full-dataset analysis."""
from pathlib import Path
import os
import duckdb
from backend.ingestion import IngestionError


def connect(path: str, read_only: bool = True):
    memory = int(os.getenv('DATASET_MEMORY_MB', '256'))
    if not 64 <= memory <= 8192:
        raise IngestionError('DATASET_MEMORY_MB must be between 64 and 8192')
    connection = duckdb.connect(path, read_only=read_only, config={
        'memory_limit': f'{memory}MB', 'threads': 2,
        'temp_directory': str(Path(path).parent / 'spill'),
    })
    # Configure the owned spill path before disabling external SQL access.
    connection.execute('SET enable_external_access=false')
    return connection

