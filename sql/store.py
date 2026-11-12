"""SQLite metadata and managed filesystem artifacts, never pickle or DB blobs."""
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import json
import os
import uuid
from sqlalchemy import create_engine, select, event
from sqlalchemy.orm import Session
from sql.models import Base, Record


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv('WORKBENCH_ROOT', '.runtime/workbench')).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.engine = create_engine('sqlite:///' + (self.root / 'metadata.sqlite3').as_posix(), connect_args={'timeout': 30})
        @event.listens_for(self.engine, 'connect')
        def configure(connection, _):
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA foreign_keys=ON')
        Base.metadata.create_all(self.engine)

    def create(self, kind, payload, id=None):
        id = id or uuid.uuid4().hex
        with Session(self.engine) as session:
            session.add(Record(id=id, kind=kind, created_at=datetime.now(timezone.utc).isoformat(), document=json.dumps(payload, allow_nan=False)))
            session.commit()
        return self.get(id, kind)

    def get(self, id, kind=None):
        with Session(self.engine) as session:
            row = session.get(Record, id)
            if row is None or kind is not None and row.kind != kind:
                raise KeyError(f'{kind or "Record"} {id} not found')
            return {'id': row.id, 'kind': row.kind, 'created_at': row.created_at, 'payload': json.loads(row.document)}

    def list(self, kind):
        with Session(self.engine) as session:
            ids = session.scalars(select(Record.id).where(Record.kind == kind).order_by(Record.created_at.desc())).all()
        return [self.get(id) for id in ids]

    def update(self, id, **changes):
        with self.lock, Session(self.engine) as session:
            # Serialize read-modify-write across the HTTP and worker processes.
            session.connection().exec_driver_sql('BEGIN IMMEDIATE')
            row = session.get(Record, id)
            if row is None:
                raise KeyError(id)
            payload = json.loads(row.document)
            payload.update(changes)
            row.document = json.dumps(payload, allow_nan=False)
            session.commit()
        return self.get(id)

    def delete(self, id):
        with Session(self.engine) as session:
            row = session.get(Record, id)
            if row is not None:
                session.delete(row)
                session.commit()

    def directory(self, kind, id):
        if not id.isalnum() or not kind.isalnum():
            raise ValueError('Invalid managed artifact identifier')
        path = self.root / kind / id
        path.mkdir(parents=True, exist_ok=True)
        return path
