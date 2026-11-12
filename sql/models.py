"""SQLAlchemy table schema for persisted research records."""
from sqlalchemy import Column, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Record(Base):
    __tablename__ = 'research_records'
    id = Column(String, primary_key=True)
    kind = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False)
    document = Column(Text, nullable=False)


