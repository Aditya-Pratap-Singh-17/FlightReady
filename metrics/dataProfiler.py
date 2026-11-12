"""Basic profiles keep missing values distinct from positive/negative infinity."""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

@dataclass
class DataProfile:
    row_count: int
    column_count: int
    columns: list = field(default_factory=list)
    data_types: dict = field(default_factory=dict)
    missing_count: dict = field(default_factory=dict)
    missing_fraction: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    mean: dict = field(default_factory=dict)
    std: dict = field(default_factory=dict)
    minimum: dict = field(default_factory=dict)
    maximum: dict = field(default_factory=dict)
    unique_count: dict = field(default_factory=dict)
    non_finite_count: dict = field(default_factory=dict)
    categorical_counts: dict = field(default_factory=dict)
    datetime_ranges: dict = field(default_factory=dict)
    suspicious_identifiers: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

class Dataprofiler:
    def __init__(self, dataframe):
        self.data = dataframe

    def profile(self) -> DataProfile:
        df = self.data
        if df.empty or not df.columns.is_unique:
            raise ValueError('Dataset must have rows and unique column names')
        p = DataProfile(len(df), len(df.columns), columns=df.columns.tolist())
        p.data_types = {c: str(df[c].dtype) for c in df}
        p.missing_count = {c: int(df[c].isna().sum()) for c in df}
        p.missing_fraction = {c: n / len(df) for c, n in p.missing_count.items()}
        p.unique_count = {c: int(df[c].nunique()) for c in df}
        p.duplicate_rows = int(df.duplicated().sum())
        for c in df:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                p.non_finite_count[c] = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum())
                finite = s.replace([np.inf, -np.inf], np.nan).dropna()
                for name, value in [('mean', finite.mean()), ('std', finite.std()), ('minimum', finite.min()), ('maximum', finite.max())]:
                    getattr(p, name)[c] = float(value) if pd.notna(value) and np.isfinite(value) else None
            else:
                p.categorical_counts[c] = {str(k): int(v) for k, v in s.value_counts().head(50).items()}
            # Date parsing is diagnostic: it does not silently change source values.
            if pd.api.types.is_datetime64_any_dtype(s) or 'timestamp' in c.lower() or c.lower().endswith('date'):
                dates = pd.to_datetime(s, errors='coerce', utc=True)
                p.datetime_ranges[c] = {'min': str(dates.min()), 'max': str(dates.max()), 'invalid': int((dates.isna() & s.notna()).sum())}
            if c.lower() in ('id', 'uuid') or c.lower().endswith('_id'):
                p.suspicious_identifiers.append(c)
        return p

DataProfiler = Dataprofiler
