"""Feature heuristics identify investigation opportunities, not proven leakage."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class FeatureAnalysisResult:
    missing_fraction: dict = field(default_factory=dict)
    constant_features: list = field(default_factory=list)
    low_variance_features: list = field(default_factory=list)
    redundant_pairs: list = field(default_factory=list)
    suspicious_identifiers: list = field(default_factory=list)
    target_mutual_information: dict = field(default_factory=dict)
    leakage_warnings: dict = field(default_factory=dict)
    skipped_features: dict = field(default_factory=dict)

class FeaturAnalyzer:
    def __init__(self, dataframe, target_column, dominance_threshold=0.99, correlation_threshold=0.95):
        if dataframe.empty or not dataframe.columns.is_unique or target_column not in dataframe:
            raise ValueError('Need nonempty data, unique columns and an existing target')
        if not 0 < dominance_threshold <= 1 or not 0 < correlation_threshold <= 1:
            raise ValueError('Invalid threshold')
        self.data, self.target_column = dataframe, target_column
        self.feature_columns = [c for c in dataframe if c != target_column]
        self.dominance_threshold, self.correlation_threshold = dominance_threshold, correlation_threshold

    def analyze_missingness(self):
        return {c: float(self.data[c].isna().mean()) for c in self.feature_columns}

    def analyze(self):
        r = FeatureAnalysisResult(missing_fraction=self.analyze_missingness())
        for c in self.feature_columns:
            s = self.data[c]
            if s.nunique() <= 1:
                r.constant_features.append(c)
            fractions = s.value_counts(normalize=True)
            if len(fractions) and fractions.iloc[0] >= self.dominance_threshold:
                r.low_variance_features.append(c)
            if c.lower() == 'id' or c.lower().endswith('_id'):
                r.suspicious_identifiers.append(c)
            if s.equals(self.data[self.target_column]):
                r.leakage_warnings[c] = 'Identical to target; review availability at prediction time'
        corr = self.data[self.feature_columns].select_dtypes(include='number').replace([np.inf, -np.inf], np.nan).corr()
        for i, a in enumerate(corr):
            for b in list(corr)[i+1:]:
                if abs(corr.loc[a, b]) >= self.correlation_threshold:
                    r.redundant_pairs.append((a, b, float(corr.loc[a, b])))
        r.skipped_features['mutual_information'] = 'Requires a declared task encoding and estimator configuration'
        return r

FeatureAnalyzer = FeaturAnalyzer
