from dataclasses import dataclass, field
from typing import Optional, Dict

"""
    This file only is used to analyze single labels.
"""

@dataclass
class LabelAnalysisResult:
    label_column: str

    #For each label, metrics of diff classes or params
    class_counts: Dict = field(default_factory=dict)
    class_fractions: Dict = field(default_factory=dict)
    missing_labels: int = 0
    missing_label_fraction: float = 0.0

    number_of_classes: Optional[int] = None

    class_imbalance_ratio: Optional[float] = None #(largest class/smallest class)

    cohen_kappa: Optional[float] = None
    raw_agreement: Optional[float] = None


class LabelAnalyzer:
    def __init__(self, dataframe, label_column: str, second_label_column: str | None = None):
        self.data = dataframe
        self.label_column = label_column
        self.second_label_column = second_label_column

    def analyze_distribution(self):
        labels = self.data[self.label_column]
        counts = labels.value_counts(dropna=True)
        class_counts = counts.to_dict()
        class_fractions = (counts / counts.sum()).to_dict()
        return class_counts, class_fractions

    def analyze_missing_labels(self):
        missing = int(self.data[self.label_column].isna().sum())
        total = len(self.data)
        fraction = missing / total if total else 0.0
        return missing, fraction

    def calculate_imbalance(self, class_counts):
        if not class_counts:
            return None

        largest = max(class_counts.values())
        smallest = min(class_counts.values())

        if smallest == 0:
            return None
        return largest / smallest

    def analyze(self) -> LabelAnalysisResult:
        if self.label_column not in self.data.columns:
            raise ValueError(f"{self.label_column} does not exist in dataset")

        class_counts, class_fractions = (self.analyze_distribution())

        missing, missing_fraction = (self.analyze_missing_labels())

        imbalance = self.calculate_imbalance(class_counts)

        kappa = agreement = None
        if self.second_label_column:
            from sklearn.metrics import cohen_kappa_score
            import math
            pair = self.data[[self.label_column, self.second_label_column]].dropna()
            if len(pair):
                agreement = float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean())
                if len(set(pair.iloc[:, 0]) | set(pair.iloc[:, 1])) > 1:
                    value = float(cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1]))
                    kappa = value if math.isfinite(value) else None
        return LabelAnalysisResult(
            label_column=self.label_column,
            class_counts=class_counts,
            class_fractions=class_fractions,
            missing_labels=missing,
            missing_label_fraction=missing_fraction,
            number_of_classes=len(class_counts),
            class_imbalance_ratio=imbalance,
            cohen_kappa=kappa,
            raw_agreement=agreement
        )
