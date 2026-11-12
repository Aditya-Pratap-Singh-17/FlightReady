from dataclasses import dataclass
from typing import Optional
from metrics.dataProfiler import DataProfile
from backend.context import Goal 

"""
    This file is made for ISO/IEC metrics checking purpose.
"""

@dataclass
class ISOIECresult:

    accuracy: Optional[float] = None
    completeness: Optional[float] = None
    consistency: Optional[float] = None
    timeliness: Optional[float] = None
    validity: Optional[float] = None
    uniqueness: Optional[float] = None
    conformity: Optional[float] = None
    traceability: Optional[float] = None

class ISOIECchecker:
    def __init__(self, profile: DataProfile, goal: Goal):
        self.profile = profile
        self.goal = goal

    # 1. Accuracy
    def check_accuracy(self) -> Optional[float]:

        """ 
            Accuracy requires trusted/reference values.
            DataProfile currently does not contain enough
            information to calculate this reliably.
        """

        if not self.goal.reference_columns:
            return None
        return None

    # 2. Completeness
    def check_completeness(self) -> float:
        total_values = (self.profile.row_count * self.profile.column_count)

        if total_values == 0:
            return 0.0

        total_missing = sum(self.profile.missing_count.values())

        completeness = 1 - (total_missing / total_values)
        return completeness

    # 3. Consistency
    def check_consistency(self) -> Optional[float]:

        """
            Proper consistency checking requires things such as:
            units, formatting rules, relationships between columns,etc.
            Current DataProfile does not store enough information.
        """
        if not self.goal.expected_units:
            return None

        return None

    # 4. Timliness
    def check_timeliness(self) -> Optional[float]:

        # Need a date column and the actual dates contained
        # inside the dataset.

        if self.goal.date_column is None:
            return None

        if self.goal.max_data_age_days is None:
            return None

        if self.goal.date_column not in self.profile.columns:
            return 0.0

        # Actual date values are not currently stored
        # inside DataProfile.
        return None

    # 5. Validity
    def check_validity(self) -> Optional[float]:

        checks = 0
        passed = 0

        # Check expected data types
        for column, expected_type in self.goal.expected_types.items():

            checks += 1

            if column not in self.profile.data_types:
                continue

            actual_type = self.profile.data_types[column]

            if expected_type.lower() in actual_type.lower():
                passed += 1

        # We cannot yet properly evaluate valid_ranges or
        # allowed_values because DataProfile only contains
        # summary statistics.

        if checks == 0:
            return None
        return passed / checks

    # 6. Uniquesness
    def check_uniqueness(self) -> float:
        total_rows = self.profile.row_count

        if total_rows == 0:
            return 0.0

        duplicate_rows = self.profile.duplicate_rows

        uniqueness = 1 - (duplicate_rows / total_rows)
        return uniqueness
    
    # 7. Conformity
    def check_conformity(self) -> Optional[float]:

        required_columns = set(self.goal.required_inputs)

        if not required_columns:
            return None

        existing_columns = set(self.profile.columns)

        matching_columns = (required_columns & existing_columns)

        conformity = (len(matching_columns) /len(required_columns))

        return conformity

    # 8. Traceability
    def check_traceability(self) -> Optional[float]:

        requirements = 0

        if self.goal.require_source:
            requirements += 1

        if self.goal.require_collection_date:
            requirements += 1

        if self.goal.require_dataset_version:
            requirements += 1

        if requirements == 0:
            return None

        # DataProfile currently doesn't contain provenance
        # metadata, so we cannot claim these requirements
        # are satisfied.

        # Missing provenance is unknown evidence, not a measured failure.
        return None


    def check(self) -> ISOIECresult:

        return ISOIECresult(
            accuracy=self.check_accuracy(),
            completeness=self.check_completeness(),
            consistency=self.check_consistency(),
            timeliness=self.check_timeliness(),
            validity=self.check_validity(),
            uniqueness=self.check_uniqueness(),
            conformity=self.check_conformity(),
            traceability=self.check_traceability()
        )

