"""Data validation module for weather data quality checks."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    check_name: str
    message: str
    details: Optional[Dict] = None


@dataclass
class DataValidator:
    """Validator for weather dataset quality checks."""

    # Expected columns in the dataset
    required_columns: List[str] = field(default_factory=lambda: [
        "min_temp", "max_temp", "mean_temp", "sunshine",
        "global_radiation", "precipitation", "pressure",
        "cloud_cover", "snow_depth"
    ])

    # Value constraints
    constraints: Dict[str, Tuple[Optional[float], Optional[float]]] = field(
        default_factory=lambda: {
            "min_temp": (-50.0, 50.0),
            "max_temp": (-50.0, 60.0),
            "mean_temp": (-50.0, 55.0),
            "sunshine": (0.0, 24.0),
            "global_radiation": (0.0, 500.0),
            "precipitation": (0.0, 500.0),
            "pressure": (80000.0, 110000.0),
            "cloud_cover": (0.0, 10.0),
            "snow_depth": (0.0, 200.0),
        }
    )

    # Maximum allowed missing value percentage
    max_missing_pct: float = 10.0

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Check that all required columns are present."""
        missing_cols = [col for col in self.required_columns if col not in df.columns]

        if missing_cols:
            return ValidationResult(
                is_valid=False,
                check_name="schema_validation",
                message=f"Missing required columns: {missing_cols}",
                details={"missing_columns": missing_cols}
            )

        return ValidationResult(
            is_valid=True,
            check_name="schema_validation",
            message="All required columns present",
            details={"columns": list(df.columns)}
        )

    def validate_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check for excessive missing values."""
        missing_pct = (df[self.required_columns].isnull().sum() / len(df) * 100).to_dict()
        excessive_missing = {col: pct for col, pct in missing_pct.items()
                           if pct > self.max_missing_pct}

        if excessive_missing:
            return ValidationResult(
                is_valid=False,
                check_name="missing_values",
                message=f"Columns with excessive missing values (>{self.max_missing_pct}%): {list(excessive_missing.keys())}",
                details={"missing_percentages": missing_pct, "threshold": self.max_missing_pct}
            )

        return ValidationResult(
            is_valid=True,
            check_name="missing_values",
            message="Missing values within acceptable limits",
            details={"missing_percentages": missing_pct}
        )

    def validate_value_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Check that values are within expected ranges."""
        violations = {}

        for col, (min_val, max_val) in self.constraints.items():
            if col not in df.columns:
                continue

            col_data = df[col].dropna()

            if min_val is not None and col_data.min() < min_val:
                violations[col] = {
                    "issue": "below_minimum",
                    "min_expected": min_val,
                    "actual_min": col_data.min(),
                    "count": (col_data < min_val).sum()
                }

            if max_val is not None and col_data.max() > max_val:
                violations[col] = {
                    "issue": "above_maximum",
                    "max_expected": max_val,
                    "actual_max": col_data.max(),
                    "count": (col_data > max_val).sum()
                }

        if violations:
            return ValidationResult(
                is_valid=False,
                check_name="value_ranges",
                message=f"Value range violations in columns: {list(violations.keys())}",
                details={"violations": violations}
            )

        return ValidationResult(
            is_valid=True,
            check_name="value_ranges",
            message="All values within expected ranges",
            details={"constraints": self.constraints}
        )

    def validate_temperature_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check temperature logical consistency (min <= mean <= max)."""
        if not all(col in df.columns for col in ["min_temp", "mean_temp", "max_temp"]):
            return ValidationResult(
                is_valid=True,
                check_name="temperature_consistency",
                message="Temperature columns not all present, skipping check"
            )

        inconsistent = df[
            (df["min_temp"] > df["mean_temp"]) |
            (df["mean_temp"] > df["max_temp"]) |
            (df["min_temp"] > df["max_temp"])
        ]

        if len(inconsistent) > 0:
            return ValidationResult(
                is_valid=False,
                check_name="temperature_consistency",
                message=f"Found {len(inconsistent)} rows with inconsistent temperature values",
                details={
                    "inconsistent_count": len(inconsistent),
                    "inconsistent_percentage": len(inconsistent) / len(df) * 100
                }
            )

        return ValidationResult(
            is_valid=True,
            check_name="temperature_consistency",
            message="Temperature values are logically consistent"
        )

    def validate_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> ValidationResult:
        """Check for duplicate rows."""
        duplicates = df.duplicated(subset=subset).sum()

        if duplicates > 0:
            return ValidationResult(
                is_valid=False,
                check_name="duplicates",
                message=f"Found {duplicates} duplicate rows",
                details={"duplicate_count": duplicates, "subset": subset}
            )

        return ValidationResult(
            is_valid=True,
            check_name="duplicates",
            message="No duplicate rows found"
        )

    def validate_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.1
    ) -> ValidationResult:
        """Check for data drift between reference and current datasets."""
        drift_detected = {}

        for col in self.required_columns:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_mean = reference_df[col].mean()
            curr_mean = current_df[col].mean()

            if ref_mean != 0:
                relative_diff = abs(curr_mean - ref_mean) / abs(ref_mean)
            else:
                relative_diff = abs(curr_mean - ref_mean)

            if relative_diff > threshold:
                drift_detected[col] = {
                    "reference_mean": ref_mean,
                    "current_mean": curr_mean,
                    "relative_difference": relative_diff
                }

        if drift_detected:
            return ValidationResult(
                is_valid=False,
                check_name="data_drift",
                message=f"Data drift detected in columns: {list(drift_detected.keys())}",
                details={"drift": drift_detected, "threshold": threshold}
            )

        return ValidationResult(
            is_valid=True,
            check_name="data_drift",
            message="No significant data drift detected",
            details={"threshold": threshold}
        )

    def validate_all(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Run all validation checks."""
        results = [
            self.validate_schema(df),
            self.validate_missing_values(df),
            self.validate_value_ranges(df),
            self.validate_temperature_consistency(df),
            self.validate_duplicates(df),
        ]
        return results

    def get_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive validation report."""
        results = self.validate_all(df)

        return {
            "overall_valid": all(r.is_valid for r in results),
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r.is_valid),
            "failed_checks": sum(1 for r in results if not r.is_valid),
            "results": [
                {
                    "check": r.check_name,
                    "is_valid": r.is_valid,
                    "message": r.message,
                    "details": r.details
                }
                for r in results
            ]
        }


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Convenience function to validate a dataset."""
    validator = DataValidator()
    report = validator.get_validation_report(df)
    return report["overall_valid"], report


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    ROOT = Path(__file__).parent.parent.parent
    data_path = ROOT / "data" / "processed" / "train.parquet"

    if data_path.exists():
        df = pd.read_parquet(data_path)
        is_valid, report = validate_dataset(df)

        print("=" * 50)
        print("DATA VALIDATION REPORT")
        print("=" * 50)
        print(f"Overall Valid: {is_valid}")
        print(f"Checks Passed: {report['passed_checks']}/{report['total_checks']}")
        print()

        for result in report["results"]:
            status = "PASS" if result["is_valid"] else "FAIL"
            print(f"[{status}] {result['check']}: {result['message']}")
    else:
        print(f"Data file not found: {data_path}")