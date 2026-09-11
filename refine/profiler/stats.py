"""Schema-aware statistical profiler. Uses inferred schema when available, otherwise applies generic checks."""

from typing import Any

import numpy as np
import polars as pl

# Default thresholds (used when no schema overrides are provided)
NULL_RATIO_THRESHOLD = 0.20
ZSCORE_THRESHOLD = 3.0
CLASS_IMBALANCE_MIN = 0.20


def profile_dataset(
    df: pl.DataFrame,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically analyzes missingness, statistical distributions, and anomalies.

    When a schema dict is provided (from schema_inference), domain-specific bounds
    and target column information drive the checks. Without a schema, generic
    statistical checks are applied to all columns.
    """
    total_rows = df.height
    profile: dict[str, Any] = {
        "total_rows": total_rows,
        "columns": {},
        "critical_issues": [],
    }

    # Extract schema info if available
    schema_columns = (schema or {}).get("columns", {})
    target_column = (schema or {}).get("target_column")

    for col in df.columns:
        col_type = str(df.schema[col])
        null_count = df[col].null_count()
        null_ratio = null_count / total_rows
        col_schema = schema_columns.get(col, {})

        col_summary: dict[str, Any] = {
            "type": col_type,
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
            "outliers_count": 0,
            "anomalies": [],
        }

        # Skip id / ignore columns from anomaly checks
        col_role = col_schema.get("role", "feature")
        if col_role in ("id", "ignore"):
            profile["columns"][col] = col_summary
            continue

        # ── 1. High Null Ratio Trigger (>= 20%) ──────────────────
        if null_ratio >= NULL_RATIO_THRESHOLD:
            col_summary["anomalies"].append(f"Nulls ({null_ratio * 100:.1f}%)")
            profile["critical_issues"].append(
                {
                    "column": col,
                    "type": "HIGH_NULL_RATIO",
                    "ratio": null_ratio,
                    "message": (
                        f"Column '{col}' has a missingness ratio of "
                        f"{null_ratio * 100:.1f}% (Threshold: {NULL_RATIO_THRESHOLD * 100:.0f}%)."
                    ),
                }
            )

        # ── 2. Numerical Checks ──────────────────────────────────
        if col_type in ("Int32", "Int64", "Float32", "Float64"):
            non_nulls = df[col].drop_nulls().to_numpy()
            if len(non_nulls) > 0:
                # 2a. Domain bounds check (from schema or skip)
                bounds = col_schema.get("valid_bounds")
                if bounds and len(bounds) == 2:
                    low, high = bounds
                    invalid_mask = (non_nulls < low) | (non_nulls > high)
                    invalid_count = int(np.sum(invalid_mask))
                    if invalid_count > 0:
                        col_summary["anomalies"].append(f"{invalid_count} Out-of-bounds")
                        profile["critical_issues"].append(
                            {
                                "column": col,
                                "type": "INVALID_BOUNDS",
                                "count": invalid_count,
                                "bounds": bounds,
                                "message": (
                                    f"Column '{col}' contains {invalid_count} values "
                                    f"outside valid domain [{low}, {high}]."
                                ),
                            }
                        )

                # 2b. Binary vs Continuous Outlier & Imbalance Checks
                unique_vals = set(non_nulls.tolist())
                is_binary = col_schema.get("semantic_type") == "binary" or unique_vals.issubset({0, 1})

                if is_binary and len(unique_vals) == 2:
                    # Generic class imbalance check for binary features
                    val_counts = df[col].drop_nulls().value_counts().sort("count")
                    min_count = val_counts["count"][0]
                    min_val = val_counts[col][0]
                    total_cnt = val_counts["count"].sum()
                    min_ratio = min_count / total_cnt
                    if min_ratio < CLASS_IMBALANCE_MIN:
                        col_summary["imbalance"] = f"Class {min_val} ({min_ratio * 100:.1f}%)"
                        col_summary["anomalies"].append(f"Imbalance ({min_ratio * 100:.1f}%)")
                        if not any(
                            iss["column"] == col and iss["type"] == "CLASS_IMBALANCE"
                            for iss in profile["critical_issues"]
                        ):
                            profile["critical_issues"].append(
                                {
                                    "column": col,
                                    "type": "CLASS_IMBALANCE",
                                    "minority_ratio": round(min_ratio, 4),
                                    "minority_value": min_val,
                                    "message": (
                                        f"Column '{col}' has severe class imbalance: "
                                        f"minority class ({min_val}) is only {min_ratio * 100:.1f}% "
                                        f"(Threshold: {CLASS_IMBALANCE_MIN * 100:.0f}%)."
                                    ),
                                }
                            )

                elif not is_binary:
                    mean = np.mean(non_nulls)
                    std = np.std(non_nulls)
                    if std > 0:
                        z_scores = np.abs((non_nulls - mean) / std)
                        outliers = int(np.sum(z_scores > ZSCORE_THRESHOLD))
                        col_summary["outliers_count"] = outliers
                        if outliers > 0:
                            col_summary["anomalies"].append(f"{outliers} Outliers")
                            profile["critical_issues"].append(
                                {
                                    "column": col,
                                    "type": "STATISTICAL_OUTLIER",
                                    "count": outliers,
                                    "message": (
                                        f"Column '{col}' contains {outliers} severe outliers "
                                        f"(|Z-score| > {ZSCORE_THRESHOLD})."
                                    ),
                                }
                            )

        # ── 3. Categorical unique value tracking ─────────────────
        if col_type == "String":
            unique_vals = df[col].drop_nulls().unique().to_list()
            col_summary["unique_values"] = unique_vals

        profile["columns"][col] = col_summary

    # ── 4. Class imbalance check on target column (if string or not yet caught) ──
    if target_column and target_column in df.columns:
        _check_class_imbalance(df, target_column, profile)

    return profile


def _check_class_imbalance(df: pl.DataFrame, target_col: str, profile: dict[str, Any]) -> None:
    """Checks whether the target column exhibits severe class imbalance."""
    if any(iss["column"] == target_col and iss["type"] == "CLASS_IMBALANCE" for iss in profile["critical_issues"]):
        return

    val_counts = df[target_col].drop_nulls().value_counts().sort("count")
    if val_counts.height < 2:
        return

    total = val_counts["count"].sum()
    min_class_count = val_counts["count"][0]
    min_class_value = val_counts[target_col][0]
    min_ratio = min_class_count / total

    if min_ratio < CLASS_IMBALANCE_MIN:
        if target_col in profile["columns"]:
            profile["columns"][target_col]["anomalies"].append(f"Imbalance ({min_ratio * 100:.1f}%)")
        profile["critical_issues"].append(
            {
                "column": target_col,
                "type": "CLASS_IMBALANCE",
                "minority_ratio": round(min_ratio, 4),
                "minority_value": min_class_value,
                "message": (
                    f"Target column '{target_col}' has severe class imbalance: "
                    f"minority class ({min_class_value}) is only {min_ratio * 100:.1f}% "
                    f"(Threshold: {CLASS_IMBALANCE_MIN * 100:.0f}%)."
                ),
            }
        )
