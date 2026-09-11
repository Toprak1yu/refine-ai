"""Schema-driven human resolution executor. Applies strategies based on inferred column types."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from refine.schema_inference import DatasetSchema
from refine.tools.synthesizer import synthesize_minority_class, synthesize_numerical_feature


def apply_human_resolutions(
    df: pl.DataFrame,
    resolutions: Dict[str, str],
    schema: Optional[DatasetSchema] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """Applies human-approved remediation strategies to anomalous features.

    Uses the inferred DatasetSchema to determine valid bounds and column roles
    instead of hardcoded column names.
    """
    audit_logs: List[str] = []
    schema = schema or DatasetSchema()

    for column, strategy in resolutions.items():
        if column not in df.columns:
            audit_logs.append(f"Skipped '{column}': column not found in dataset.")
            continue

        if ":" in strategy:
            cmd, param = strategy.split(":", 1)
            strategy = f"{cmd.upper().strip()}:{param.strip()}"
        else:
            strategy = strategy.upper().strip()
        col_schema = schema.columns.get(column)
        col_type = str(df.schema[column])
        is_numerical = col_type in ("Int32", "Int64", "Float32", "Float64")

        # ── DROP ─────────────────────────────────────────────────
        if strategy == "DROP":
            initial_count = df.height
            drop_conditions = [pl.col(column).is_null()]

            if is_numerical:
                if col_schema and col_schema.valid_bounds and len(col_schema.valid_bounds) == 2:
                    low, high = col_schema.valid_bounds
                    drop_conditions.append((pl.col(column) < low) | (pl.col(column) > high))
                non_nulls = df[column].drop_nulls().to_numpy()
                if len(non_nulls) > 0 and np.std(non_nulls) > 0:
                    mean = float(np.mean(non_nulls))
                    std = float(np.std(non_nulls))
                    drop_conditions.append(((pl.col(column) - mean).abs() / std) > 3.0)

            df = df.filter(~pl.any_horizontal(drop_conditions))
            dropped_count = initial_count - df.height
            audit_logs.append(f"Applied DROP strategy on '{column}': pruned {dropped_count} records.")

        # ── STATISTICAL_IMPUTE ───────────────────────────────────
        elif strategy == "STATISTICAL_IMPUTE":
            if is_numerical:
                df, logs = _impute_numerical(df, column, col_schema)
                audit_logs.extend(logs)
            elif col_type == "String":
                df, logs = _impute_categorical(df, column)
                audit_logs.extend(logs)
            else:
                audit_logs.append(
                    f"STATISTICAL_IMPUTE skipped for '{column}': unsupported type '{col_type}'."
                )

        # ── SYNTHETIC_SYNTHESIS ──────────────────────────────────
        elif strategy == "SYNTHETIC_SYNTHESIS":
            is_target = (
                col_schema and col_schema.role == "target"
            ) or column == schema.target_column

            if is_target:
                target_ratio = 0.35
                df, syn_logs = synthesize_minority_class(
                    df, target_col=column, target_ratio=target_ratio, schema=schema
                )
                audit_logs.extend(syn_logs)
            elif is_numerical:
                bounds = (col_schema.valid_bounds if col_schema else None) or [0, None]
                df, syn_logs = synthesize_numerical_feature(
                    df, column=column, valid_bounds=bounds
                )
                audit_logs.extend(syn_logs)
            else:
                audit_logs.append(
                    f"SYNTHETIC_SYNTHESIS selected for '{column}', "
                    f"but no synthesis method available for type '{col_type}'. Skipped."
                )

        # ── MANUAL_INPUT ─────────────────────────────────────────
        elif strategy.startswith("MANUAL_INPUT"):
            parts = strategy.split(":", 1)
            raw_val = parts[1].strip() if len(parts) > 1 else ""
            if not raw_val:
                audit_logs.append(f"MANUAL_INPUT on '{column}' skipped: no override value provided.")
            else:
                df, logs = _apply_manual_override(df, column, raw_val, col_schema)
                audit_logs.extend(logs)

        else:
            audit_logs.append(f"Unknown strategy '{strategy}' for column '{column}'. Skipped.")

    return df, audit_logs


def _apply_manual_override(
    df: pl.DataFrame,
    column: str,
    raw_val: str,
    col_schema: Optional[object] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """Replaces anomalies (nulls, out-of-bounds, |Z|>3) with an operator-provided manual value."""
    logs: List[str] = []
    col_type = str(df.schema[column])

    try:
        if "Int" in col_type:
            typed_val = int(float(raw_val))
        elif "Float" in col_type:
            typed_val = float(raw_val)
        elif "Boolean" in col_type:
            typed_val = raw_val.lower() in ("true", "1", "yes", "t")
        else:
            typed_val = str(raw_val)
    except Exception as e:
        logs.append(f"MANUAL_INPUT failed on '{column}': could not cast '{raw_val}' to {col_type} ({e}).")
        return df, logs

    # Build anomaly filter for numerical columns or simple null filter for non-numerical
    if col_type in ("Int32", "Int64", "Float32", "Float64"):
        non_nulls = df[column].drop_nulls().to_numpy()
        anomaly_conditions = [pl.col(column).is_null()]

        bounds = getattr(col_schema, "valid_bounds", None) if col_schema else None
        if bounds and len(bounds) == 2:
            low, high = bounds
            anomaly_conditions.append((pl.col(column) < low) | (pl.col(column) > high))

        if len(non_nulls) > 0 and np.std(non_nulls) > 0:
            mean = float(np.mean(non_nulls))
            std = float(np.std(non_nulls))
            anomaly_conditions.append(((pl.col(column) - mean).abs() / std) > 3.0)

        anomaly_filter = pl.any_horizontal(anomaly_conditions)
    else:
        anomaly_filter = pl.col(column).is_null()

    replaced_count = df.filter(anomaly_filter).height

    df = df.with_columns(
        pl.when(anomaly_filter)
        .then(pl.lit(typed_val))
        .otherwise(pl.col(column))
        .alias(column)
    )

    logs.append(f"Applied MANUAL_INPUT on '{column}': replaced {replaced_count} anomalous records with override value '{typed_val}'.")
    return df, logs


def _impute_numerical(
    df: pl.DataFrame,
    column: str,
    col_schema: Optional[object] = None,
) -> Tuple[pl.DataFrame, List[str]]:
    """Imputes anomalous numerical values (nulls, out-of-bounds, |Z|>3) with the median of valid records."""
    logs: List[str] = []

    non_nulls = df[column].drop_nulls().to_numpy()
    if len(non_nulls) == 0:
        logs.append(f"STATISTICAL_IMPUTE skipped for '{column}': no valid values to compute median.")
        return df, logs

    mean = float(np.mean(non_nulls))
    std = float(np.std(non_nulls))

    anomaly_conditions = [pl.col(column).is_null()]

    bounds = getattr(col_schema, "valid_bounds", None) if col_schema else None
    if bounds and len(bounds) == 2:
        low, high = bounds
        anomaly_conditions.append((pl.col(column) < low) | (pl.col(column) > high))

    if std > 0:
        anomaly_conditions.append(((pl.col(column) - mean).abs() / std) > 3.0)

    anomaly_filter = pl.any_horizontal(anomaly_conditions)
    valid_vals = df.filter(~anomaly_filter)[column].to_numpy()
    if len(valid_vals) == 0:
        valid_vals = non_nulls

    median_val = float(np.median(valid_vals))
    col_dtype = str(df.schema[column])
    if "Int" in col_dtype:
        median_val = int(median_val)

    df = df.with_columns(
        pl.when(anomaly_filter)
        .then(median_val)
        .otherwise(pl.col(column))
        .alias(column)
    )

    logs.append(f"Applied STATISTICAL_IMPUTE on '{column}': imputed anomalies with median ({median_val}).")
    return df, logs


def _impute_categorical(
    df: pl.DataFrame, column: str
) -> Tuple[pl.DataFrame, List[str]]:
    """Imputes missing categorical values with the mode (most frequent value)."""
    logs: List[str] = []

    mode_df = df[column].drop_nulls().value_counts().sort("count", descending=True)
    if mode_df.height == 0:
        logs.append(f"STATISTICAL_IMPUTE skipped for '{column}': no valid values to compute mode.")
        return df, logs

    mode_val = mode_df[column][0]

    df = df.with_columns(
        pl.when(pl.col(column).is_null())
        .then(pl.lit(mode_val))
        .otherwise(pl.col(column))
        .alias(column)
    )

    logs.append(f"Applied STATISTICAL_IMPUTE on '{column}': imputed nulls with mode ('{mode_val}').")
    return df, logs