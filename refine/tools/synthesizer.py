"""Schema-driven synthetic data generator. Produces realistic records from actual column distributions."""

import random
from typing import Any

import numpy as np
import polars as pl
from faker import Faker

from refine.schema_inference import DatasetSchema

fake = Faker("en_US")


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python random, NumPy, and Faker."""
    random.seed(seed)
    np.random.seed(seed)
    fake.seed_instance(seed)
    Faker.seed(seed)


set_seed(42)


def synthesize_minority_class(
    df: pl.DataFrame,
    target_col: str,
    target_ratio: float = 0.35,
    schema: DatasetSchema | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Synthesizes realistic records for the minority class to mitigate class imbalance.

    Uses the inferred DatasetSchema to understand column roles and generate
    appropriate values for every column — no hardcoded column names.
    """
    logs: list[str] = []
    schema = schema or DatasetSchema()
    total_records = df.height

    if total_records == 0:
        logs.append(f"Empty dataset provided for '{target_col}'. Skipped synthesis.")
        return df, logs

    col_schema = schema.columns.get(target_col)
    minority_val = (
        col_schema.minority_class_value
        if col_schema and col_schema.minority_class_value is not None
        else _detect_minority_value(df, target_col)
    )

    if minority_val is None:
        logs.append(f"Could not determine minority class for '{target_col}'. Skipped synthesis.")
        return df, logs

    minority_count = df.filter(pl.col(target_col) == minority_val).height
    current_ratio = minority_count / total_records

    if current_ratio >= target_ratio:
        logs.append(
            f"Class balance already adequate for '{target_col}' ({current_ratio * 100:.1f}%). Skipped synthesis."
        )
        return df, logs

    needed_rows = int((target_ratio * total_records - minority_count) / (1 - target_ratio))
    if needed_rows <= 0:
        return df, logs

    synthetic_rows = _generate_synthetic_rows(df, needed_rows, target_col, minority_val, schema)

    syn_df = pl.DataFrame(synthetic_rows, infer_schema_length=None)

    cast_exprs = []
    for col in syn_df.columns:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(df.schema[col]))
    if cast_exprs:
        syn_df = syn_df.with_columns(cast_exprs)

    updated_df = pl.concat([df, syn_df])
    new_ratio = updated_df.filter(pl.col(target_col) == minority_val).height / updated_df.height

    logs.append(
        f"Synthesized {needed_rows} realistic minority records for '{target_col}'. "
        f"Imbalance corrected from {current_ratio * 100:.1f}% to {new_ratio * 100:.1f}%."
    )

    return updated_df, logs


def synthesize_numerical_feature(
    df: pl.DataFrame,
    column: str,
    valid_bounds: list[float] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Synthesizes valid values for missing or extreme outlier numerical fields."""
    logs: list[str] = []

    lower_bound = valid_bounds[0] if valid_bounds and len(valid_bounds) >= 1 and valid_bounds[0] is not None else 0
    upper_bound = (
        valid_bounds[1] if valid_bounds and len(valid_bounds) >= 2 and valid_bounds[1] is not None else float("inf")
    )

    valid_data = df.filter(
        pl.col(column).is_not_null() & (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
    )[column].to_numpy()

    if len(valid_data) == 0:
        logs.append(f"No valid data in '{column}' to compute distribution. Skipped synthesis.")
        return df, logs

    mean_val = float(np.mean(valid_data))
    std_val = float(np.std(valid_data))

    anomalous_mask = (pl.col(column).is_null()) | (pl.col(column) > upper_bound) | (pl.col(column) < lower_bound)
    anom_count = df.filter(anomalous_mask).height

    if anom_count == 0:
        return df, logs

    raw_syn = np.clip(
        np.random.normal(mean_val, std_val, anom_count),
        lower_bound,
        min(upper_bound, mean_val + 3 * std_val),
    )
    col_dtype = df.schema[column]
    if col_dtype in (pl.Int32, pl.Int64):
        syn_values = np.round(raw_syn).astype(int)
    else:
        syn_values = np.round(raw_syn, 2)

    mask = df.select(anomalous_mask.alias("m")).to_series().to_numpy()
    values = df[column].fill_null(0).to_numpy().copy()
    values[mask] = syn_values
    df = df.with_columns(pl.Series(column, values).cast(df.schema[column]))

    logs.append(
        f"Synthesized and imputed {anom_count} anomalous/missing values in '{column}' "
        f"using Gaussian distribution (mean={mean_val:,.0f}, bounds=[{lower_bound:,.0f}, {upper_bound:,.0f}])."
    )
    return df, logs


def _detect_minority_value(df: pl.DataFrame, target_col: str) -> Any:
    """Auto-detects the minority class value in a column."""
    val_counts = df[target_col].drop_nulls().value_counts().sort("count")
    if val_counts.height == 0:
        return None
    return val_counts[target_col][0]


def _generate_synthetic_rows(
    df: pl.DataFrame,
    n_rows: int,
    target_col: str,
    target_val: Any,
    schema: DatasetSchema,
) -> list[dict[str, Any]]:
    """Generates synthetic rows by sampling from existing column distributions."""
    rows: list[dict[str, Any]] = []
    max_id = df.height + 1000

    generators = _build_column_generators(df, schema, target_col)

    for i in range(n_rows):
        row: dict[str, Any] = {}
        for col in df.columns:
            if col == target_col:
                row[col] = target_val
            elif col in generators:
                row[col] = generators[col](i, max_id)
            else:
                row[col] = None
        rows.append(row)

    return rows


def _build_column_generators(df: pl.DataFrame, schema: DatasetSchema, target_col: str) -> dict[str, Any]:
    """Creates a generator function per column based on its schema role and type."""
    generators: dict[str, Any] = {}

    for col in df.columns:
        if col == target_col:
            continue

        col_schema = schema.columns.get(col)
        col_type = str(df.schema[col])
        role = col_schema.role if col_schema else "feature"
        semantic = (
            col_schema.semantic_type
            if col_schema
            else ("numerical" if col_type in ("Int32", "Int64", "Float32", "Float64") else "text")
        )

        if role == "id":
            if "Int" in col_type:
                generators[col] = lambda i, max_id, _col=col: max_id + i
            else:
                generators[col] = lambda i, max_id, _col=col: f"SYN_{max_id + i}"
            continue

        if role == "ignore":
            if "name" in col.lower():
                generators[col] = lambda i, max_id: fake.name()
            elif "email" in col.lower():
                generators[col] = lambda i, max_id: fake.email()
            elif "address" in col.lower():
                generators[col] = lambda i, max_id: fake.address()
            else:
                generators[col] = lambda i, max_id: None
            continue

        if semantic == "numerical" or col_type in ("Int32", "Int64", "Float32", "Float64"):
            bounds = col_schema.valid_bounds if col_schema else None
            non_null = df[col].drop_nulls().to_numpy()

            if len(non_null) > 0:
                mean = float(np.mean(non_null))
                std = float(np.std(non_null)) or 1.0
                low = bounds[0] if bounds else float(np.min(non_null))
                high = bounds[1] if bounds else float(np.max(non_null))
                is_int = "Int" in col_type

                def _gen_num(i, max_id, _m=mean, _s=std, _l=low, _h=high, _is_int=is_int):
                    val = np.clip(np.random.normal(_m, _s), _l, _h)
                    return int(val) if _is_int else float(val)

                generators[col] = _gen_num
            continue

        if semantic == "categorical" or col_type == "String":
            existing_vals = df[col].drop_nulls().unique().to_list()
            if existing_vals:
                generators[col] = lambda i, max_id, _vals=existing_vals: random.choice(_vals)
            continue

        if semantic == "binary":
            val_counts = df[col].drop_nulls().value_counts()
            values = val_counts[col].to_list()
            counts = val_counts["count"].to_list()
            total = sum(counts)
            if total > 0 and values:
                weights = [c / total for c in counts]

                def _gen_binary(i, max_id, _v=values, _w=weights):
                    return random.choices(_v, weights=_w, k=1)[0]

                generators[col] = _gen_binary

    return generators
