"""Schema-driven deterministic cleaner. Applies alias mappings from inferred schema."""

from typing import List, Optional, Tuple

import polars as pl
from refine.schema_inference import DatasetSchema, infer_schema


def run_deterministic_clean(
    df: pl.DataFrame, schema: Optional[DatasetSchema] = None
) -> Tuple[pl.DataFrame, List[str]]:
    """Applies unambiguous sanitation operations without requiring human consent.

    When a DatasetSchema is provided, alias mappings are read from the schema.
    If schema is None, it is automatically inferred.
    """
    logs: List[str] = []

    if schema is None:
        schema, _ = infer_schema(df)

    # 1. Strip whitespace across all string columns
    string_cols = [col for col in df.columns if df.schema[col] == pl.String]
    if string_cols:
        df = df.with_columns([pl.col(col).str.strip_chars() for col in string_cols])
        logs.append(f"Applied whitespace stripping across {len(string_cols)} string features.")

    # 2. Apply categorical alias normalization from schema
    if schema:
        for col_name, col_schema in schema.columns.items():
            if col_name not in df.columns:
                continue
            aliases = col_schema.canonical_aliases
            if not aliases:
                continue

            # Build a flat mapping: alias -> canonical value
            alias_mapping: dict[str, str] = {}
            for canonical, alias_list in aliases.items():
                for alias in alias_list:
                    alias_mapping[alias] = canonical

            if alias_mapping:
                df = df.with_columns(pl.col(col_name).replace(alias_mapping))
                logs.append(
                    f"Normalized {len(alias_mapping)} aliases in '{col_name}' "
                    f"to canonical values: {list(aliases.keys())}."
                )

    return df, logs