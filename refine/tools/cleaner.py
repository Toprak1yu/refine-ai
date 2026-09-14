"""Schema-driven deterministic cleaner. Applies alias mappings from inferred schema."""

import polars as pl

from refine.schema_inference import DatasetSchema, infer_schema


def run_deterministic_clean(df: pl.DataFrame, schema: DatasetSchema | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Applies unambiguous sanitation operations without requiring human consent.

    When a DatasetSchema is provided, alias mappings are read from the schema.
    If schema is None, it is automatically inferred.
    """
    logs: list[str] = []

    if schema is None:
        schema, _ = infer_schema(df)

    string_cols = [col for col in df.columns if df.schema[col] == pl.String]
    if string_cols:
        df = df.with_columns([pl.col(col).str.strip_chars() for col in string_cols])
        logs.append(f"Applied whitespace stripping across {len(string_cols)} string features.")

    if schema:
        for col_name, col_schema in schema.columns.items():
            if col_name not in df.columns:
                continue
            aliases = col_schema.canonical_aliases
            if not aliases:
                continue

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
