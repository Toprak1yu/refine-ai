"""AI-driven dataset schema inference using LLM (Ollama) with heuristic fallback."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Schema Models
# ────────────────────────────────────────────────────────────────────

class ColumnSchema(BaseModel):
    """Schema definition for a single dataset column."""
    role: str = Field(description="One of: id, target, feature, ignore")
    semantic_type: str = Field(description="One of: numerical, categorical, binary, text")
    valid_bounds: Optional[List[float]] = Field(
        default=None, description="[min, max] valid domain range for numerical columns"
    )
    canonical_aliases: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Canonical value -> list of known aliases for categorical columns"
    )
    minority_class_value: Optional[Any] = Field(
        default=None, description="Minority class value for binary target columns"
    )
    description: Optional[str] = None


class DatasetSchema(BaseModel):
    """Complete inferred schema for a dataset."""
    id_columns: List[str] = []
    target_column: Optional[str] = None
    ignore_columns: List[str] = []
    columns: Dict[str, ColumnSchema] = {}


# ────────────────────────────────────────────────────────────────────
# Dataset Summary Builder (LLM'e gönderilecek özet)
# ────────────────────────────────────────────────────────────────────

def _build_dataset_summary(df: pl.DataFrame) -> str:
    """Builds a concise text summary of the dataset for LLM consumption."""
    lines: List[str] = []
    lines.append(f"Total rows: {df.height}, Total columns: {df.width}")
    lines.append("")
    lines.append(f"{'Column':<25} {'Type':<10} {'Nulls':<8} {'Null%':<8} {'Unique':<8} {'Sample Values'}")
    lines.append("-" * 95)

    for col in df.columns:
        col_type = str(df.schema[col])
        null_count = df[col].null_count()
        null_pct = f"{(null_count / df.height) * 100:.1f}%"
        unique_count = df[col].drop_nulls().n_unique()

        samples = df[col].drop_nulls().head(5).to_list()
        sample_str = str(samples)
        if len(sample_str) > 60:
            sample_str = sample_str[:60] + "..."

        lines.append(f"{col:<25} {col_type:<10} {null_count:<8} {null_pct:<8} {unique_count:<8} {sample_str}")

    # Numerical column statistics
    numerical_cols = [
        col for col in df.columns
        if str(df.schema[col]) in ("Int32", "Int64", "Float32", "Float64")
    ]
    if numerical_cols:
        lines.append("")
        lines.append("Numerical Statistics:")
        for col in numerical_cols:
            vals = df[col].drop_nulls().to_numpy()
            if len(vals) > 0:
                lines.append(
                    f"  {col}: min={np.min(vals)}, max={np.max(vals)}, "
                    f"mean={np.mean(vals):.2f}, median={np.median(vals):.2f}, std={np.std(vals):.2f}"
                )

    # First 5 rows as sample
    lines.append("")
    lines.append("Sample rows (first 5):")
    for row in df.head(5).to_dicts():
        lines.append(str(row))

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# LLM-Based Schema Inference (Ollama)
# ────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are a Senior Data Architect. Analyze the dataset summary below and output a JSON schema that describes the structure of this dataset.

Your JSON output MUST follow this exact structure:
{
  "id_columns": ["list of identifier/key columns"],
  "target_column": "ML target/label column name, or null if none detected",
  "ignore_columns": ["columns to skip: free-text names, addresses, etc."],
  "columns": {
    "column_name": {
      "role": "id | target | feature | ignore",
      "semantic_type": "numerical | categorical | binary | text",
      "valid_bounds": [min, max],
      "canonical_aliases": {"CanonicalValue": ["alias1", "alias2"]},
      "minority_class_value": 1,
      "description": "brief one-line description"
    }
  }
}

Rules:
- "valid_bounds": ONLY for numerical features. Use real-world domain knowledge (e.g. human age: [0, 120], salary in USD: [0, 500000]).
- "canonical_aliases": ONLY for categorical features with inconsistent spellings or abbreviations. Map the canonical form to all its known aliases.
- "minority_class_value": ONLY for the target column when class imbalance exists.
- Detect ID columns by name patterns (id, key, code) AND high uniqueness.
- Detect ignore columns: personal names, free text, addresses — anything with very high cardinality and no analytical value.
- Output ONLY valid JSON. No markdown fences, no explanation text."""


import urllib.request


def _is_ollama_online(base_url: str) -> bool:
    """Fast check (timeout 0.3s) if local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.3):
            return True
    except Exception:
        return False


def _infer_schema_with_llm(
    df: pl.DataFrame, model_name: Optional[str] = None
) -> Optional[DatasetSchema]:
    """Attempts to infer dataset schema using a local Ollama LLM."""
    model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Fast-fail if Ollama is not active to prevent slow timeouts
    if not _is_ollama_online(base_url):
        return None

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOllama(model=model, base_url=base_url, temperature=0.0, timeout=10.0)

        summary = _build_dataset_summary(df)
        user_prompt = f"Analyze this dataset and return the JSON schema:\n\n{summary}"

        response = llm.invoke([
            SystemMessage(content=_LLM_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        raw = response.content.strip()

        # Extract JSON from potential markdown code fences
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()

        schema_dict = json.loads(raw)
        return _parse_schema_dict(schema_dict, df)

    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# Heuristic Fallback (Model-free inference)
# ────────────────────────────────────────────────────────────────────

# Name patterns used for heuristic column role detection
_ID_PATTERNS = re.compile(
    r"(?:^id$|_id$|^index$|^key$|_key$|^uuid$|^pk$|product_code|item_code|customer_code)",
    re.IGNORECASE,
)
_IGNORE_PATTERNS = re.compile(
    r"(?:name|description|comment|note|text|address|detail|summary)",
    re.IGNORECASE,
)
_TARGET_PATTERNS = re.compile(
    r"(?:target|label|class|churn|outcome|^y$|result|flag|status|converted|survived|default|fraud)",
    re.IGNORECASE,
)


def _infer_schema_heuristic(df: pl.DataFrame) -> DatasetSchema:
    """Pure rule-based schema inference — no external model required."""
    columns: Dict[str, ColumnSchema] = {}
    id_columns: List[str] = []
    ignore_columns: List[str] = []
    target_candidates: List[str] = []

    for col in df.columns:
        col_type = str(df.schema[col])
        null_count = df[col].null_count()
        non_null = df[col].drop_nulls()
        unique_count = non_null.n_unique()
        total_non_null = non_null.len()
        is_numerical = col_type in ("Int32", "Int64", "Float32", "Float64")
        is_string = col_type == "String"
        uniqueness_ratio = (unique_count / total_non_null) if total_non_null > 0 else 0.0

        # ── 1. ID columns (explicit ID patterns + reasonable uniqueness) ──
        if _ID_PATTERNS.search(col) and (uniqueness_ratio > 0.7 or total_non_null < 10):
            id_columns.append(col)
            columns[col] = ColumnSchema(
                role="id",
                semantic_type="text" if is_string else "numerical",
                description=f"Identifier column ({unique_count} unique values)",
            )
            continue

        # ── 2. Ignore columns (free text, comments, names, addresses) ────
        if is_string and (
            _IGNORE_PATTERNS.search(col)
            or (uniqueness_ratio > 0.7 and unique_count > 30)
        ):
            ignore_columns.append(col)
            columns[col] = ColumnSchema(
                role="ignore",
                semantic_type="text",
                description=f"High-cardinality or free text ({unique_count} unique values)",
            )
            continue

        # ── 3. Binary columns (numerical or string, 2 unique values) ──
        if unique_count == 2 and total_non_null > 5:
            minority_val = _detect_minority_class(df, col)
            target_candidates.append(col)
            columns[col] = ColumnSchema(
                role="feature",  # may be promoted to target below
                semantic_type="binary",
                minority_class_value=minority_val,
                description=f"Binary feature ({unique_count} unique values)",
            )
            continue

        # ── Numerical features ───────────────────────────────────
        if is_numerical:
            bounds = _estimate_bounds(df, col)
            columns[col] = ColumnSchema(
                role="feature",
                semantic_type="numerical",
                valid_bounds=bounds,
                description=f"Numerical feature ({unique_count} unique values)",
            )
            continue

        # ── Categorical features ─────────────────────────────────
        if is_string:
            aliases = _detect_aliases_heuristic(non_null.unique().to_list())
            columns[col] = ColumnSchema(
                role="feature",
                semantic_type="categorical",
                canonical_aliases=aliases if aliases else None,
                description=f"Categorical feature ({unique_count} unique values)",
            )
            continue

        # ── Fallback ─────────────────────────────────────────────
        columns[col] = ColumnSchema(
            role="feature",
            semantic_type="numerical" if is_numerical else "categorical",
        )

    # ── Select target column from candidates ─────────────────────
    target_column = _select_target(target_candidates)
    if target_column and target_column in columns:
        columns[target_column].role = "target"

    return DatasetSchema(
        id_columns=id_columns,
        target_column=target_column,
        ignore_columns=ignore_columns,
        columns=columns,
    )


def _detect_minority_class(df: pl.DataFrame, col: str) -> Any:
    """Finds the minority class value in a binary column."""
    val_counts = df[col].drop_nulls().value_counts().sort("count")
    return val_counts[col][0]


def _estimate_bounds(df: pl.DataFrame, col: str) -> Optional[List[float]]:
    """Estimates valid domain bounds using Tukey's robust IQR (Interquartile Range) method.
    Purely mathematical; contains zero hardcoded column names.
    """
    vals = df[col].drop_nulls().to_numpy()
    if len(vals) == 0:
        return None

    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    iqr = float(q75 - q25)

    multiplier = 1.5 if iqr > 0 else 0.0
    lower = q25 - multiplier * iqr if multiplier > 0 else float(np.min(vals))
    upper = q75 + multiplier * iqr if multiplier > 0 else float(np.max(vals))

    # If 95%+ of values are non-negative, data is non-negative by nature (e.g. counts, duration, physical metrics)
    if (np.sum(vals >= 0) / len(vals)) >= 0.95 and lower < 0:
        lower = 0.0

    return [round(float(lower), 2), round(float(upper), 2)]


def _detect_aliases_heuristic(
    values: List[str], col_name: str = ""
) -> Optional[Dict[str, List[str]]]:
    """Groups categorical variants purely algorithmically (case, punctuation, and multi-word acronyms).
    Contains zero hardcoded dictionaries, country names, or entity names.
    """
    if not values or len(values) <= 1:
        return None

    aliases: Dict[str, List[str]] = {}
    assigned: set[str] = set()

    # 1. Group by cleaned alphanumeric lowercase key (case/punctuation variants)
    groups: Dict[str, List[str]] = {}
    for val in values:
        clean_key = re.sub(r"[^a-zA-Z0-9]", "", val).lower()
        if clean_key:
            groups.setdefault(clean_key, []).append(val)

    for _key, variants in groups.items():
        if len(variants) > 1:
            canonical = max(variants, key=lambda v: (v == v.title(), len(v)))
            alias_list = [v for v in variants if v != canonical]
            if alias_list:
                aliases[canonical] = alias_list
                assigned.update(variants)

    # 2. Algorithmic multi-word acronym matching (e.g. initials of multi-word string matching short codes)
    for val in values:
        words = [w for w in re.split(r"[\s_\-]+", val.strip()) if w]
        if len(words) > 1:
            initials = "".join(w[0] for w in words).upper()
            # Match values that are the exact initials, or initials with prefix/suffix (e.g. US, USA, US_OFFICIAL)
            matching = [
                v for v in values
                if v not in assigned and (
                    v.strip().upper() == initials
                    or (len(initials) >= 2 and v.strip().upper().startswith(initials) and len(v.strip()) <= len(initials) + 1)
                    or (v.strip().upper().startswith(f"{initials}_"))
                )
            ]
            if matching:
                aliases.setdefault(val, []).extend(matching)
                assigned.update(matching)

    return aliases if aliases else None


def _select_target(candidates: List[str]) -> Optional[str]:
    """Selects the most likely target column from binary candidates."""
    if not candidates:
        return None

    # Prefer columns whose name suggests a target/label
    named = [c for c in candidates if _TARGET_PATTERNS.search(c)]
    if named:
        return named[0]

    # Otherwise pick the first binary candidate
    return candidates[0]


# ────────────────────────────────────────────────────────────────────
# Schema Dict Parser (shared by LLM and external configs)
# ────────────────────────────────────────────────────────────────────

def _parse_schema_dict(schema_dict: Dict[str, Any], df: pl.DataFrame) -> DatasetSchema:
    """Converts a raw JSON dict (from LLM or file) into a validated DatasetSchema."""
    valid_columns = set(df.columns)
    columns: Dict[str, ColumnSchema] = {}

    for col_name, col_info in schema_dict.get("columns", {}).items():
        if col_name not in valid_columns:
            continue
        columns[col_name] = ColumnSchema(
            role=col_info.get("role", "feature"),
            semantic_type=col_info.get("semantic_type", "numerical"),
            valid_bounds=col_info.get("valid_bounds"),
            canonical_aliases=col_info.get("canonical_aliases"),
            minority_class_value=col_info.get("minority_class_value"),
            description=col_info.get("description"),
        )

    return DatasetSchema(
        id_columns=[c for c in schema_dict.get("id_columns", []) if c in valid_columns],
        target_column=(
            schema_dict["target_column"]
            if schema_dict.get("target_column") in valid_columns
            else None
        ),
        ignore_columns=[c for c in schema_dict.get("ignore_columns", []) if c in valid_columns],
        columns=columns,
    )


# ────────────────────────────────────────────────────────────────────
# Public Entry Point
# ────────────────────────────────────────────────────────────────────

def infer_schema(
    df: pl.DataFrame, model_name: Optional[str] = None
) -> Tuple[DatasetSchema, str]:
    """
    Infers the dataset schema using LLM first, falling back to heuristics.

    Args:
        df: The raw Polars DataFrame to analyze.
        model_name: Optional Ollama model name override.

    Returns:
        A tuple of (DatasetSchema, inference_method) where inference_method
        is ``"llm"`` or ``"heuristic"``.
    """
    # 1. Try LLM inference
    schema = _infer_schema_with_llm(df, model_name)
    if schema:
        return schema, "llm"

    # 2. Fallback to deterministic heuristics
    schema = _infer_schema_heuristic(df)
    return schema, "heuristic"
