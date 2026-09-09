import polars as pl
from typing import Tuple, List

def run_deterministic_clean(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """Applies unambiguous sanitation operations without requiring human consent."""
    logs: List[str] = []
    
    # 1. Strip whitespace across all string columns
    for col in df.columns:
        if df.schema[col] == pl.String:
            df = df.with_columns(pl.col(col).str.strip_chars())
            
    logs.append("Applied whitespace stripping across all string features.")
    
    # 2. Canonicalize standard country variations
    if "country" in df.columns:
        country_mapping = {
            "US": "United States",
            "USA": "United States",
            "united states": "United States",
            "US_OFFICIAL": "United States"
        }
        df = df.with_columns(
            pl.col("country").replace(country_mapping)
        )
        logs.append("Normalized country aliases to canonical 'United States'.")
        
    return df, logs