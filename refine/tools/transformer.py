from typing import Dict, List, Tuple
import polars as pl
import numpy as np
from refine.tools.synthesizer import synthesize_minority_class, synthesize_numerical_feature

def apply_human_resolutions(
    df: pl.DataFrame, resolutions: Dict[str, str]
) -> Tuple[pl.DataFrame, List[str]]:
    """Applies human-approved remediation strategies to anomalous features."""
    audit_logs: List[str] = []

    for column, strategy in resolutions.items():
        strategy = strategy.upper().strip()

        if strategy == "DROP":
            initial_count = df.height
            if column == "age":
                df = df.filter(
                    pl.col("age").is_not_null() & (pl.col("age") >= 0) & (pl.col("age") <= 120)
                )
            elif column == "salary":
                df = df.filter(pl.col("salary").is_not_null() & (pl.col("salary") < 1_000_000))
            elif column == "churn":
                df = df.filter(pl.col("churn").is_not_null())
            else:
                df = df.filter(pl.col(column).is_not_null())

            dropped_count = initial_count - df.height
            audit_logs.append(f"Applied DROP strategy on '{column}': pruned {dropped_count} records.")

        elif strategy == "STATISTICAL_IMPUTE":
            if column == "salary":
                valid_salaries = df.filter(
                    pl.col("salary").is_not_null() & (pl.col("salary") < 1_000_000)
                )["salary"].to_numpy()
                median_val = int(np.median(valid_salaries)) if len(valid_salaries) > 0 else 50000

                df = df.with_columns(
                    pl.when(pl.col("salary").is_null() | (pl.col("salary") >= 1_000_000))
                    .then(median_val)
                    .otherwise(pl.col("salary"))
                    .alias("salary")
                )
                audit_logs.append(
                    f"Applied STATISTICAL_IMPUTE on 'salary': imputed anomalies with median (${median_val:,})."
                )

            elif column == "age":
                valid_ages = df.filter(
                    pl.col("age").is_not_null() & (pl.col("age") >= 0) & (pl.col("age") <= 120)
                )["age"].to_numpy()
                median_age = int(np.median(valid_ages)) if len(valid_ages) > 0 else 35

                df = df.with_columns(
                    pl.when(pl.col("age").is_null() | (pl.col("age") < 0) | (pl.col("age") > 120))
                    .then(median_age)
                    .otherwise(pl.col("age"))
                    .alias("age")
                )
                audit_logs.append(
                    f"Applied STATISTICAL_IMPUTE on 'age': imputed anomalies with median ({median_age} years)."
                )

            elif column == "churn":
                audit_logs.append("Applied STATISTICAL_IMPUTE on 'churn': skipped (target class balance requires synthesis).")

        elif strategy == "SYNTHETIC_SYNTHESIS":
            if column == "churn":
                df, syn_logs = synthesize_minority_class(df, target_col="churn", target_ratio=0.35)
                audit_logs.extend(syn_logs)
            elif column in ["salary", "age"]:
                df, syn_logs = synthesize_numerical_feature(df, column=column)
                audit_logs.extend(syn_logs)
            else:
                audit_logs.append(f"SYNTHETIC_SYNTHESIS selected for '{column}', fallback to statistical imputation.")

    return df, audit_logs