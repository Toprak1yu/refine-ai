from typing import Any, Dict, List
import polars as pl
import numpy as np

def profile_dataset(df: pl.DataFrame) -> Dict[str, Any]:
    """Deterministically analyzes missingness, statistical distributions, and anomalies."""
    total_rows = df.height
    profile: Dict[str, Any] = {
        "total_rows": total_rows,
        "columns": {},
        "critical_issues": []
    }
    
    for col in df.columns:
        col_type = str(df.schema[col])
        null_count = df[col].null_count()
        null_ratio = null_count / total_rows
        
        col_summary: Dict[str, Any] = {
            "type": col_type,
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
            "outliers_count": 0,
            "anomalies": []
        }
        
        # 1. High Null Ratio Trigger (>= 20%)
        if null_ratio >= 0.20:
            profile["critical_issues"].append({
                "column": col,
                "type": "HIGH_NULL_RATIO",
                "ratio": null_ratio,
                "message": f"Column '{col}' has a missingness ratio of {null_ratio*100:.1f}% (Threshold: 20%)."
            })
            
        # 2. Numerical Checks (Bounds & Z-Score)
        if col_type in ["Int32", "Int64", "Float32", "Float64"]:
            non_nulls = df[col].drop_nulls().to_numpy()
            if len(non_nulls) > 0:
                if col == "age":
                    invalid_ages = (non_nulls < 0) | (non_nulls > 120)
                    invalid_count = int(np.sum(invalid_ages))
                    if invalid_count > 0:
                        profile["critical_issues"].append({
                            "column": col,
                            "type": "INVALID_BOUNDS",
                            "count": invalid_count,
                            "message": f"Column '{col}' contains {invalid_count} negative or unrealistic values."
                        })
                
                # Z-Score Outlier Check (|Z| > 3.0)
                mean = np.mean(non_nulls)
                std = np.std(non_nulls)
                if std > 0:
                    z_scores = np.abs((non_nulls - mean) / std)
                    outliers = int(np.sum(z_scores > 3.0))
                    col_summary["outliers_count"] = outliers
                    if outliers > 0:
                        profile["critical_issues"].append({
                            "column": col,
                            "type": "STATISTICAL_OUTLIER",
                            "count": outliers,
                            "message": f"Column '{col}' contains {outliers} severe outliers (|Z-score| > 3.0)."
                        })
                        
        if col_type == "String" and col in ["country", "churn"]:
            unique_vals = df[col].drop_nulls().unique().to_list()
            col_summary["unique_values"] = unique_vals
            
        profile["columns"][col] = col_summary
        
    return profile