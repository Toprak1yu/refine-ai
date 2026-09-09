import polars as pl
from refine.profiler.stats import profile_dataset
from refine.tools.cleaner import run_deterministic_clean

def test_profiler_detects_outliers():
    # Standart sapmayı düşük tutmak için 30 adet normal veri ve uç değerler ekliyoruz
    ages = [35] * 30 + [-5, 200]
    salaries = [50000] * 30 + [50000, 100000000]
    
    df = pl.DataFrame({
        "age": ages,
        "salary": salaries
    })
    
    profile = profile_dataset(df)
    
    # Artık hem age (INVALID_BOUNDS) hem de salary (STATISTICAL_OUTLIER) yakalanmalı
    assert len(profile["critical_issues"]) >= 2

def test_cleaner_normalizes_aliases():
    df = pl.DataFrame({"country": ["US", "USA", "United States"]})
    clean_df, _ = run_deterministic_clean(df)
    assert clean_df["country"].to_list() == ["United States", "United States", "United States"]