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


def test_schema_inference_heuristic():
    from refine.schema_inference import _infer_schema_heuristic

    df = pl.DataFrame({
        "transaction_id": [f"TX_{i}" for i in range(50)],
        "description": [f"Desc {i}" for i in range(50)],
        "amount": [100.0] * 48 + [-10.0, 99999.0],
        "is_fraud": [0] * 46 + [1] * 4,
    })

    schema = _infer_schema_heuristic(df)
    assert "transaction_id" in schema.id_columns
    assert "description" in schema.ignore_columns
    assert schema.target_column == "is_fraud"
    assert "amount" in schema.columns
    assert schema.columns["amount"].semantic_type == "numerical"


def test_generic_pipeline_arbitrary_dataset(tmp_path):
    from refine.graph import build_pipeline_graph
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver

    df = pl.DataFrame({
        "order_id": [f"ORD_{i}" for i in range(60)],
        "amount": [100.0] * 58 + [50000.0, None],
        "status": [1] * 55 + [0] * 5,
    })

    in_file = str(tmp_path / "orders.csv")
    out_file = str(tmp_path / "clean_orders.csv")
    df.write_csv(in_file)

    checkpointer = MemorySaver()
    graph = build_pipeline_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test_session"}}

    initial_state = {
        "raw_file_path": in_file,
        "processed_file_path": out_file,
        "initial_row_count": 0,
        "records": [],
        "inferred_schema": None,
        "schema_method": None,
        "profile": None,
        "critical_issues": [],
        "expert_advice": None,
        "human_resolutions": {},
        "audit_trail": [],
        "is_completed": False,
    }

    graph.invoke(initial_state, config=config)
    snapshot = graph.get_state(config)
    assert bool(snapshot.tasks and any(t.interrupts for t in snapshot.tasks))

    # Resume graph
    resolutions = {"amount": "STATISTICAL_IMPUTE", "status": "SYNTHETIC_SYNTHESIS"}
    graph.invoke(Command(resume=resolutions), config=config)

    clean_df = pl.read_csv(out_file)
    assert clean_df["amount"].null_count() == 0
    assert clean_df["amount"].max() < 50000.0
    assert clean_df.height > 60  # minority class synthesized


def test_manual_input_strategy():
    from refine.tools.transformer import apply_human_resolutions
    from refine.schema_inference import DatasetSchema, ColumnSchema

    df = pl.DataFrame({
        "score": [50, 55, 60, None, 9999],
        "category": ["A", "B", None, "A", "B"],
    })

    schema = DatasetSchema(
        columns={
            "score": ColumnSchema(role="feature", semantic_type="numerical", valid_bounds=[0, 100]),
            "category": ColumnSchema(role="feature", semantic_type="categorical"),
        }
    )

    resolutions = {
        "score": "MANUAL_INPUT:75",
        "category": "MANUAL_INPUT:DefaultCat",
    }

    clean_df, logs = apply_human_resolutions(df, resolutions, schema)

    # Score anomalies (None and 9999) should be replaced with 75
    assert clean_df["score"].to_list() == [50, 55, 60, 75, 75]
    # Category null should be replaced with 'DefaultCat'
    assert clean_df["category"].to_list() == ["A", "B", "DefaultCat", "A", "B"]
    assert any("override value '75'" in log for log in logs)
    assert any("override value 'DefaultCat'" in log for log in logs)