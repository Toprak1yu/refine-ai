import polars as pl

from refine.profiler.stats import profile_dataset
from refine.tools.cleaner import run_deterministic_clean


def test_profiler_detects_outliers():
    ages = [35] * 30 + [-5, 200]
    salaries = [50000] * 30 + [50000, 100000000]

    df = pl.DataFrame({"age": ages, "salary": salaries})

    profile = profile_dataset(df)

    assert len(profile["critical_issues"]) >= 2


def test_cleaner_normalizes_aliases():
    df = pl.DataFrame({"country": ["US", "USA", "United States"]})
    clean_df, _ = run_deterministic_clean(df)
    assert clean_df["country"].to_list() == ["United States", "United States", "United States"]


def test_schema_inference_heuristic():
    from refine.schema_inference import _infer_schema_heuristic

    df = pl.DataFrame(
        {
            "transaction_id": [f"TX_{i}" for i in range(50)],
            "description": [f"Desc {i}" for i in range(50)],
            "amount": [100.0] * 48 + [-10.0, 99999.0],
            "is_fraud": [0] * 46 + [1] * 4,
        }
    )

    schema = _infer_schema_heuristic(df)
    assert "transaction_id" in schema.id_columns
    assert "description" in schema.ignore_columns
    assert schema.target_column == "is_fraud"
    assert "amount" in schema.columns
    assert schema.columns["amount"].semantic_type == "numerical"


def test_generic_pipeline_arbitrary_dataset(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from refine.graph import build_pipeline_graph

    df = pl.DataFrame(
        {
            "order_id": [f"ORD_{i}" for i in range(60)],
            "amount": [100.0] * 58 + [50000.0, None],
            "status": [1] * 55 + [0] * 5,
        }
    )

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

    resolutions = {"amount": "STATISTICAL_IMPUTE", "status": "SYNTHETIC_SYNTHESIS"}
    graph.invoke(Command(resume=resolutions), config=config)

    clean_df = pl.read_csv(out_file)
    assert clean_df["amount"].null_count() == 0
    assert clean_df["amount"].max() < 50000.0
    assert clean_df.height > 60


def test_manual_input_strategy():
    from refine.schema_inference import ColumnSchema, DatasetSchema
    from refine.tools.transformer import apply_human_resolutions

    df = pl.DataFrame(
        {
            "score": [50, 55, 60, None, 9999],
            "category": ["A", "B", None, "A", "B"],
        }
    )

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

    assert clean_df["score"].to_list() == [50, 55, 60, 75, 75]
    assert clean_df["category"].to_list() == ["A", "B", "DefaultCat", "A", "B"]
    assert any("override value '75'" in log for log in logs)
    assert any("override value 'DefaultCat'" in log for log in logs)


def test_seed_reproducibility():
    from refine.schema_inference import ColumnSchema, DatasetSchema
    from refine.tools.synthesizer import set_seed, synthesize_minority_class

    df = pl.DataFrame(
        {
            "id": list(range(20)),
            "score": [50.0 + i for i in range(20)],
            "label": [0] * 18 + [1] * 2,
        }
    )
    schema = DatasetSchema(
        columns={
            "id": ColumnSchema(role="id", semantic_type="id"),
            "score": ColumnSchema(role="feature", semantic_type="numerical"),
            "label": ColumnSchema(role="target", semantic_type="categorical"),
        },
        id_columns=["id"],
        target_column="label",
    )

    set_seed(42)
    syn1, _ = synthesize_minority_class(df, "label", target_ratio=0.3, schema=schema)

    set_seed(42)
    syn2, _ = synthesize_minority_class(df, "label", target_ratio=0.3, schema=schema)

    set_seed(999)
    syn3, _ = synthesize_minority_class(df, "label", target_ratio=0.3, schema=schema)

    assert syn1["score"].to_list() == syn2["score"].to_list()
    assert syn1["score"].to_list() != syn3["score"].to_list()


def test_input_validation(tmp_path):
    from pathlib import Path

    import pytest
    import typer

    from refine.cli import _validate_input_file

    with pytest.raises(typer.Exit):
        _validate_input_file(Path(tmp_path / "non_existent.csv"))

    empty_file = Path(tmp_path / "empty.csv")
    empty_file.touch()
    with pytest.raises(typer.Exit):
        _validate_input_file(empty_file)

    corrupt_file = Path(tmp_path / "corrupt.csv")
    corrupt_file.write_text("col1,col2\n1\n2,3,4,5\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        _validate_input_file(corrupt_file)


def test_cli_profile_command(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    sample_csv = tmp_path / "test_sample.csv"
    df = pl.DataFrame({"x": [1, 2, 3, 1000], "y": ["a", "b", "c", "d"]})
    df.write_csv(str(sample_csv))

    result = runner.invoke(app, ["profile", str(sample_csv)])
    assert result.exit_code == 0
    assert "Profiling dataset:" in result.output
    assert "x" in result.output
    assert "y" in result.output


def test_cli_reset_command(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from refine.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    (tmp_path / ".checkpoints.db").write_text("mock sqlite")
    (tmp_path / ".checkpoints.db-wal").write_text("mock wal")

    result = runner.invoke(app, ["reset", "--yes"])
    assert result.exit_code == 0
    assert "Checkpoints reset successfully." in result.output
    assert "Deleted: .checkpoints.db" in result.output
    assert not (tmp_path / ".checkpoints.db").exists()
    assert not (tmp_path / ".checkpoints.db-wal").exists()


def test_post_remediation_and_telemetry(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from refine.graph import build_pipeline_graph

    df = pl.DataFrame(
        {
            "id": [f"ID_{i}" for i in range(50)],
            "val": [10.0] * 48 + [-999.0, 999.0],
            "target": [1] * 45 + [0] * 5,
        }
    )
    in_file = str(tmp_path / "input.csv")
    out_file = str(tmp_path / "output.csv")
    df.write_csv(in_file)

    checkpointer = MemorySaver()
    graph = build_pipeline_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session_telemetry_test"}}

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
        "session_id": "session_telemetry_test",
        "random_seed": 1234,
        "start_time": "2026-09-11T12:00:00",
        "end_time": None,
        "execution_duration_sec": None,
    }

    graph.invoke(initial_state, config=config)
    graph.invoke(
        Command(resume={"val": "STATISTICAL_IMPUTE", "target": "SYNTHETIC_SYNTHESIS"}),
        config=config,
    )

    final_state = graph.get_state(config).values
    assert final_state["is_completed"] is True
    assert final_state["session_id"] == "session_telemetry_test"
    assert final_state["random_seed"] == 1234
    assert final_state["execution_duration_sec"] is not None
    assert final_state["execution_duration_sec"] >= 0

    trail = " ".join(final_state["audit_trail"])
    assert "Verification" in trail or "clean" in trail.lower()


def test_get_recommended_strategies():
    from refine.tools.advisor import get_recommended_strategies

    issues = [
        {"column": "age", "type": "INVALID_BOUNDS"},
        {"column": "salary", "type": "STATISTICAL_OUTLIER"},
        {"column": "churn", "type": "CLASS_IMBALANCE"},
    ]

    strat = get_recommended_strategies(issues)
    assert strat["age"] == "STATISTICAL_IMPUTE"
    assert strat["salary"] == "SYNTHETIC_SYNTHESIS"
    assert strat["churn"] == "SYNTHETIC_SYNTHESIS"

    custom_advice = "• 'salary':\n  ➜ Recommended Decision: STATISTICAL_IMPUTE\n  ➜ Decision Impact: test."
    strat2 = get_recommended_strategies(issues, advice_text=custom_advice)
    assert strat2["salary"] == "STATISTICAL_IMPUTE"


def test_generate_expert_advice_unified_column_recommendation(monkeypatch):
    from refine.tools.advisor import generate_expert_advice

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    issues = [
        {"column": "age", "type": "INVALID_BOUNDS", "count": 32},
        {"column": "age", "type": "STATISTICAL_OUTLIER", "outliers_count": 15},
        {"column": "churn", "type": "CLASS_IMBALANCE", "minority_ratio": 0.08},
    ]
    advice = generate_expert_advice({}, issues)

    assert advice.count("• 'age'") == 1
    assert "Invalid Bounds & Statistical Outliers" in advice
    assert "Recommended Decision: STATISTICAL_IMPUTE" in advice
    assert advice.count("• 'churn'") == 1


def test_build_execution_manifest():
    from refine.profiler.reporters import build_execution_manifest

    manifest = build_execution_manifest(
        raw_path="data/raw/test.csv",
        processed_path="data/processed/clean.csv",
        session_id="test_sess",
        total_rows=100,
        audit_trail=["Normalized 5 aliases in 'country' to canonical values."],
        critical_issues=[{"column": "age", "type": "STATISTICAL_OUTLIER"}],
        recommended_strategies={"age": "STATISTICAL_IMPUTE"},
        profile={"columns": {"age": {"outliers_count": 4, "null_count": 2}}},
    )

    assert any(f["action"] == "[READ]" and "test.csv" in f["path"] for f in manifest["target_files"])
    assert any(f["action"] == "[CREATE]" and "clean.csv" in f["path"] for f in manifest["target_files"])
    assert any(act["tool"] == "CLEAN" and act["column"] == "country" for act in manifest["planned_actions"])
    assert any(act["tool"] == "IMPUTE" and act["column"] == "age" for act in manifest["planned_actions"])


def test_cli_run_dry_run_command(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    out_file = tmp_path / "clean.csv"

    df = pl.DataFrame(
        {
            "id": list(range(30)),
            "age": [25] * 28 + [-10, 300],
            "target": [0] * 28 + [1, 1],
        }
    )
    df.write_csv(str(raw_file))

    result = runner.invoke(app, ["run", "-f", str(raw_file), "-o", str(out_file), "--dry-run"], input="y\n")
    assert result.exit_code == 0
    assert "PLANNED EXECUTION MANIFEST" in result.output
    assert "DRY-RUN COMPLETE" in result.output
    assert not out_file.exists()


def test_cli_run_user_rejects_manifest_manual_selection(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    out_file = tmp_path / "clean.csv"

    df = pl.DataFrame(
        {
            "id": list(range(30)),
            "age": [25] * 28 + [-10, 300],
            "target": [0] * 28 + [1, 1],
        }
    )
    df.write_csv(str(raw_file))

    user_inputs = "n\n2\n3\n"
    result = runner.invoke(
        app,
        ["run", "-f", str(raw_file), "-o", str(out_file), "-t", "test_reject_sess"],
        input=user_inputs,
    )
    assert result.exit_code == 0
    assert "PLANNED EXECUTION MANIFEST" in result.output
    assert "Operator opted for manual column-by-column governance" in result.output
    assert "PIPELINE EXECUTION COMPLETED" in result.output
    assert out_file.exists()


def test_drop_strategy_removes_column_not_rows():
    from refine.schema_inference import ColumnSchema, DatasetSchema
    from refine.tools.transformer import apply_human_resolutions

    df = pl.DataFrame(
        {
            "id": list(range(10)),
            "age": [25] * 7 + [None, -10, 200],
            "salary": [50000] * 10,
        }
    )
    schema = DatasetSchema(
        columns={
            "id": ColumnSchema(role="id", semantic_type="id"),
            "age": ColumnSchema(role="feature", semantic_type="numerical", valid_bounds=[0, 100]),
            "salary": ColumnSchema(role="feature", semantic_type="numerical"),
        }
    )

    clean_df, logs = apply_human_resolutions(df, {"age": "DROP"}, schema)

    assert "age" not in clean_df.columns
    assert "id" in clean_df.columns
    assert "salary" in clean_df.columns
    assert clean_df.height == 10
    assert any("removed feature column" in log for log in logs)


def test_render_streaming_panel():
    from refine.profiler.reporters import render_streaming_panel, stream_line

    render_streaming_panel("Test Panel", "Header", "Body word1 word2", stream=False)
    render_streaming_panel("Test Panel 2", "Header 2", "Body word1 word2", stream=True)
    stream_line("  [bold green]✓[/bold green] [dim][TOOL: TEST][/dim] Sample status line", stream=False)
    stream_line("  [bold green]✓[/bold green] [dim][TOOL: TEST][/dim] Sample status line", stream=True)
    stream_line("", stream=True)


def test_cli_no_stream_option(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    out_file = tmp_path / "clean.csv"

    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df.write_csv(str(raw_file))

    result = runner.invoke(
        app,
        ["run", "-f", str(raw_file), "-o", str(out_file), "--no-stream", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "DRY-RUN COMPLETE" in result.output


def test_cli_default_interactive_refine(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df.write_csv(str(raw_file))

    result = runner.invoke(app, [], input=f"{raw_file}\ny\n")
    assert "refine-ai" in result.output
    assert "Please enter the path to the CSV dataset" in result.output


def test_cli_manifest_approval_source_attribution(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from refine.cli import app

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    out_file = tmp_path / "clean.csv"

    df = pl.DataFrame(
        {
            "id": list(range(20)),
            "score": [50] * 18 + [-999, 999],
        }
    )
    df.write_csv(str(raw_file))

    result = runner.invoke(
        app,
        ["run", "-f", str(raw_file), "-o", str(out_file), "-t", "test_source_sess"],
        input="y\n",
    )
    assert result.exit_code == 0
    assert "Rule-engine recommended execution manifest approved" in result.output
    assert "AI-recommended" not in result.output


def test_prompt_model_selection():
    import io
    import sys

    from refine.cli import prompt_model_selection

    orig_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("2\n")
        selected = prompt_model_selection(["model_a", "model_b"])
        assert selected == "model_b"

        sys.stdin = io.StringIO("3\n")
        selected_skip = prompt_model_selection(["model_a", "model_b"])
        assert selected_skip is None
    finally:
        sys.stdin = orig_stdin


def test_get_installed_ollama_models_offline(monkeypatch):
    from refine.tools.advisor import get_installed_ollama_models

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    models = get_installed_ollama_models()
    assert models == []

    monkeypatch.setenv("OLLAMA_DISABLED", "1")
    assert get_installed_ollama_models() == []


def test_mixed_type_csv_parsing(tmp_path):
    from refine.cli import _validate_input_file

    mixed_file = tmp_path / "mixed.csv"
    lines = ["id,amount\n"]
    for i in range(150):
        lines.append(f"{i},100\n")
    lines.append("151,1024.86\n")
    mixed_file.write_text("".join(lines))

    _validate_input_file(mixed_file)
    df = pl.read_csv(mixed_file, infer_schema_length=None, ignore_errors=True)
    assert df.height == 151
    assert df["amount"].dtype == pl.Float64


def test_nominal_numeric_columns_ignored():
    from refine.schema_inference import _infer_schema_heuristic

    df = pl.DataFrame(
        {
            "permit_id": [f"P_{i}" for i in range(200)],
            "Street Number": list(range(100, 300)),
            "Zipcode": [10001 + (i % 50) for i in range(200)],
            "phone": list(range(5550000, 5550200)),
            "amount": [float(100 + i) for i in range(200)],
            "age": [25 + (i % 60) for i in range(200)],
        }
    )

    schema = _infer_schema_heuristic(df)

    assert "Street Number" in schema.ignore_columns
    assert schema.columns["Street Number"].role == "ignore"

    assert "Zipcode" in schema.ignore_columns
    assert schema.columns["Zipcode"].role == "ignore"

    assert "phone" in schema.ignore_columns
    assert schema.columns["phone"].role == "ignore"

    assert schema.columns["amount"].role == "feature"
    assert schema.columns["amount"].semantic_type == "numerical"

    assert schema.columns["age"].role == "feature"
    assert schema.columns["age"].semantic_type == "numerical"


def test_excessive_missingness_recommends_drop():
    from refine.tools.advisor import get_recommended_strategies

    issues = [
        {"column": "sparse_col", "type": "HIGH_NULL_RATIO", "ratio": 0.95},
        {"column": "moderate_col", "type": "HIGH_NULL_RATIO", "ratio": 0.25},
    ]

    strat = get_recommended_strategies(issues)
    assert strat["sparse_col"] == "DROP"
    assert strat["moderate_col"] == "STATISTICAL_IMPUTE"


def test_manifest_mode_impute_for_strings():
    from refine.profiler.reporters import build_execution_manifest

    manifest = build_execution_manifest(
        raw_path="data/raw/test.csv",
        processed_path="data/processed/clean.csv",
        session_id="test_sess",
        total_rows=100,
        audit_trail=[],
        critical_issues=[
            {"column": "cat_col", "type": "HIGH_NULL_RATIO", "ratio": 0.25},
            {"column": "num_col", "type": "HIGH_NULL_RATIO", "ratio": 0.25},
        ],
        recommended_strategies={"cat_col": "STATISTICAL_IMPUTE", "num_col": "STATISTICAL_IMPUTE"},
        profile={
            "columns": {
                "cat_col": {"type": "String", "outliers_count": 0, "null_count": 25},
                "num_col": {"type": "Float64", "outliers_count": 0, "null_count": 25},
            }
        },
    )

    actions = {a["column"]: a["description"] for a in manifest["planned_actions"]}
    assert "Mode Impute" in actions["cat_col"]
    assert "Median Impute" in actions["num_col"]


def test_anomaly_percentages_displayed():
    from refine.profiler.stats import profile_dataset

    df = pl.DataFrame(
        {
            "val": [10.0] * 99 + [10000.0] * 1,
        }
    )
    schema_dict = {
        "columns": {
            "val": {
                "role": "feature",
                "semantic_type": "numerical",
                "valid_bounds": [0.0, 50.0],
            }
        }
    }
    profile = profile_dataset(df, schema=schema_dict)
    anomalies_str = " ".join(profile["columns"]["val"]["anomalies"])
    assert "Out-of-bounds (1.0%)" in anomalies_str
    assert "Outliers (1.0%)" in anomalies_str


def test_prompt_output_directory(tmp_path):
    import io
    import sys
    from pathlib import Path

    from refine.cli import prompt_output_directory

    raw_file = tmp_path / "dataset.csv"
    orig_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("1\n")
        dest_same = prompt_output_directory(raw_file)
        assert dest_same == tmp_path

        sys.stdin = io.StringIO(f"2\n{tmp_path / 'custom_out'}\n")
        dest_custom = prompt_output_directory(raw_file)
        assert dest_custom == tmp_path / "custom_out"

        sys.stdin = io.StringIO("2\n\n")
        dest_empty_custom = prompt_output_directory(raw_file)
        assert dest_empty_custom == Path("data/processed")

        sys.stdin = io.StringIO("3\n")
        dest_default = prompt_output_directory(raw_file)
        assert dest_default == Path("data/processed")
    finally:
        sys.stdin = orig_stdin


def test_cli_output_directory_interactive_same_dir(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from refine.cli import app

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    runner = CliRunner()
    raw_file = tmp_path / "input_data.csv"
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df.write_csv(str(raw_file))

    result = runner.invoke(app, [], input=f"{raw_file}\n1\ny\n")
    assert result.exit_code == 0
    assert (tmp_path / "clean_input_data.csv").exists()
    assert (tmp_path / "clean_input_data_audit_report.md").exists()


def test_cli_output_directory_interactive_custom_dir(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from refine.cli import app

    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    runner = CliRunner()
    raw_file = tmp_path / "input_data.csv"
    custom_dir = tmp_path / "my_custom_destination"
    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df.write_csv(str(raw_file))

    result = runner.invoke(app, [], input=f"{raw_file}\n2\n{custom_dir}\ny\n")
    assert result.exit_code == 0
    assert (custom_dir / "clean_input_data.csv").exists()
    assert (custom_dir / "clean_input_data_audit_report.md").exists()


def test_cli_run_directory_output_option(tmp_path):
    from typer.testing import CliRunner

    from refine.cli import app

    runner = CliRunner()
    raw_file = tmp_path / "raw.csv"
    out_dir = tmp_path / "processed_dir"
    out_dir.mkdir()

    df = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df.write_csv(str(raw_file))

    result = runner.invoke(
        app,
        ["run", "-f", str(raw_file), "-o", str(out_dir), "--no-stream", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "DRY-RUN COMPLETE" in result.output
