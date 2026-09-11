from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from refine.logger import get_logger
from refine.profiler.stats import profile_dataset
from refine.schema_inference import DatasetSchema, infer_schema
from refine.state import AgentState
from refine.tools.advisor import generate_expert_advice, get_recommended_strategies
from refine.tools.cleaner import run_deterministic_clean
from refine.tools.reporter import generate_markdown_audit
from refine.tools.synthesizer import set_seed
from refine.tools.transformer import apply_human_resolutions

logger = get_logger()


def schema_inference_node(state: AgentState) -> dict[str, Any]:
    """Uses AI (LLM) or heuristic fallback to infer dataset schema from raw records."""
    start_time = state.get("start_time") or datetime.now().isoformat()
    seed = state.get("random_seed") or 42
    set_seed(seed)

    logger.info(f"Starting schema inference (session={state.get('session_id')}, seed={seed})")

    df = pl.DataFrame(state["records"]) if state.get("records") else pl.read_csv(state["raw_file_path"])
    schema, method = infer_schema(df)
    schema_dict = schema.model_dump()

    audit_entry = (
        f"Schema inferred via {method}: "
        f"target='{schema.target_column}', "
        f"id_cols={schema.id_columns}, "
        f"ignore_cols={schema.ignore_columns}, "
        f"{len(schema.columns)} columns classified."
    )
    logger.info(audit_entry)

    return {
        "records": df.to_dicts(),
        "initial_row_count": df.height,
        "inferred_schema": schema_dict,
        "schema_method": method,
        "start_time": start_time,
        "random_seed": seed,
        "audit_trail": state.get("audit_trail", []) + [audit_entry],
    }


def profile_node(state: AgentState) -> dict[str, Any]:
    """Computes statistical profile using the inferred schema to detect anomalies and class imbalance."""
    df = pl.DataFrame(state["records"])
    schema_dict = state.get("inferred_schema")
    profile = profile_dataset(df, schema_dict)

    crit_count = len(profile["critical_issues"])
    logger.info(f"Dataset profiling completed: {crit_count} critical issues detected.")

    return {
        "profile": profile,
        "critical_issues": profile["critical_issues"],
        "audit_trail": state.get("audit_trail", [])
        + [f"Dataset profiling completed: {crit_count} critical issues detected."],
    }


def deterministic_clean_node(state: AgentState) -> dict[str, Any]:
    """Executes safe, invariant data sanitation without human intervention."""
    df = pl.DataFrame(state["records"])
    schema = _load_schema(state)
    df, logs = run_deterministic_clean(df, schema)

    for log in logs:
        logger.info(f"Deterministic clean: {log}")

    return {"records": df.to_dicts(), "audit_trail": state["audit_trail"] + logs}


def evaluate_anomalies_node(state: AgentState) -> dict[str, Any]:
    """Evaluates operational invariants. Emits LLM advice and triggers HITL interrupt."""
    critical_issues = state["critical_issues"]

    if critical_issues and not state.get("human_resolutions"):
        logger.info(f"Triggering HITL interrupt for {len(critical_issues)} critical issues.")
        advice = generate_expert_advice(state["profile"], critical_issues)
        recommended = get_recommended_strategies(critical_issues, advice)

        human_decisions = interrupt(
            {
                "instruction": "Critical anomalies require human governance before pipeline execution.",
                "issues": critical_issues,
                "expert_advice": advice,
                "recommended_strategies": recommended,
                "available_strategies": ["DROP", "STATISTICAL_IMPUTE", "SYNTHETIC_SYNTHESIS", "MANUAL_INPUT"],
            }
        )
        return {"human_resolutions": human_decisions, "expert_advice": advice}

    return {"human_resolutions": state.get("human_resolutions", {})}


def apply_resolutions_node(state: AgentState) -> dict[str, Any]:
    """Applies human choices to fix anomalies, then verifies data cleanliness."""
    df = pl.DataFrame(state["records"])
    schema = _load_schema(state)
    resolutions = state.get("human_resolutions", {})

    if resolutions:
        logger.info(f"Applying human resolutions: {resolutions}")
        df, logs = apply_human_resolutions(df, resolutions, schema)

        # Post-remediation verification
        post_profile = profile_dataset(df, state.get("inferred_schema"))
        remaining = post_profile["critical_issues"]
        if remaining:
            verify_msg = f"Post-remediation verification: {len(remaining)} residual issues detected."
            logger.warning(verify_msg)
        else:
            verify_msg = "Post-remediation verification passed: dataset is clean and validated."
            logger.info(verify_msg)

        return {"records": df.to_dicts(), "audit_trail": state["audit_trail"] + logs + [verify_msg]}
    return {}


def export_node(state: AgentState) -> dict[str, Any]:
    """Exports cleaned records and generates an executive Markdown audit report with full telemetry."""
    df = pl.DataFrame(state["records"])
    output_path = state.get("processed_file_path") or "data/processed/clean_output.csv"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)

    end_time = datetime.now().isoformat()
    start_dt = datetime.fromisoformat(state["start_time"]) if state.get("start_time") else datetime.now()
    duration = max(0.0, (datetime.now() - start_dt).total_seconds())

    metadata = {
        "session_id": state.get("session_id"),
        "random_seed": state.get("random_seed"),
        "start_time": state.get("start_time"),
        "end_time": end_time,
        "execution_duration_sec": duration,
    }

    report_path = generate_markdown_audit(
        raw_path=state["raw_file_path"],
        processed_path=output_path,
        initial_rows=state.get("initial_row_count", df.height),
        final_rows=df.height,
        resolutions=state.get("human_resolutions", {}),
        audit_trail=state["audit_trail"] + [f"Final clean dataset exported to '{output_path}'."],
        expert_advice=state.get("expert_advice"),
        metadata=metadata,
    )

    logger.info(f"Exported clean dataset to '{output_path}' (duration={duration:.2f}s). Report: {report_path}")

    return {
        "processed_file_path": output_path,
        "end_time": end_time,
        "execution_duration_sec": duration,
        "is_completed": True,
        "audit_trail": state["audit_trail"]
        + [f"Clean dataset exported to '{output_path}'.", f"Governance Audit Report compiled to '{report_path}'."],
    }


def _load_schema(state: AgentState) -> DatasetSchema:
    """Reconstructs the DatasetSchema from state dict."""
    schema_dict = state.get("inferred_schema")
    if schema_dict:
        return DatasetSchema.model_validate(schema_dict)
    return DatasetSchema()


def build_pipeline_graph():
    """Compiles the LangGraph StateGraph with HITL interruption capabilities."""
    builder = StateGraph(AgentState)

    builder.add_node("schema_inference", schema_inference_node)
    builder.add_node("profile", profile_node)
    builder.add_node("deterministic_clean", deterministic_clean_node)
    builder.add_node("evaluate_anomalies", evaluate_anomalies_node)
    builder.add_node("apply_resolutions", apply_resolutions_node)
    builder.add_node("export", export_node)

    builder.add_edge(START, "schema_inference")
    builder.add_edge("schema_inference", "profile")
    builder.add_edge("profile", "deterministic_clean")
    builder.add_edge("deterministic_clean", "evaluate_anomalies")
    builder.add_edge("evaluate_anomalies", "apply_resolutions")
    builder.add_edge("apply_resolutions", "export")
    builder.add_edge("export", END)

    return builder
