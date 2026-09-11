from pathlib import Path
from typing import Any, Dict
import polars as pl
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from refine.state import AgentState
from refine.schema_inference import infer_schema, DatasetSchema
from refine.profiler.stats import profile_dataset
from refine.tools.cleaner import run_deterministic_clean
from refine.tools.transformer import apply_human_resolutions
from refine.tools.advisor import generate_expert_advice
from refine.tools.reporter import generate_markdown_audit


def schema_inference_node(state: AgentState) -> Dict[str, Any]:
    """Uses AI (LLM) or heuristic fallback to infer the dataset schema from raw records."""
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

    return {
        "records": df.to_dicts(),
        "initial_row_count": df.height,
        "inferred_schema": schema_dict,
        "schema_method": method,
        "audit_trail": state.get("audit_trail", []) + [audit_entry],
    }


def profile_node(state: AgentState) -> Dict[str, Any]:
    """Computes statistical profile using the inferred schema to detect anomalies and class imbalance."""
    df = pl.DataFrame(state["records"])
    schema_dict = state.get("inferred_schema")
    profile = profile_dataset(df, schema_dict)
    
    return {
        "profile": profile,
        "critical_issues": profile["critical_issues"],
        "audit_trail": state.get("audit_trail", []) + ["Dataset profiling completed."],
    }


def deterministic_clean_node(state: AgentState) -> Dict[str, Any]:
    """Executes safe, invariant data sanitation without human intervention."""
    df = pl.DataFrame(state["records"])
    schema = _load_schema(state)
    df, logs = run_deterministic_clean(df, schema)
    
    return {
        "records": df.to_dicts(),
        "audit_trail": state["audit_trail"] + logs
    }


def evaluate_anomalies_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates operational invariants. Emits LLM advice and triggers HITL interrupt."""
    critical_issues = state["critical_issues"]
    
    if critical_issues and not state.get("human_resolutions"):
        # Generate expert LLM guidance using RULES.md
        advice = generate_expert_advice(state["profile"], critical_issues)
        
        human_decisions = interrupt({
            "instruction": "Critical anomalies require human governance before pipeline execution.",
            "issues": critical_issues,
            "expert_advice": advice,
            "available_strategies": ["DROP", "STATISTICAL_IMPUTE", "SYNTHETIC_SYNTHESIS", "MANUAL_INPUT"]
        })
        return {"human_resolutions": human_decisions, "expert_advice": advice}
        
    return {"human_resolutions": state.get("human_resolutions", {})}


def apply_resolutions_node(state: AgentState) -> Dict[str, Any]:
    """Applies human choices to fix anomalies."""
    df = pl.DataFrame(state["records"])
    schema = _load_schema(state)
    resolutions = state.get("human_resolutions", {})
    
    if resolutions:
        df, logs = apply_human_resolutions(df, resolutions, schema)
        return {
            "records": df.to_dicts(),
            "audit_trail": state["audit_trail"] + logs
        }
    return {}


def export_node(state: AgentState) -> Dict[str, Any]:
    """Exports cleaned records and generates an executive Markdown audit report."""
    df = pl.DataFrame(state["records"])
    output_path = state.get("processed_file_path") or "data/processed/clean_output.csv"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    
    report_path = generate_markdown_audit(
        raw_path=state["raw_file_path"],
        processed_path=output_path,
        initial_rows=state.get("initial_row_count", df.height),
        final_rows=df.height,
        resolutions=state.get("human_resolutions", {}),
        audit_trail=state["audit_trail"] + [f"Final clean dataset exported to '{output_path}'."],
        expert_advice=state.get("expert_advice")
    )
    
    return {
        "processed_file_path": output_path,
        "is_completed": True,
        "audit_trail": state["audit_trail"] + [
            f"Clean dataset exported to '{output_path}'.",
            f"Governance Audit Report compiled to '{report_path}'."
        ]
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
    
    builder.add_node("profile", profile_node)
    builder.add_node("schema_inference", schema_inference_node)
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