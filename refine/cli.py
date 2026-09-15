import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from refine import __version__
from refine.graph import build_pipeline_graph
from refine.logger import setup_logger
from refine.profiler.reporters import (
    build_execution_manifest,
    render_completion_summary,
    render_execution_manifest,
    render_profile_table,
    render_streaming_panel,
    stream_line,
)
from refine.profiler.stats import profile_dataset
from refine.schema_inference import infer_schema

app = typer.Typer(
    name="refine",
    help="refine-ai: Autonomous Data Pipeline & Synthesis Agent with HITL Governance.",
    add_completion=False,
)
console = Console()
logger = setup_logger()


def render_interrupt_ui(interrupt_payload: dict, show_advice: bool = True, stream: bool = True) -> dict[str, str]:
    """Displays LLM advice and collects human decisions for anomalous features."""
    expert_advice = interrupt_payload.get("expert_advice")
    if show_advice and expert_advice:
        render_streaming_panel(
            title="Agent Guidance",
            header_text="[bold cyan]🧠 Senior Data Architect Reasoning (via RULES.md):[/bold cyan]",
            body_text=expert_advice,
            border_style="cyan",
            stream=stream,
        )

    issues = interrupt_payload.get("issues", [])
    strategies = interrupt_payload.get("available_strategies", [])

    table = Table(title="Anomalous Features Requiring Human Intervention")
    table.add_column("Target Feature", style="cyan", no_wrap=True)
    table.add_column("Issue Type", style="magenta")
    table.add_column("Details", style="white")

    if console.is_terminal and stream:
        with Live(table, console=console, refresh_per_second=25):
            for issue in issues:
                table.add_row(issue["column"], issue["type"], issue["message"])
                time.sleep(0.12)
    else:
        for issue in issues:
            table.add_row(issue["column"], issue["type"], issue["message"])
        console.print(table)
    console.print(f"\n[bold]Available Remediation Strategies:[/bold] [green]{', '.join(strategies)}[/green]")

    decisions: dict[str, str] = {}
    unique_columns = list(dict.fromkeys(issue["column"] for issue in issues))
    total_cols = len(unique_columns)

    for idx, col in enumerate(unique_columns, 1):
        console.print()
        console.print(f"  [bold cyan]Feature {idx}/{total_cols}:[/bold cyan] [bold white]{col}[/bold white]")
        console.print(
            "  [dim]Options: [1] DROP  [2] STATISTICAL_IMPUTE  [3] SYNTHETIC_SYNTHESIS  [4] MANUAL_INPUT[/dim]"
        )

        choice = Prompt.ask(
            "  [bold cyan]➜[/bold cyan] [bold]Select strategy[/bold]",
            choices=["1", "2", "3", "4", "DROP", "STATISTICAL_IMPUTE", "SYNTHETIC_SYNTHESIS", "MANUAL_INPUT"],
            show_choices=False,
        )

        mapping = {
            "1": "DROP",
            "2": "STATISTICAL_IMPUTE",
            "3": "SYNTHETIC_SYNTHESIS",
            "4": "MANUAL_INPUT",
            "DROP": "DROP",
            "STATISTICAL_IMPUTE": "STATISTICAL_IMPUTE",
            "SYNTHETIC_SYNTHESIS": "SYNTHETIC_SYNTHESIS",
            "MANUAL_INPUT": "MANUAL_INPUT",
        }
        selected_strategy = mapping[choice]

        if selected_strategy in strategies:
            if selected_strategy == "MANUAL_INPUT":
                override_val = Prompt.ask(f"  [bold cyan]?[/bold cyan] Enter override value for '{col}'")
                decisions[col] = f"MANUAL_INPUT:{override_val}"
                console.print(
                    f"  [bold green]✓[/bold green] Assigned [bold green]MANUAL_INPUT[/bold green] (value: '{override_val}') to '{col}'"
                )
            else:
                decisions[col] = selected_strategy
                console.print(
                    f"  [bold green]✓[/bold green] Assigned [bold green]{selected_strategy}[/bold green] to '{col}'"
                )
        else:
            console.print(f"  [bold red]✗ Invalid strategy:[/bold red] {choice}")
            return {}

    console.print()
    return decisions


def _validate_input_file(file_path: Path) -> None:
    """Validates that input raw data file exists, is readable, and non-empty."""
    if not file_path.exists():
        console.print(f"\n[bold red]Error:[/bold red] Target raw data file '{file_path}' not found.\n")
        logger.error(f"File not found: {file_path}")
        raise typer.Exit(code=1)

    if file_path.stat().st_size == 0:
        console.print(f"\n[bold red]Error:[/bold red] Target file '{file_path}' is empty (0 bytes).\n")
        logger.error(f"Empty file: {file_path}")
        raise typer.Exit(code=1)

    try:
        sample = pl.read_csv(file_path, n_rows=5)
        if sample.width == 0:
            console.print(f"\n[bold red]Error:[/bold red] Target file '{file_path}' contains no valid columns.\n")
            raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"\n[bold red]Error parsing CSV file '{file_path}':[/bold red] {e}\n")
        logger.error(f"CSV parse failure on {file_path}: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def run(
    file: str = typer.Option(..., "--file", "-f", help="Input raw CSV path"),
    output: str | None = typer.Option(None, "--output", "-o", help="Processed target CSV path"),
    thread_id: str = typer.Option("session_001", "--thread-id", "-t", help="Checkpoint session thread ID"),
    model: str = typer.Option(
        "qwen2.5-coder:14b", "--model", "-m", help="Ollama LLM model name for reasoning & inference"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for deterministic reproducibility"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview planned execution and manifest without modifying files on disk"
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable or disable ChatGPT-style streaming output"),
):
    """Executes the pipeline graph, gracefully pausing on anomalies for human governance."""
    raw_path = Path(file)
    _validate_input_file(raw_path)

    if output is None:
        clean_name = f"clean_{raw_path.name}" if not raw_path.name.startswith("clean_") else raw_path.name
        output = str(Path("data/processed") / clean_name)

    if model:
        os.environ["OLLAMA_MODEL"] = model

    db_path = Path(".checkpoints.db")
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except Exception as e:
        console.print(f"\n[bold red]Database Error:[/bold red] Could not initialize checkpoint database: {e}\n")
        logger.error(f"SqliteSaver initialization error: {e}")
        raise typer.Exit(code=1) from None

    graph = build_pipeline_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    console.print(
        f"\n[bold green]► Initializing refine-ai pipeline:[/bold green] [cyan]{file}[/cyan] "
        f"(Session: [yellow]{thread_id}[/yellow], Seed: [yellow]{seed}[/yellow])\n"
    )

    initial_state = {
        "raw_file_path": str(raw_path),
        "processed_file_path": output,
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
        "session_id": thread_id,
        "random_seed": seed,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "execution_duration_sec": None,
    }

    try:
        console.print("[bold cyan]Executing pipeline graph...[/bold cyan]\n")
        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name in chunk:
                if node_name == "schema_inference":
                    stream_line(
                        "  [bold green]✓[/bold green] [dim][TOOL: SCHEMA_INFERENCE][/dim] Inferred column roles and semantic types.",
                        stream=stream,
                    )
                    if console.is_terminal and stream:
                        time.sleep(0.12)
                elif node_name == "profile":
                    stream_line(
                        "  [bold green]✓[/bold green] [dim][TOOL: PROFILER][/dim] Statistical profile completed & anomalies detected.",
                        stream=stream,
                    )
                    if console.is_terminal and stream:
                        time.sleep(0.12)
                elif node_name == "deterministic_clean":
                    stream_line(
                        "  [bold green]✓[/bold green] [dim][TOOL: CLEANER][/dim] Whitespace & canonical aliases standardized.",
                        stream=stream,
                    )
                    if console.is_terminal and stream:
                        time.sleep(0.12)
                elif node_name == "evaluate_anomalies":
                    stream_line(
                        "  [bold green]✓[/bold green] [dim][TOOL: ADVISOR][/dim] Evaluated data hygiene & operational rules.",
                        stream=stream,
                    )
                    if console.is_terminal and stream:
                        time.sleep(0.12)

        state_snapshot = graph.get_state(config)
        profile_data = state_snapshot.values.get("profile")

        if profile_data:
            console.print()
            render_profile_table(profile_data, stream=stream)
            console.print()

        if state_snapshot.tasks and any(task.interrupts for task in state_snapshot.tasks):
            interrupt_info = state_snapshot.tasks[0].interrupts[0].value
            expert_advice = interrupt_info.get("expert_advice")
            is_ai = bool(
                interrupt_info.get("source") == "llm" or (expert_advice and expert_advice.startswith("[Ollama:"))
            )

            if expert_advice:
                header_text = (
                    "[bold cyan]🧠 Senior Data Architect Reasoning (via RULES.md & Ollama LLM):[/bold cyan]"
                    if is_ai
                    else "[bold cyan]📋 Deterministic Rule-Engine Reasoning (via RULES.md Fallback):[/bold cyan]"
                )
                render_streaming_panel(
                    title="Agent Guidance",
                    header_text=header_text,
                    body_text=expert_advice,
                    border_style="cyan",
                    stream=stream,
                )
                console.print()

            manifest = build_execution_manifest(
                raw_path=str(raw_path),
                processed_path=output,
                session_id=thread_id,
                total_rows=state_snapshot.values.get("initial_row_count", 0),
                audit_trail=state_snapshot.values.get("audit_trail", []),
                critical_issues=interrupt_info.get("issues", []),
                recommended_strategies=interrupt_info.get("recommended_strategies", {}),
                profile=profile_data,
            )
            render_execution_manifest(manifest, stream=stream)
            console.print()

            approve = Confirm.ask(
                "  [bold cyan]?[/bold cyan] [bold]Do you approve executing these file operations?[/bold]"
            )

            if approve:
                approval_msg = (
                    "✓ AI-recommended execution manifest approved."
                    if is_ai
                    else "✓ Rule-engine recommended execution manifest approved."
                )
                console.print(f"\n[bold green]{approval_msg}[/bold green]")
                human_decisions = interrupt_info.get("recommended_strategies", {})
            else:
                console.print(
                    "\n[bold yellow]ℹ  Operator opted for manual column-by-column governance.[/bold yellow]\n"
                )
                human_decisions = render_interrupt_ui(interrupt_info, show_advice=False, stream=stream)

            if dry_run:
                console.print(
                    Panel.fit(
                        "[bold yellow]► DRY-RUN COMPLETE:[/bold yellow] All planned operations inspected. No files were modified or written to disk.",
                        border_style="yellow",
                    )
                )
                raise typer.Exit(code=0) from None

            console.print("\n[bold green]► Resuming execution graph with decisions...[/bold green]\n")
            for chunk in graph.stream(Command(resume=human_decisions), config=config, stream_mode="updates"):
                for node_name in chunk:
                    if node_name == "apply_resolutions":
                        stream_line(
                            "  [bold green]✓[/bold green] [dim][TOOL: TRANSFORMER][/dim] Applied human resolutions & verified cleanliness.",
                            stream=stream,
                        )
                        if console.is_terminal and stream:
                            time.sleep(0.12)
                    elif node_name == "export":
                        stream_line(
                            "  [bold green]✓[/bold green] [dim][TOOL: EXPORTER][/dim] Clean dataset and audit documentation exported.",
                            stream=stream,
                        )
                        if console.is_terminal and stream:
                            time.sleep(0.12)
            console.print()
        else:
            if dry_run:
                manifest = build_execution_manifest(
                    raw_path=str(raw_path),
                    processed_path=output,
                    session_id=thread_id,
                    total_rows=state_snapshot.values.get("initial_row_count", 0),
                    audit_trail=state_snapshot.values.get("audit_trail", []),
                    critical_issues=[],
                    recommended_strategies={},
                    profile=profile_data,
                )
                render_execution_manifest(manifest, stream=stream)
                console.print(
                    Panel.fit(
                        "[bold yellow]► DRY-RUN COMPLETE:[/bold yellow] Cleanliness verified. No files were modified or written to disk.",
                        border_style="yellow",
                    )
                )
                raise typer.Exit(code=0) from None

        final_state = graph.get_state(config).values
        render_completion_summary(
            processed_path=str(final_state.get("processed_file_path") or ""),
            audit_trail=final_state.get("audit_trail", []),
            duration_sec=final_state.get("execution_duration_sec"),
            stream=stream,
        )

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Execution Failed:[/bold red] {e}\n")
        logger.exception(f"Unhandled error during pipeline run: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def profile(
    file: str = typer.Argument(..., help="Path to raw CSV file to profile"),
    model: str = typer.Option("qwen2.5-coder:14b", "--model", "-m", help="Ollama LLM model name"),
):
    """Profiles a dataset and inspects anomalies without modifying or writing data."""
    raw_path = Path(file)
    _validate_input_file(raw_path)

    if model:
        os.environ["OLLAMA_MODEL"] = model

    console.print(f"\n[bold blue]► Profiling dataset:[/bold blue] [cyan]{file}[/cyan]\n")

    try:
        df = pl.read_csv(raw_path)
        schema, method = infer_schema(df)
        prof = profile_dataset(df, schema.model_dump())
        render_profile_table(prof)
        console.print(f"[dim]Schema inferred using method: [bold cyan]{method}[/bold cyan][/dim]\n")
    except Exception as e:
        console.print(f"\n[bold red]Profiling Error:[/bold red] {e}\n")
        logger.exception(f"Profile command failed: {e}")
        raise typer.Exit(code=1) from None


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt and reset checkpoints immediately"),
):
    """Cleans all session checkpoints and resets execution state."""
    db_files = [Path(".checkpoints.db"), Path(".checkpoints.db-shm"), Path(".checkpoints.db-wal")]
    existing = [f for f in db_files if f.exists()]

    if not existing:
        console.print("[dim]No checkpoint databases found. Nothing to reset.[/dim]")
        return

    if not yes:
        confirmed = Confirm.ask(f"Are you sure you want to delete {len(existing)} checkpoint file(s)?")
        if not confirmed:
            console.print("[yellow]Reset cancelled.[/yellow]")
            return

    for f in existing:
        try:
            f.unlink()
            console.print(f" [green]✓ Deleted:[/green] {f.name}")
            logger.info(f"Deleted checkpoint file: {f}")
        except Exception as e:
            console.print(f" [red]✗ Could not delete {f.name}:[/red] {e}")

    console.print("[bold green]✓ Checkpoints reset successfully.[/bold green]")


@app.command()
def status(
    thread_id: str = typer.Argument(..., help="Session thread ID to inspect"),
):
    """Inspects the state and progress of a checkpointed session."""
    db_path = Path(".checkpoints.db")
    if not db_path.exists():
        console.print("[yellow]No checkpoint database found (.checkpoints.db). Run a pipeline first.[/yellow]")
        return

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = build_pipeline_graph().compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        state = graph.get_state(config)
        if not state.values:
            console.print(f"[yellow]No session found with Thread ID:[/yellow] [bold cyan]{thread_id}[/bold cyan]")
            return

        vals = state.values
        table = Table(title=f"Session Status: {thread_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Raw File", str(vals.get("raw_file_path", "N/A")))
        table.add_row("Processed File", str(vals.get("processed_file_path", "N/A")))
        table.add_row(
            "Is Completed", "[green]Yes[/green]" if vals.get("is_completed") else "[yellow]No (Paused/Active)[/yellow]"
        )
        table.add_row("Start Time", str(vals.get("start_time", "N/A")))
        table.add_row("End Time", str(vals.get("end_time", "N/A")))
        dur = vals.get("execution_duration_sec")
        table.add_row("Duration", f"{dur:.2f}s" if dur else "N/A")
        table.add_row("Seed", str(vals.get("random_seed", "N/A")))
        table.add_row(
            "Interrupted",
            "[bold red]Yes (Waiting for Human Input)[/bold red]"
            if (state.tasks and any(t.interrupts for t in state.tasks))
            else "[green]No[/green]",
        )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Status query failed:[/bold red] {e}")
        logger.exception(f"Status query failure for {thread_id}: {e}")


@app.command()
def version():
    """Prints the CLI version and environment details."""
    console.print(f"[bold green]refine-ai v{__version__}[/bold green]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
):
    """refine-ai: Autonomous Data Pipeline & Synthesis Agent with HITL Governance."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print(
            Panel.fit(
                f"[bold cyan]🚀 refine-ai[/bold cyan] [dim]v{__version__}[/dim]\n"
                "[dim]Autonomous Data Pipeline & Synthesis Agent with HITL Governance[/dim]",
                border_style="cyan",
                padding=(1, 4),
            )
        )
        console.print()
        selected_file = Prompt.ask("  [bold cyan]➜[/bold cyan] [bold]Please enter the path to the CSV dataset[/bold]")

        if not selected_file or not selected_file.strip():
            console.print("\n  [bold yellow]ℹ  No file path specified. Exiting.[/bold yellow]\n")
            raise typer.Exit(code=0)

        console.print()

        raw_path = Path(selected_file.strip())
        clean_name = f"clean_{raw_path.name}" if not raw_path.name.startswith("clean_") else raw_path.name
        ctx.invoke(
            run,
            file=selected_file.strip(),
            output=str(Path("data/processed") / clean_name),
            thread_id="session_001",
            model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
            seed=42,
            dry_run=False,
            stream=True,
        )


if __name__ == "__main__":
    app()
