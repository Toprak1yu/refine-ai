import sqlite3
from pathlib import Path
from typing import Dict
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from refine.graph import build_pipeline_graph
from refine.profiler.reporters import render_profile_table

app = typer.Typer(
    name="refine",
    help="refine-ai: Autonomous Data Pipeline & Synthesis Agent with HITL Governance.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def render_interrupt_ui(interrupt_payload: dict) -> Dict[str, str]:
    """Displays detected critical anomalies, LLM advice, and collects human decisions."""
    console.print(
        Panel.fit(
            "[bold yellow]⚠️  HUMAN-IN-THE-LOOP (HITL) GOVERNANCE REQUIRED[/bold yellow]\n"
            f"[dim]{interrupt_payload.get('instruction')}[/dim]",
            border_style="yellow",
        )
    )

    # Display Senior Data Engineer / LLM Advice Panel
    expert_advice = interrupt_payload.get("expert_advice")
    if expert_advice:
        console.print(
            Panel(
                f"[bold cyan]🧠 Senior Data Architect Reasoning (via RULES.md):[/bold cyan]\n\n{expert_advice}",
                border_style="cyan",
                title="Agent Guidance"
            )
        )

    issues = interrupt_payload.get("issues", [])
    strategies = interrupt_payload.get("available_strategies", [])

    table = Table(title="Anomalous Features Requiring Human Intervention")
    table.add_column("Target Feature", style="cyan", no_wrap=True)
    table.add_column("Issue Type", style="magenta")
    table.add_column("Details", style="white")

    for issue in issues:
        table.add_row(issue["column"], issue["type"], issue["message"])

    console.print(table)
    console.print(f"\n[bold]Available Remediation Strategies:[/bold] [green]{', '.join(strategies)}[/green]\n")

    decisions: Dict[str, str] = {}
    unique_columns = list({issue["column"] for issue in issues})

    for col in unique_columns:
        prompt_text = (
            f"Select strategy for feature '[bold cyan]{col}[/bold cyan]' "
            f"([1] DROP, [2] STATISTICAL_IMPUTE, [3] SYNTHETIC_SYNTHESIS)"
        )
        choice = Prompt.ask(prompt_text, choices=["1", "2", "3", "DROP", "STATISTICAL_IMPUTE", "SYNTHETIC_SYNTHESIS"], default="2")

        mapping = {
            "1": "DROP",
            "2": "STATISTICAL_IMPUTE",
            "3": "SYNTHETIC_SYNTHESIS",
            "DROP": "DROP",
            "STATISTICAL_IMPUTE": "STATISTICAL_IMPUTE",
            "SYNTHETIC_SYNTHESIS": "SYNTHETIC_SYNTHESIS",
        }
        selected_strategy = mapping[choice]
        
        if selected_strategy in strategies:
            decisions[col] = selected_strategy
            console.print(f" -> Assigned [bold green]{selected_strategy}[/bold green] to '[bold cyan]{col}[/bold cyan]'")
        else:
            console.print(f" -> [bold red]Invalid strategy:[/bold red] {choice}")
            console.print(f" -> [bold red]Available strategies:[/bold red] {', '.join(strategies)}")
            console.print(f" -> [bold red]Please select a valid strategy.[/bold red]")
            return {}

    return decisions


@app.command()
def run(
    file: str = typer.Option("data/raw/dirty_customers.csv", "--file", "-f", help="Input raw CSV path"),
    output: str = typer.Option("data/processed/clean_customers.csv", "--output", "-o", help="Processed target CSV path"),
    thread_id: str = typer.Option("session_001", "--thread-id", "-t", help="Checkpoint session thread ID"),
):
    """Executes the pipeline graph, gracefully pausing on anomalies for human governance."""
    raw_path = Path(file)
    if not raw_path.exists():
        console.print(f"[bold red]Error:[/bold red] Target raw data file '{file}' not found.")
        raise typer.Exit(code=1)

    db_path = Path(".checkpoints.db")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Compile graph with persistent memory
    graph = build_pipeline_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[bold green]► Initializing pipeline for:[/bold green] [cyan]{file}[/cyan] (Session: {thread_id})\n")

    # Initial state
    initial_state = {
        "raw_file_path": str(raw_path),
        "processed_file_path": output,
        "records": [],
        "profile": None,
        "critical_issues": [],
        "human_resolutions": {},
        "audit_trail": [],
        "is_completed": False,
    }

    # Step 1: Run graph until completion or interrupt
    result = graph.invoke(initial_state, config=config)

    # Check if graph paused due to interrupt()
    state_snapshot = graph.get_state(config)

    if state_snapshot.tasks and any(task.interrupts for task in state_snapshot.tasks):
        # Extract the interrupt payload
        interrupt_info = state_snapshot.tasks[0].interrupts[0].value
        human_decisions = render_interrupt_ui(interrupt_info)

        console.print("\n[bold green]► Resuming execution graph with human decisions...[/bold green]\n")
        # Step 2: Resume graph with Command(resume=...)
        result = graph.invoke(Command(resume=human_decisions), config=config)

    # Final Summary & Audit Report
    final_state = graph.get_state(config).values
    console.print(Panel.fit("[bold green]✓ PIPELINE EXECUTION COMPLETED[/bold green]", border_style="green"))

    console.print("\n[bold cyan]Audit Log Trail:[/bold cyan]")
    for entry in final_state.get("audit_trail", []):
        console.print(f" [dim]•[/dim] {entry}")

    console.print(f"\n[bold]Output Dataset:[/bold] [green]{final_state.get('processed_file_path')}[/green]\n")

@app.command()
def version():
    """Prints the CLI version."""
    console.print("[bold green]refine-ai v0.1.0[/bold green]")

if __name__ == "__main__":
    app()