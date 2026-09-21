import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def stream_line(
    markup_or_text: str,
    stream: bool = True,
    char_delay: float = 0.015,
) -> None:
    """Renders a single line with typewriter-style streaming."""
    if not console.is_terminal or not stream:
        console.print(markup_or_text)
        return

    try:
        t = Text.from_markup(markup_or_text)
    except Exception:
        console.print(markup_or_text)
        return

    if len(t) == 0:
        console.print()
        return

    with Live(console=console, refresh_per_second=60) as live:
        for i in range(1, len(t) + 1):
            live.update(t[:i])
            time.sleep(char_delay)


def render_profile_table(profile: dict[str, Any], stream: bool = True) -> None:
    """Renders dataset profile and detected anomalies in a structured CLI panel."""
    table = Table(
        title=f"[bold blue]Dataset Profile & Anomaly Audit[/bold blue]  [dim]•[/dim]  [bold]Total Records:[/bold] {profile['total_rows']}",
        border_style="blue",
        header_style="bold",
    )
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Data Type", style="magenta")
    table.add_column("Status / Anomalies", style="white")

    rows = []
    for col_name, stats in profile["columns"].items():
        anomalies = stats.get("anomalies", [])
        if anomalies:
            status_str = f"[bold red]⚠️ {', '.join(anomalies)}[/bold red]"
        else:
            status_str = "[bold green]✓ Healthy[/bold green]"

        rows.append((col_name, stats["type"], status_str))

    if console.is_terminal and stream:
        with Live(table, console=console, refresh_per_second=25):
            for row in rows:
                table.add_row(*row)
                time.sleep(0.12)
    else:
        for row in rows:
            table.add_row(*row)
        console.print(table)


def render_streaming_panel(
    title: str,
    header_text: str,
    body_text: str,
    border_style: str = "cyan",
    stream: bool = True,
    word_delay: float = 0.035,
) -> None:
    """Renders a panel with ChatGPT-style streaming text effect."""
    full_text = f"{header_text}\n\n{body_text}" if header_text else body_text
    if not console.is_terminal or not stream:
        console.print(Panel(full_text, border_style=border_style, title=title, padding=(1, 2)))
        return

    words = body_text.split(" ")
    current = ""
    with Live(console=console, refresh_per_second=30) as live:
        for i, w in enumerate(words):
            current += w if i == 0 else " " + w
            content = f"{header_text}\n\n{current}" if header_text else current
            live.update(Panel(content, border_style=border_style, title=title, padding=(1, 2)))
            time.sleep(word_delay)


def build_execution_manifest(
    raw_path: str,
    processed_path: str,
    session_id: str,
    total_rows: int,
    audit_trail: list[str],
    critical_issues: list[dict[str, Any]],
    recommended_strategies: dict[str, str],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Constructs the structured execution manifest describing planned file ops and mutations."""
    audit_path = str(Path(processed_path).with_name(f"{Path(processed_path).stem}_audit_report.md"))

    target_files = [
        {"action": "[READ]", "path": str(raw_path), "desc": f"{total_rows} rows"},
        {"action": "[CREATE]", "path": str(processed_path), "desc": ""},
        {"action": "[CREATE]", "path": audit_path, "desc": ""},
        {"action": "[WRITE]", "path": ".checkpoints.db", "desc": f"State: {session_id}"},
    ]

    planned_actions: list[dict[str, str]] = []

    for log in audit_trail:
        if "Normalized" in log and "aliases in" in log:
            col = "country"
            if "'" in log:
                col = log.split("'")[1]
            planned_actions.append(
                {
                    "tool": "CLEAN",
                    "column": col,
                    "description": "Normalize aliases to canonical values",
                }
            )
        elif "whitespace stripping" in log.lower():
            planned_actions.append(
                {
                    "tool": "CLEAN",
                    "column": "strings",
                    "description": "Strip whitespace padding across string features",
                }
            )

    col_stats = profile.get("columns", {}) if profile else {}
    for col, strat in recommended_strategies.items():
        stats = col_stats.get(col, {})
        outliers = stats.get("outliers_count", 0)
        nulls = stats.get("null_count", 0)

        is_imbalance = any(i.get("type") == "CLASS_IMBALANCE" and i.get("column") == col for i in critical_issues)

        if strat == "STATISTICAL_IMPUTE":
            parts = []
            if outliers:
                outlier_pct = f" ({(outliers / total_rows) * 100:.1f}%)" if total_rows > 0 else ""
                parts.append(f"{outliers} outliers{outlier_pct}")
            if nulls:
                null_pct = f" ({(nulls / total_rows) * 100:.1f}%)" if total_rows > 0 else ""
                parts.append(f"{nulls} NA{null_pct}")
            detail = f" ({', '.join(parts)})" if parts else ""
            col_type = stats.get("type", "")
            sem_type = stats.get("semantic_type", "")
            impute_method = "Mode" if (col_type == "String" or sem_type == "categorical") else "Median"
            planned_actions.append(
                {
                    "tool": "IMPUTE",
                    "column": col,
                    "description": f"{impute_method} Impute{detail}",
                }
            )
        elif strat == "SYNTHETIC_SYNTHESIS":
            if is_imbalance:
                ratio = stats.get("minority_ratio", 0.0)
                m_count = int(round(total_rows * ratio)) if ratio else 0
                needed = max(1, int(np.ceil((0.35 * total_rows - m_count) / 0.65))) if total_rows > 0 else 0
                planned_actions.append(
                    {
                        "tool": "SYNTHESIS",
                        "column": col,
                        "description": f"Generate +{needed} synthetic rows to balance class",
                    }
                )
            else:
                outlier_pct = f" ({(outliers / total_rows) * 100:.1f}%)" if total_rows > 0 else ""
                planned_actions.append(
                    {
                        "tool": "SYNTHESIS",
                        "column": col,
                        "description": f"Gaussian Synthesis ({outliers} outliers{outlier_pct})",
                    }
                )
        elif strat in ("DROP", "DROP_COLUMN"):
            if nulls:
                null_pct = f" ({(nulls / total_rows) * 100:.1f}%)" if total_rows > 0 else ""
                desc = f"Drop column ({nulls} NA{null_pct}, preserves all records)"
            else:
                desc = "Drop column (preserves all records)"
            planned_actions.append(
                {
                    "tool": "DROP",
                    "column": col,
                    "description": desc,
                }
            )
        else:
            override = strat.split(":", 1)[1] if ":" in strat else strat
            planned_actions.append(
                {
                    "tool": "MANUAL",
                    "column": col,
                    "description": f"Manual override: {override}",
                }
            )

    return {
        "target_files": target_files,
        "planned_actions": planned_actions,
    }


def render_execution_manifest(manifest: dict[str, Any], stream: bool = True) -> None:
    """Renders the planned file operations and data mutations panel with streaming effect."""
    lines: list[str] = []

    lines.append("[bold]Target Files:[/bold]")
    for item in manifest.get("target_files", []):
        desc = f" [dim]({item['desc']})[/dim]" if item.get("desc") else ""
        lines.append(f"  • [bold yellow]{item['action']:<12}[/bold yellow] [white]{item['path']}[/white]{desc}")

    lines.append("\n[bold]Planned Actions & Data Mutations:[/bold]")
    for act in manifest.get("planned_actions", []):
        tool_label = f"[TOOL: {act['tool']}]"
        lines.append(
            f"  • [bold green]{tool_label:<18}[/bold green] [cyan]{act['column']:<10}[/cyan] -> {act['description']}"
        )

    if console.is_terminal and stream:
        accumulated: list[str] = []
        with Live(console=console, refresh_per_second=25) as live:
            for line in lines:
                accumulated.append(line)
                live.update(
                    Panel(
                        "\n".join(accumulated),
                        title="[bold]📋 PLANNED EXECUTION MANIFEST[/bold]",
                        border_style="blue",
                        padding=(1, 2),
                        expand=False,
                    )
                )
                time.sleep(0.08)
    else:
        panel_content = "\n".join(lines)
        console.print(
            Panel(
                panel_content,
                title="[bold]📋 PLANNED EXECUTION MANIFEST[/bold]",
                border_style="blue",
                padding=(1, 2),
                expand=False,
            )
        )


def render_completion_summary(
    processed_path: str,
    audit_trail: list[str],
    duration_sec: float | None = None,
    stream: bool = True,
) -> None:
    """Renders a structured completion panel displaying generated artifacts and audit logs."""
    report_path = (
        str(Path(processed_path).with_name(f"{Path(processed_path).stem}_audit_report.md")) if processed_path else ""
    )

    lines: list[str] = ["[bold]Output Artifacts:[/bold]"]
    if processed_path:
        lines.append(f"  • [bold cyan]Clean Dataset:[/bold cyan]  [white]{processed_path}[/white]")
    if report_path:
        lines.append(f"  • [bold cyan]Audit Report:[/bold cyan]   [white]{report_path}[/white]")
    if duration_sec is not None:
        lines.append(f"  • [bold cyan]Duration:[/bold cyan]       [white]{duration_sec:.2f}s[/white]")

    if audit_trail:
        lines.append("")
        lines.append("[bold]Audit Log Trail:[/bold]")
        for entry in audit_trail:
            lines.append(f"  [green]✓[/green] [white]{entry}[/white]")

    panel = Panel(
        "\n".join(lines),
        title="[bold green]✓ PIPELINE EXECUTION COMPLETED[/bold green]",
        border_style="green",
        padding=(1, 3),
        expand=False,
    )

    console.print()
    if console.is_terminal and stream:
        accumulated: list[str] = []
        with Live(console=console, refresh_per_second=25) as live:
            for line in lines:
                accumulated.append(line)
                live.update(
                    Panel(
                        "\n".join(accumulated),
                        title="[bold green]✓ PIPELINE EXECUTION COMPLETED[/bold green]",
                        border_style="green",
                        padding=(1, 3),
                        expand=False,
                    )
                )
                time.sleep(0.04)
    else:
        console.print(panel)
    console.print()
