from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def render_profile_table(profile: dict[str, Any]) -> None:
    """Renders dataset profile and detected anomalies in a structured CLI panel."""
    console.print(Panel.fit("[bold blue]Dataset Profile & Anomaly Audit[/bold blue]", border_style="blue"))

    table = Table(title=f"Total Records: {profile['total_rows']}")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Data Type", style="magenta")
    table.add_column("Missing (%)", style="yellow")
    table.add_column("Outliers", style="red")
    table.add_column("Status / Anomalies", style="white")

    for col_name, stats in profile["columns"].items():
        null_str = f"{stats['null_count']} ({stats['null_ratio'] * 100:.1f}%)"
        outlier_str = str(stats.get("outliers_count", 0))

        anomalies = stats.get("anomalies", [])
        if anomalies:
            status_str = f"[bold red]⚠️ {', '.join(anomalies)}[/bold red]"
        else:
            status_str = "[bold green]✓ Healthy[/bold green]"

        table.add_row(col_name, stats["type"], null_str, outlier_str, status_str)

    console.print(table)
