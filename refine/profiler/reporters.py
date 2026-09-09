from typing import Any, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def render_profile_table(profile: Dict[str, Any]) -> None:
    """Renders dataset profile and detected anomalies in a structured CLI panel."""
    console.print(Panel.fit("[bold blue]Dataset Profile & Anomaly Audit[/bold blue]", border_style="blue"))
    
    table = Table(title=f"Total Records: {profile['total_rows']}")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Data Type", style="magenta")
    table.add_column("Missing (%)", style="yellow")
    table.add_column("Outliers", style="red")
    
    for col_name, stats in profile["columns"].items():
        null_str = f"{stats['null_count']} ({stats['null_ratio']*100:.1f}%)"
        outlier_str = str(stats.get("outliers_count", 0))
        table.add_row(col_name, stats["type"], null_str, outlier_str)
        
    console.print(table)
    
    if profile["critical_issues"]:
        console.print("\n[bold red]⚠️  Critical Anomalies Detected (HITL Triggers):[/bold red]")
        for issue in profile["critical_issues"]:
            console.print(f" [red]•[/red] {issue['message']}")
    else:
        console.print("\n[bold green]✓ No critical anomalies found. Proceeding with standard flow.[/bold green]")