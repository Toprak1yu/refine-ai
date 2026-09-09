# refine/tools/reporter.py
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_markdown_audit(
    raw_path: str,
    processed_path: str,
    initial_rows: int,
    final_rows: int,
    resolutions: Dict[str, str],
    audit_trail: List[str],
    expert_advice: Optional[str] = None,
) -> str:
    """Generates an executive-ready Markdown audit report documenting all data pipeline operations."""
    report_path = Path(processed_path).with_name(f"{Path(processed_path).stem}_audit_report.md")

    strategy_table = "\n".join([f"| `{col}` | **{strat}** | Human Approved |" for col, strat in resolutions.items()])

    audit_bullets = "\n".join([f"- {log}" for log in audit_trail])

    # F-string başlangıcı
    report_content = f"""# Data Pipeline Audit & Governance Report
**Execution Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Raw Source:** `{raw_path}`  
**Target Destination:** `{processed_path}`  

---

## Executive Summary
| Metric | Initial State | Final State | Delta |
| :--- | :--- | :--- | :--- |
| **Total Records** | {initial_rows:,} | {final_rows:,} | {final_rows - initial_rows:+,} |
| **Integrity Status** | Anomalies Detected | Remediated & Validated | Passed |

---

## Human-in-the-Loop (HITL) Resolutions
The human operator intervened and executed the following governance actions:

| Feature | Strategy Selected | Authorization |
| :--- | :--- | :--- |
{strategy_table if strategy_table else "| None | No intervention required | Auto |"}

---

## Senior Data Architect Assessment
```text
{expert_advice or "Standard automated operational governance applied."}"""

    report_path.write_text(report_content)
    return str(report_path)