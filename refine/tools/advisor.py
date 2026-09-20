import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


def is_ollama_online(base_url: str | None = None) -> bool:
    """Fast check (timeout 0.3s) if local Ollama daemon is reachable and enabled."""
    if os.getenv("OLLAMA_DISABLED") == "1":
        return False
    url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.3):
            return True
    except Exception:
        return False


_is_ollama_online = is_ollama_online


def get_installed_ollama_models(base_url: str | None = None) -> list[str]:
    """Queries the local Ollama daemon for installed model tags."""
    if os.getenv("OLLAMA_DISABLED") == "1":
        return []
    url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", []) if "name" in m]
    except Exception:
        return []


def _format_col_issues(issues: list[dict[str, Any]]) -> str:
    parts = []
    seen = set()
    for issue in issues:
        itype = issue.get("type", "")
        if itype == "INVALID_BOUNDS" and "bounds" not in seen:
            seen.add("bounds")
            parts.append("Invalid Bounds")
        elif itype == "STATISTICAL_OUTLIER" and "outlier" not in seen:
            seen.add("outlier")
            parts.append("Statistical Outliers")
        elif itype == "CLASS_IMBALANCE" and "imbalance" not in seen:
            seen.add("imbalance")
            ratio = issue.get("minority_ratio", 0) * 100
            parts.append(f"Class Imbalance - {ratio:.1f}%")
        elif itype == "HIGH_NULL_RATIO" and "null" not in seen:
            seen.add("null")
            ratio = issue.get("ratio", 0) * 100
            parts.append(f"High Missing Ratio - {ratio:.1f}%")
        elif itype not in seen and itype:
            seen.add(itype)
            parts.append(itype)
    return " & ".join(parts)


def _build_column_recommendation(col: str, issues: list[dict[str, Any]], col_type: str = "") -> str:
    issue_desc = _format_col_issues(issues)
    issue_types = {i.get("type", "") for i in issues}
    null_ratio = max((i.get("ratio", 0.0) for i in issues if i.get("type") == "HIGH_NULL_RATIO"), default=0.0)

    if null_ratio >= 0.50:
        strategy = "DROP"
        impact = (
            f"High missingness ({null_ratio * 100:.1f}%) exceeds the 50% threshold for reliable imputation; "
            f"dropping the column preserves data integrity without losing records."
        )
    elif "CLASS_IMBALANCE" in issue_types:
        imb_issue = next(i for i in issues if i.get("type") == "CLASS_IMBALANCE")
        val = imb_issue.get("minority_value", 1)
        strategy = "SYNTHETIC_SYNTHESIS"
        impact = (
            f"Synthesize records for minority class ({val}) to reach 35% balance; "
            f"mitigates model bias and increases total row count."
        )
    elif "INVALID_BOUNDS" in issue_types and "STATISTICAL_OUTLIER" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        impact = (
            "Both out-of-bounds values and statistical outliers are resolved in a single step "
            "using median imputation; preserves data integrity and keeps row count constant."
        )
    elif "INVALID_BOUNDS" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        impact = (
            "Corrupted and out-of-bounds cells are imputed with the median; avoids row loss and maintains sample size."
        )
    elif "STATISTICAL_OUTLIER" in issue_types:
        strategy = "SYNTHETIC_SYNTHESIS"
        impact = (
            "Outliers are replaced with realistic Gaussian distribution values; "
            "preserves sample variance and bell-curve geometry."
        )
    elif "HIGH_NULL_RATIO" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        if col_type == "String":
            impact = "Missing categorical values are imputed with the most frequent value (mode) to prevent row loss."
        else:
            impact = "Missing cells are populated with the statistical median to prevent row loss."
    else:
        strategy = "STATISTICAL_IMPUTE"
        impact = "Anomalous cells are imputed to maintain data integrity."

    return f"• '{col}' ({issue_desc}):\n  ➜ Recommended Decision: {strategy}\n  ➜ Decision Impact: {impact}"


def generate_expert_advice(profile: dict[str, Any], critical_issues: list) -> str:
    """Reads RULES.md, evaluates anomalies via local Ollama, and falls back gracefully if offline."""
    rules_path = Path("RULES.md")
    rules_content = rules_path.read_text() if rules_path.exists() else "Apply standard statistical governance."

    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in critical_issues:
        col = issue.get("column", "unknown")
        grouped.setdefault(col, []).append(issue)

    col_stats = profile.get("columns", {}) if profile else {}
    recommendations = [
        _build_column_recommendation(col, issues, col_stats.get(col, {}).get("type", ""))
        for col, issues in grouped.items()
    ]

    heuristic_advice = (
        "[Local Rule-Engine Fallback]:\n\n" + "\n\n".join(recommendations)
        if recommendations
        else "[Local Rule-Engine Fallback]: No critical remediation required."
    )

    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    if not _is_ollama_online(ollama_base_url):
        return heuristic_advice

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model_name, base_url=ollama_base_url, temperature=0.1, timeout=15.0)

        system_prompt = (
            f"You are a Senior Principal Data Architect enforcing this operational governance ruleset:\n\n"
            f"{rules_content}\n\n"
            f"For each anomalous column listed below, you MUST recommend EXACTLY ONE decisive strategy.\n"
            f"Allowed strategies: DROP, STATISTICAL_IMPUTE, SYNTHETIC_SYNTHESIS, MANUAL_INPUT.\n\n"
            f"CORE RULES:\n"
            f"- If missingness is >= 50% (HIGH_NULL_RATIO with ratio >= 0.50), ALWAYS recommend DROP.\n"
            f"- For columns with moderate missingness (< 50%) or invalid bounds, recommend STATISTICAL_IMPUTE.\n"
            f"- For target columns with class imbalance, recommend SYNTHETIC_SYNTHESIS.\n"
            f"- For continuous numerical columns with outliers, recommend SYNTHETIC_SYNTHESIS.\n"
            f"- Do NOT recommend SYNTHETIC_SYNTHESIS for categorical or discrete columns.\n\n"
            f"CRITICAL FORMATTING:\n"
            f"Do NOT write any preamble, introduction, summary, numbered list, or markdown headers.\n"
            f"Output ONLY a bullet point for each column strictly in this format:\n"
            f"• '<column_name>':\n"
            f"  ➜ Recommended Decision: <EXACTLY_ONE_STRATEGY>\n"
            f"  ➜ Decision Impact: 1-2 sentences explaining the impact.\n"
        )

        anomalies_summary = []
        for col, issues in grouped.items():
            col_type = col_stats.get(col, {}).get("type", "unknown")
            issue_lines = [f"  - {i.get('type')}: {i.get('message', '')}" for i in issues]
            anomalies_summary.append(f"Column '{col}' (Type: {col_type}):\n" + "\n".join(issue_lines))

        user_content = "Anomalous Columns to evaluate:\n\n" + "\n\n".join(anomalies_summary)

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])

        return f"[Ollama: {model_name}]\n\n{response.content.strip()}"

    except Exception:
        return heuristic_advice


def get_recommended_strategies(critical_issues: list[dict[str, Any]], advice_text: str = "") -> dict[str, str]:
    """Extracts or deduces a single decisive recommended strategy per anomalous column."""
    recommendations: dict[str, str] = {}
    valid_strategies = {"DROP", "STATISTICAL_IMPUTE", "SYNTHETIC_SYNTHESIS", "MANUAL_INPUT"}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in critical_issues:
        col = issue.get("column")
        if col:
            grouped.setdefault(col, []).append(issue)

    for col, col_issues in grouped.items():
        if advice_text:
            pattern = rf"['\"]?{re.escape(col)}['\"]?.*?Recommended Decision:\s*([A-Z_]+)"
            match = re.search(pattern, advice_text, re.DOTALL | re.IGNORECASE)
            if match and match.group(1).upper() in valid_strategies:
                recommendations[col] = match.group(1).upper()
                continue

        issue_types = {i.get("type", "") for i in col_issues}
        null_ratio = max((i.get("ratio", 0.0) for i in col_issues if i.get("type") == "HIGH_NULL_RATIO"), default=0.0)

        if null_ratio >= 0.50:
            recommendations[col] = "DROP"
        elif "CLASS_IMBALANCE" in issue_types:
            recommendations[col] = "SYNTHETIC_SYNTHESIS"
        elif "INVALID_BOUNDS" in issue_types and "STATISTICAL_OUTLIER" in issue_types:
            recommendations[col] = "STATISTICAL_IMPUTE"
        elif "STATISTICAL_OUTLIER" in issue_types:
            recommendations[col] = "SYNTHETIC_SYNTHESIS"
        elif "INVALID_BOUNDS" in issue_types or "HIGH_NULL_RATIO" in issue_types:
            recommendations[col] = "STATISTICAL_IMPUTE"
        else:
            recommendations[col] = "STATISTICAL_IMPUTE"

    return recommendations
