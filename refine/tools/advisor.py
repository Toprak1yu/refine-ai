import os
import re
import urllib.request
from pathlib import Path
from typing import Any


def _is_ollama_online(base_url: str) -> bool:
    """Fast check (timeout 0.3s) if local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.3):
            return True
    except Exception:
        return False


def _format_col_issues(issues: list[dict[str, Any]]) -> str:
    parts = []
    seen = set()
    for issue in issues:
        itype = issue.get("type", "")
        if itype == "INVALID_BOUNDS" and "bounds" not in seen:
            seen.add("bounds")
            parts.append("Geçersiz Sınır Değerleri")
        elif itype == "STATISTICAL_OUTLIER" and "outlier" not in seen:
            seen.add("outlier")
            parts.append("Aşırı Uç Değerler")
        elif itype == "CLASS_IMBALANCE" and "imbalance" not in seen:
            seen.add("imbalance")
            ratio = issue.get("minority_ratio", 0) * 100
            parts.append(f"Sınıf Dengesizliği - %{ratio:.1f}")
        elif itype == "HIGH_NULL_RATIO" and "null" not in seen:
            seen.add("null")
            ratio = issue.get("ratio", 0) * 100
            parts.append(f"Yüksek Boşluk Oranı - %{ratio:.1f}")
        elif itype not in seen and itype:
            seen.add(itype)
            parts.append(itype)
    return " & ".join(parts)


def _build_column_recommendation(col: str, issues: list[dict[str, Any]]) -> str:
    issue_desc = _format_col_issues(issues)
    issue_types = {i.get("type", "") for i in issues}

    if "CLASS_IMBALANCE" in issue_types:
        imb_issue = next(i for i in issues if i.get("type") == "CLASS_IMBALANCE")
        val = imb_issue.get("minority_value", 1)
        strategy = "SYNTHETIC_SYNTHESIS"
        impact = (
            f"Azınlık sınıfı ({val}) için sentetik satırlar türetilerek denge %35'e çıkarılır; "
            f"modelin yanlı (biased) öğrenmesi engellenir, toplam satır sayısı artar."
        )
    elif "INVALID_BOUNDS" in issue_types and "STATISTICAL_OUTLIER" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        impact = (
            "Hem mantıksal sınır dışı hatalar hem de aşırı uç sapmalar medyan ile doldurularak "
            "iki sorun tek adımda çözülür; veri bütünlüğü sağlanır, satır sayısı sabit kalır."
        )
    elif "INVALID_BOUNDS" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        impact = (
            "Bozuk ve mantıksal sınır dışı hücreler medyan ile doldurulur; "
            "veri kaybı yaşanmaz, satır sayısı sabit kalır."
        )
    elif "STATISTICAL_OUTLIER" in issue_types:
        strategy = "SYNTHETIC_SYNTHESIS"
        impact = (
            "Uç değerler doğal Gauss dağılımına uygun gerçekçi değerlerle değiştirilir; "
            "verinin varyansı ve çan eğrisi şekli korunur."
        )
    elif "HIGH_NULL_RATIO" in issue_types:
        strategy = "STATISTICAL_IMPUTE"
        impact = "Boş hücreler istatistiksel merkezi değerle tamamlanarak veri kaybı önlenir."
    else:
        strategy = "STATISTICAL_IMPUTE"
        impact = "Anomali içeren kayıtlar medyan/mod ile doldurularak veri bütünlüğü korunur."

    return f"• '{col}' ({issue_desc}):\n  ➜ Önerilen Karar: {strategy}\n  ➜ Seçimin Sonucu: {impact}"


def generate_expert_advice(profile: dict[str, Any], critical_issues: list) -> str:
    """Reads RULES.md, evaluates anomalies via local Ollama, and falls back gracefully if offline."""
    # 1. Read operational contract (RULES.md)
    rules_path = Path("RULES.md")
    rules_content = rules_path.read_text() if rules_path.exists() else "Apply standard statistical governance."

    # 2. Dynamic heuristic fallback when local LLM server is unreachable (group by column)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in critical_issues:
        col = issue.get("column", "unknown")
        grouped.setdefault(col, []).append(issue)

    recommendations = [_build_column_recommendation(col, issues) for col, issues in grouped.items()]

    heuristic_advice = (
        "[Local Rule-Engine Fallback]:\n" + "\n\n".join(recommendations)
        if recommendations
        else "[Local Rule-Engine Fallback]: No critical remediation required."
    )

    # 3. Connect to local Ollama instance
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    if not _is_ollama_online(ollama_base_url):
        return heuristic_advice

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        # Initialize local model with a short timeout so CLI doesn't hang if Ollama isn't running
        llm = ChatOllama(model=model_name, base_url=ollama_base_url, temperature=0.1, timeout=15.0)

        system_prompt = (
            f"You are a Senior Principal Data Architect enforcing this operational governance ruleset:\n\n"
            f"{rules_content}\n\n"
            f"Analyze the dataset profile and critical anomalies. Provide concise, structured recommendations in Turkish.\n"
            f"CRITICAL RULE: For each anomalous feature, you MUST recommend EXACTLY ONE decisive strategy (strictly one of: DROP, STATISTICAL_IMPUTE, SYNTHETIC_SYNTHESIS, MANUAL_INPUT). Do NOT offer multiple choices, alternatives, or words like 'veya' / 'or'. Be completely decisive.\n\n"
            f"Format strictly as:\n"
            f"• '<column_name>':\n"
            f"  ➜ Önerilen Karar: <EXACTLY_ONE_STRATEGY>\n"
            f"  ➜ Seçimin Sonucu: 1-2 sentences explaining the concrete impact on data integrity, row count, variance, or ML model bias if this strategy is chosen."
        )

        user_content = f"Dataset Profile:\n{profile}\n\nCritical Anomalies Detected:\n{critical_issues}"

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_content)])

        return f"[Ollama: {model_name}]\n\n{response.content.strip()}"

    except Exception:
        # If Ollama daemon is down or model is not pulled, safely return heuristic reasoning
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
        # 1. If advice_text is provided, attempt regex extraction for this column
        if advice_text:
            pattern = rf"['\"]?{re.escape(col)}['\"]?.*?Önerilen Karar:\s*([A-Z_]+)"
            match = re.search(pattern, advice_text, re.DOTALL | re.IGNORECASE)
            if match and match.group(1).upper() in valid_strategies:
                recommendations[col] = match.group(1).upper()
                continue

        # 2. Unified heuristic rule-based mapping (aligned with RULES.md)
        issue_types = {i.get("type", "") for i in col_issues}
        if "CLASS_IMBALANCE" in issue_types:
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
