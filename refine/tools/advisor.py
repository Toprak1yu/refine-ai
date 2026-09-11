import os
import urllib.request
from pathlib import Path
from typing import Any, Dict


def _is_ollama_online(base_url: str) -> bool:
    """Fast check (timeout 0.3s) if local Ollama daemon is reachable."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.3):
            return True
    except Exception:
        return False


def generate_expert_advice(profile: Dict[str, Any], critical_issues: list) -> str:
    """Reads RULES.md, evaluates anomalies via local Ollama, and falls back gracefully if offline."""
    # 1. Read operational contract (RULES.md)
    rules_path = Path("RULES.md")
    rules_content = rules_path.read_text() if rules_path.exists() else "Apply standard statistical governance."

    # 2. Dynamic heuristic fallback when local LLM server is unreachable
    recommendations = []
    for issue in critical_issues:
        col = issue.get("column", "unknown")
        itype = issue.get("type", "")
        if itype == "INVALID_BOUNDS":
            recommendations.append(f"• '{col}' contains invalid boundaries: Recommend STATISTICAL_IMPUTE with median.")
        elif itype == "STATISTICAL_OUTLIER":
            recommendations.append(f"• '{col}' contains severe Z-score outliers: Recommend STATISTICAL_IMPUTE or SYNTHETIC_SYNTHESIS.")
        elif itype == "CLASS_IMBALANCE":
            recommendations.append(f"• '{col}' exhibits severe class imbalance: Recommend SYNTHETIC_SYNTHESIS to prevent classifier bias.")
        elif itype == "HIGH_NULL_RATIO":
            recommendations.append(f"• '{col}' has high missingness ratio (>=20%): Recommend STATISTICAL_IMPUTE or DROP.")
        else:
            recommendations.append(f"• '{col}' anomaly detected ({itype}): Recommend STATISTICAL_IMPUTE.")

    heuristic_advice = (
        "[Local Rule-Engine Fallback]:\n" + "\n".join(recommendations)
        if recommendations
        else "[Local Rule-Engine Fallback]: No critical remediation required."
    )

    # 3. Connect to local Ollama instance
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b") 
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    if not _is_ollama_online(ollama_base_url):
        return heuristic_advice

    try:
        from langchain_ollama import ChatOllama

        # Initialize local model with a short timeout so CLI doesn't hang if Ollama isn't running
        llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1,
            timeout=10.0
        )

        system_prompt = (
            f"You are a Senior Principal Data Architect enforcing this operational governance ruleset:\n\n"
            f"{rules_content}\n\n"
            f"Analyze the dataset profile and critical anomalies. Provide concise, bulleted recommendations. "
            f"For each anomalous feature, explicitly advise whether to use [1] DROP, [2] STATISTICAL_IMPUTE, "
            f"or [3] SYNTHETIC_SYNTHESIS and provide a 1-sentence technical justification."
        )

        user_content = f"Dataset Profile:\n{profile}\n\nCritical Anomalies Detected:\n{critical_issues}"

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        
        return f"[Ollama: {model_name}]\n{response.content.strip()}"

    except Exception:
        # If Ollama daemon is down or model is not pulled, safely return heuristic reasoning
        return heuristic_advice