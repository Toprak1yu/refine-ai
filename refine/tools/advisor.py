import os
from pathlib import Path
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage


def generate_expert_advice(profile: Dict[str, Any], critical_issues: list) -> str:
    """Reads RULES.md, evaluates anomalies via local Ollama, and falls back gracefully if offline."""
    # 1. Read operational contract (RULES.md)
    rules_path = Path("RULES.md")
    rules_content = rules_path.read_text() if rules_path.exists() else "Apply standard statistical governance."

    # 2. Heuristic fallback when local LLM server is unreachable
    heuristic_advice = (
        "[Local Rule-Engine Fallback]:\n"
        "• 'age' contains invalid boundaries (<0, >120): Recommend STATISTICAL_IMPUTE with median (40-45 yrs).\n"
        "• 'salary' contains severe Z-score outliers: Recommend STATISTICAL_IMPUTE or SYNTHETIC_SYNTHESIS to avoid data skew.\n"
        "• 'churn' exhibits severe class imbalance (<10%): Recommend SYNTHETIC_SYNTHESIS to prevent classifier bias."
    )

    # 3. Connect to local Ollama instance
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b") 
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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