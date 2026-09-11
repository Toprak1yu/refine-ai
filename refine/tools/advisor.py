import os
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


def generate_expert_advice(profile: dict[str, Any], critical_issues: list) -> str:
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
            recommendations.append(
                f"• '{col}' (Geçersiz Sınır Değerleri):\n"
                f"  ➜ Önerilen Karar: STATISTICAL_IMPUTE\n"
                f"  ➜ Seçimin Sonucu: Bozuk ve geçersiz hücreler medyan ile doldurulur; veri kaybı yaşanmaz, satır sayısı sabit kalır."
            )
        elif itype == "STATISTICAL_OUTLIER":
            recommendations.append(
                f"• '{col}' (Aşırı Uç Değerler):\n"
                f"  ➜ Önerilen Karar: SYNTHETIC_SYNTHESIS\n"
                f"  ➜ Seçimin Sonucu: Uç değerler doğal Gauss dağılımına uygun gerçekçi değerlerle değiştirilir; verinin varyansı ve çan eğrisi şekli korunur."
            )
        elif itype == "CLASS_IMBALANCE":
            ratio = issue.get("minority_ratio", 0) * 100
            val = issue.get("minority_value", 1)
            recommendations.append(
                f"• '{col}' (Sınıf Dengesizliği - %{ratio:.1f}):\n"
                f"  ➜ Önerilen Karar: SYNTHETIC_SYNTHESIS\n"
                f"  ➜ Seçimin Sonucu: Azınlık sınıfı ({val}) için sentetik satırlar türetilerek denge %35'e çıkarılır; modelin yanlı (biased) öğrenmesi engellenir, toplam satır sayısı artar."
            )
        elif itype == "HIGH_NULL_RATIO":
            ratio = issue.get("ratio", 0) * 100
            recommendations.append(
                f"• '{col}' (Yüksek Boşluk Oranı - %{ratio:.1f}):\n"
                f"  ➜ Önerilen Karar: STATISTICAL_IMPUTE\n"
                f"  ➜ Seçimin Sonucu: Boş hücreler istatistiksel merkezi değerle tamamlanarak veri kaybı önlenir."
            )
        else:
            recommendations.append(
                f"• '{col}' ({itype}):\n"
                f"  ➜ Önerilen Karar: STATISTICAL_IMPUTE\n"
                f"  ➜ Seçimin Sonucu: Anomali içeren kayıtlar medyan/mod ile doldurularak veri bütünlüğü korunur."
            )

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

        return f"[Ollama: {model_name}]\n{response.content.strip()}"

    except Exception:
        # If Ollama daemon is down or model is not pulled, safely return heuristic reasoning
        return heuristic_advice
