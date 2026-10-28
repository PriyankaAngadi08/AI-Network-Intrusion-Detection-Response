
from investigation.rag.retriever import CybersecurityRetriever
from investigation.llm.prompts import build_investigation_prompt
from investigation.llm.llm_client import investigate_with_llm
from investigation.reports.report_generator import generate_report

from threat_intelligence.ioc_extractor import extract_iocs
from threat_intelligence.enrichment import enrich_iocs
from threat_intelligence.mitre_mapping import map_attack


# Reuse the retriever and its index within this Python process.
_retriever = None


def get_retriever():
    global _retriever

    if _retriever is None:
        _retriever = CybersecurityRetriever()

    return _retriever


def investigate_alert(alert) -> dict:
    """Investigate an alert using RAG and a local LLM."""

    # Step 1: Convert the alert to a dictionary.
    if hasattr(alert, "model_dump"):
        alert_data = alert.model_dump()
    else:
        alert_data = alert.dict()

    # Step 2: Extract and enrich IP indicators.
    iocs = extract_iocs(alert)
    enriched_iocs = enrich_iocs(iocs)

    # Step 3: Map the attack type to a candidate MITRE technique.
    mitre_result = map_attack(alert.attack_type)

    # Step 4: Retrieve relevant cybersecurity knowledge.
    query = (
        f"Investigate {alert.attack_type}. "
        f"Source IP: {alert.source_ip}. "
        f"Destination IP: {alert.destination_ip}. "
        "What evidence should be checked and what defensive actions "
        "may be appropriate?"
    )

    evidence = get_retriever().retrieve(query, top_k=3)

    # Step 5: Build the prompt using the alert and retrieved evidence.
    prompt = build_investigation_prompt(
        alert=alert_data,
        iocs=enriched_iocs,
        mitre_technique=mitre_result,
        evidence=evidence,
    )

    # Step 6: Ask the local Ollama LLM to generate an investigation.
    ai_analysis = investigate_with_llm(prompt)

    # Step 7: Prepare the investigation data.
    investigation = {
        "investigation_id": f"INV-{alert.alert_id}",
        "alert_id": alert.alert_id,
        "status": "completed",
        "attack_type": alert.attack_type,
        "label": alert.label,
        "detection_confidence": alert.confidence,
        "dataset": alert.dataset,
        "iocs": enriched_iocs,
        "mitre_technique": mitre_result,
        "retrieved_evidence": [
            {
                "source": item["source"],
                "text": item["text"],
                "similarity": item["similarity"],
            }
            for item in evidence
        ],
        "ai_analysis": ai_analysis,
        "summary": ai_analysis.get(
            "summary", "The AI did not provide a summary."
        ),
        "severity": ai_analysis.get("severity", "undetermined"),
        "uncertainties": ai_analysis.get("uncertainties", []),
        "recommendations": ai_analysis.get("recommendations", []),
        "sources": ai_analysis.get("sources", []),
    }

    # Step 8: Generate the final structured report.
    return generate_report(investigation)
