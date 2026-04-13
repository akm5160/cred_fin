import os
import json
import logging
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
 

from openai import AzureOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.infra.llm import client
from app.infra.config_models import CIBILDetails, CreditRiskSummary, MLScores, LoanHistory, RiskDecision, UnderwritingState
from app.infra.cache import cache_get
from app.ML.cibil import fetch_cibil
from app.ML.pd import predict_credit_risk
from app.infra.rl import check_rate_limit
# ============================================================
# 4. LANGGRAPH NODES
# ============================================================
 
llm = client
 
 
def node_rate_limiter(state: UnderwritingState) -> dict:
    """Guard node: check rate limits and token budget."""
    allowed, reason = check_rate_limit(state.identifier_person)
    if not allowed:
        return {"error_log": state.error_log + [f"Rate limit: {reason}"]}
    return {}
 
 
def node_init(state: UnderwritingState) -> dict:
    """
    GenAI node: parse and normalize input.
    Production: LLM extracts fields from unstructured docs (PDFs, scans).
    """
    logging.info(f"[INIT] loan={state.loan_account_id}")
    return {"updated_at": datetime.now(timezone.utc).isoformat()}
 
 
def node_data_ingestion(state: UnderwritingState) -> dict:
    """GenAI node: LLM checks if application has sufficient data."""
    prompt = f"""You are a credit analyst. Check if this loan application has enough data for underwriting.
Respond ONLY with valid JSON (no markdown).
 
Application: name={state.applicant_name}, loan=₹{state.loan_amount:,.0f}, income=₹{state.current_income:,.0f},
DTI={state.current_dti}%, employment={state.employment_years}yr, purpose={state.loan_purpose},
history={len(state.history)} loans, CIBIL={'yes' if state.cibil_details else 'no'}
 
JSON: {{"is_sufficient": true/false, "missing_fields": [], "questions": []}}"""
 
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(content)
 
    return {
        "is_data_sufficient": result.get("is_sufficient", False),
        "clarification_questions": result.get("questions", []),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
 
 
def node_ask_clarification(state: UnderwritingState) -> dict:
    """
    GenAI node: generate clarification questions.
    Production: human-in-the-loop interrupt — analyst provides missing data.
    """
    logging.info(f"[CLARIFICATION] Round {state.clarification_count + 1}: {state.clarification_questions}")
    return {
        "clarification_count": state.clarification_count + 1,
        "is_data_sufficient": True,  # Simulate analyst response
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
 
 
def node_cibil_fetcher(state: UnderwritingState) -> dict:
    """External tool node: fetch credit bureau score."""
    logging.info(f"[CIBIL] Fetching for {state.identifier_person}")
    return {"cibil_details": fetch_cibil(state.identifier_person)}
 
 
def node_cache_check(state: UnderwritingState) -> dict:
    """Cache node: check if result exists for this loan_id."""
    cached = cache_get(state.loan_account_id)
    if cached:
        return {
            "cache_hit": True,
            "ml_scores": MLScores(**cached["ml_scores"]),
            "risk_summary": CreditRiskSummary(**cached["risk_summary"]),
        }
    return {"cache_hit": False}
 
 
def node_decision_engine(state: UnderwritingState) -> dict:
    """Programmatic node: orchestrate ML model scoring."""
    logging.info("[DECISION ENGINE] Running ML models...")
    return {"ml_scores": predict_credit_risk(state)}
 
 
def node_recommendation(state: UnderwritingState) -> dict:
    """GenAI node: LLM synthesizes ML scores into risk narrative + decision."""
    scores = state.ml_scores
    cibil = state.cibil_details
 
    prompt = f"""You are a senior credit risk analyst. Generate a credit risk assessment.
Respond ONLY with valid JSON (no markdown).
 
APPLICANT: {state.applicant_name}, Loan ₹{state.loan_amount:,.0f} ({state.loan_purpose}),
Income ₹{state.current_income:,.0f}, DTI {state.current_dti}%, Employment {state.employment_years}yr,
History: {len(state.history)} loans, Dependents: {len(state.dependents)}
 
CIBIL: score={cibil.score if cibil else 'N/A'}, utilization={cibil.credit_utilization if cibil else 'N/A'}%,
overdue={cibil.overdue_accounts if cibil else 'N/A'}
 
ML SCORES: PD={scores.pd_score:.2%}, rate={scores.interest_rate_forecast}%,
sector_risk={scores.sector_risk_score:.2f}, country_risk={scores.country_risk_score:.2f},
features={json.dumps(scores.feature_importances)}
 
RULES: PD<0.15 AND CIBIL>720 → APPROVE | PD>0.50 OR CIBIL<600 → REJECT | else → REVIEW
 
JSON: {{"narrative": "2-3 para summary with numbers", "decision": "APPROVE|REVIEW|REJECT",
"confidence": 0.0-1.0, "key_risk_drivers": ["..."], "mitigating_factors": ["..."],
"recommended_conditions": ["..."]}}"""
 
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
 
    risk_summary = CreditRiskSummary(**json.loads(content))
    cache_set(state.loan_account_id, {"ml_scores": scores.model_dump(), "risk_summary": risk_summary.model_dump()})
 
    return {"risk_summary": risk_summary}
 
 
def node_update_state(state: UnderwritingState) -> dict:
    """Programmatic node: persist final decision chain to DB."""
    logging.info(f"[UPDATE STATE] Decision: {state.risk_summary.decision if state.risk_summary else 'N/A'}")
    return {"updated_at": datetime.now(timezone.utc).isoformat()}
 
 
# ============================================================
# 5. ROUTING
# ============================================================
 
def route_after_data_check(state: UnderwritingState) -> str:
    if state.is_data_sufficient:
        return "cibil_fetcher"
    if state.clarification_count >= state.max_clarifications:
        return "cibil_fetcher"
    return "ask_clarification"
 
 
def route_after_cache(state: UnderwritingState) -> str:
    return "update_state" if state.cache_hit else "decision_engine"
 
 
# ============================================================
# 6. GRAPH
# ============================================================
 
def build_graph() -> StateGraph:
    graph = StateGraph(UnderwritingState)
 
    graph.add_node("rate_limiter", node_rate_limiter)
    graph.add_node("init", node_init)
    graph.add_node("data_ingestion", node_data_ingestion)
    graph.add_node("ask_clarification", node_ask_clarification)
    graph.add_node("cibil_fetcher", node_cibil_fetcher)
    graph.add_node("cache_check", node_cache_check)
    graph.add_node("decision_engine", node_decision_engine)
    graph.add_node("recommendation", node_recommendation)
    graph.add_node("update_state", node_update_state)
 
    graph.add_edge(START, "rate_limiter")
    graph.add_edge("rate_limiter", "init")
    graph.add_edge("init", "data_ingestion")
    graph.add_conditional_edges("data_ingestion", route_after_data_check,
        {"cibil_fetcher": "cibil_fetcher", "ask_clarification": "ask_clarification"})
    graph.add_edge("ask_clarification", "data_ingestion")
    graph.add_edge("cibil_fetcher", "cache_check")
    graph.add_conditional_edges("cache_check", route_after_cache,
        {"update_state": "update_state", "decision_engine": "decision_engine"})
    graph.add_edge("decision_engine", "recommendation")
    graph.add_edge("recommendation", "update_state")
    graph.add_edge("update_state", END)
 
    return graph
 
 
def get_checkpointer():
    """CosmosDB checkpointer if configured, else MemorySaver."""
    cosmos_conn = os.getenv("COSMOS_DB_CONNECTION_STRING")
    if cosmos_conn:
        try:
            from langgraph.checkpoint.cosmosdb import CosmosDBSaver
            return CosmosDBSaver(
                connection_string=cosmos_conn,
                database_name=os.getenv("COSMOS_DB_NAME", "underwriting_copilot"),
                container_name="checkpoints",
            )
        except (ImportError, Exception) as e:
            logging.warning(f"CosmosDB unavailable ({e}), using MemorySaver")
    return MemorySaver()


if __name__=="__main__":

    from dotenv import load_dotenv
    load_dotenv()

    graph = build_graph()
    graph = graph.compile()
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved to graph.png")