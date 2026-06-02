"""
LangGraph state flow orchestrator. Links input guardrails, intent routing,
hybrid retrieval, failover generation, and output guardrails.
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os

from src.retriever import HotelRetriever
from src.intent_classifier import IntentClassifier
from src.language_detector import detect_language
from src.guardrail import apply_guardrail, get_human_handoff, check_similarity_threshold
from src.input_guardrail import scan_input_safety
from src.groq_client import GroqClient
from src.logger import logger


# Define unified state dictionary for LangGraph transitions
class AgentState(TypedDict):
    message: str
    session_id: str
    history: List[Dict[str, str]]
    language: str
    intent: str
    retrieved_docs: List[Dict[str, Any]]
    response: str
    guardrail_triggered: bool
    guardrail_reason: str
    failover_active: bool


# Initialize clients
retriever = HotelRetriever(top_k=4)
intent_classifier = IntentClassifier()
groq_client = GroqClient()

GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a professional hotel concierge assistant for The Grand Hotel, Mumbai.

CRITICAL RULES — you MUST follow these without exception:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. NEVER invent, assume, or estimate prices, room rates, or fees that are not explicitly stated in the CONTEXT.
3. NEVER provide payment links, booking URLs, or reservation systems not mentioned in the CONTEXT.
4. If the CONTEXT does not contain enough information to answer the question, say so clearly and offer to connect the guest with a human team member.
5. Do not make up contact details, phone numbers, or email addresses not present in the CONTEXT.
6. Be warm, professional, and concise.
7. Respond in the SAME LANGUAGE as the guest's message ({language_instruction}).
8. NEVER use emojis in your response under any circumstances. Keep the tone completely professional, premium, and corporate.

CONTEXT (retrieved from hotel knowledge base):
{context}

CONVERSATION HISTORY:
{history}

INTENT: {intent}
GUEST MESSAGE: {user_message}

Respond to the guest now:"""


def format_context(retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return "No relevant information found."
    sections = []
    for doc in retrieved_docs:
        # Support either semantic or hybrid bm25 scores
        score_info = f"Score: RRF={doc.get('rrf_score', 0):.4f}"
        sections.append(
            f"[{doc['category'].upper()}] {doc['title']} ({score_info})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(sections)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous messages."
    lines = []
    for turn in history[-6:]:
        role = "Guest" if turn["role"] == "user" else "Concierge"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def get_language_instruction(language: str) -> str:
    return {
        "en": "English",
        "hi": "Hindi (Devanagari script)",
        "hinglish": "Hinglish (Roman script Hindi-English mix)"
    }.get(language, "English")


# ─── GRAPH NODES ─────────────────────────────────────────────────────────────

def input_guardrail_node(state: AgentState) -> AgentState:
    """Interceptors adversarial inputs and prompt injections."""
    logger.info("LangGraph Node: [input_guardrail] starting...")
    
    # Run safety checks
    # Try using Gemini for safety checks, fallback to Groq if rate-limited!
    safety_model = None
    try:
        safety_model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        safety_model = groq_client
        
    safety_result = scan_input_safety(state["message"], safety_model)
    
    if not safety_result["safe"]:
        return {
            **state,
            "response": safety_result["refusal_response"],
            "guardrail_triggered": True,
            "guardrail_reason": safety_result["reason"],
            "intent": "adversarial"
        }
        
    return {
        **state,
        "guardrail_triggered": False,
        "guardrail_reason": ""
    }


def language_detector_node(state: AgentState) -> AgentState:
    """Detects message tongue using optimized local logic."""
    logger.info("LangGraph Node: [language_detector] starting...")
    lang = detect_language(state["message"])
    return {
        **state,
        "language": lang
    }


def intent_classifier_node(state: AgentState) -> AgentState:
    """Classifies user intent, falling back to Groq under rate-limits."""
    logger.info("LangGraph Node: [intent_classifier] starting...")
    intent = "other"
    failover = state.get("failover_active", False)
    
    try:
        intent = intent_classifier.classify(state["message"])
    except Exception as e:
        logger.warning(f"Gemini intent classification failed: {e}. Attempting failover to Groq...")
        try:
            # Re-create classification using Groq
            from src.intent_classifier import CLASSIFICATION_PROMPT, VALID_INTENTS
            prompt = CLASSIFICATION_PROMPT.format(message=state["message"])
            resp = groq_client.generate_content(prompt)
            result_intent = resp.text.strip().lower()
            
            # Extract valid intent via substring matching
            matched_intent = None
            for vi in VALID_INTENTS:
                if vi in result_intent:
                    matched_intent = vi
                    break
                    
            if matched_intent:
                intent = matched_intent
                failover = True
                logger.info(f"Groq failover intent classified successfully as: '{intent}'")
        except Exception as ge:
            logger.error(f"Groq failover intent classification failed: {ge}. Defaulting to 'other'")
            intent = "other"
            
    return {
        **state,
        "intent": intent,
        "failover_active": failover
    }


def retriever_node(state: AgentState) -> AgentState:
    """Performs Hybrid search vector retrieval."""
    logger.info("LangGraph Node: [retriever] starting...")
    docs = retriever.retrieve(state["message"])
    return {
        **state,
        "retrieved_docs": docs
    }


def response_generator_node(state: AgentState) -> AgentState:
    """Generates grounded responses with multi-LLM rate-limits failover."""
    logger.info("LangGraph Node: [response_generator] starting...")
    docs = state["retrieved_docs"]
    lang = state["language"]
    intent = state["intent"]
    failover = state.get("failover_active", False)

    # 1. Level 1 Pre-generation semantic similarity guardrail check
    # Check if ANY of the retrieved docs cleared the 0.40 similarity threshold
    similarity_ok = check_similarity_threshold(docs)
    if not similarity_ok:
        score = docs[0]["similarity_score"] if docs else 0.0
        reason = f"Low similarity score: {score:.4f} (Threshold: 0.40)"
        logger.warning(f"Guardrail Level 1 triggered in LangGraph: {reason}")
        return {
            **state,
            "response": get_human_handoff(lang),
            "guardrail_triggered": True,
            "guardrail_reason": reason
        }

    # 2. Build system RAG prompt
    context = format_context(docs)
    history = format_history(state["history"])
    lang_instruction = get_language_instruction(lang)

    prompt = SYSTEM_PROMPT.format(
        language_instruction=lang_instruction,
        context=context,
        history=history,
        intent=intent,
        user_message=state["message"]
    )

    response_text = ""
    # 3. Attempt Generation via Gemini
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        response_text = response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini generation failed due to error: {e}. Triggering Groq failover node...")
        # Failover trigger
        try:
            resp = groq_client.generate_content(prompt)
            response_text = resp.text.strip()
            failover = True
        except Exception as ge:
            logger.error(f"Groq failover generation failed: {ge}. Returning human handoff.")
            return {
                **state,
                "response": get_human_handoff(lang),
                "guardrail_triggered": True,
                "guardrail_reason": f"API failover block: {str(ge)}"
            }

    return {
        **state,
        "response": response_text,
        "failover_active": failover
    }


def output_guardrail_node(state: AgentState) -> AgentState:
    """Scans response for post-gen hallucinations and domain safety."""
    logger.info("LangGraph Node: [output_guardrail] starting...")
    
    # If a guardrail was already triggered in a prior node, bypass output guardrail
    if state.get("guardrail_triggered", False):
        return state
        
    guardrail_result = apply_guardrail(
        state["response"],
        state["retrieved_docs"],
        state["language"]
    )
    
    return {
        **state,
        "response": guardrail_result["text"],
        "guardrail_triggered": not guardrail_result["safe"],
        "guardrail_reason": guardrail_result["reason"]
    }


# ─── STATE GRAPH COMPILATION ──────────────────────────────────────────────────

workflow = StateGraph(AgentState)

# Register graph nodes
workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("language_detector", language_detector_node)
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("response_generator", response_generator_node)
workflow.add_node("output_guardrail", output_guardrail_node)

# Define conditional route after input guardrail check
def route_after_input_guardrail(state: AgentState):
    if state.get("guardrail_triggered", False):
        logger.warning("Input Guardrail triggered: routing directly to END.")
        return END
    return "language_detector"

workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    {
        END: END,
        "language_detector": "language_detector"
    }
)

# Set standard edges sequential flow
workflow.add_edge("language_detector", "intent_classifier")
workflow.add_edge("intent_classifier", "retriever")
workflow.add_edge("retriever", "response_generator")
workflow.add_edge("response_generator", "output_guardrail")
workflow.add_edge("output_guardrail", END)

# Set start entry point
workflow.set_entry_point("input_guardrail")

# Compile state flow graph
compiled_graph = workflow.compile()
logger.info("Successfully compiled LangGraph state workflow graph application.")
