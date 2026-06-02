"""
Main RAG pipeline combining retrieval, intent classification, language detection,
generation, and guardrail enforcement.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

from src.retriever import HotelRetriever
from src.intent_classifier import IntentClassifier
from src.language_detector import detect_language
from src.guardrail import apply_guardrail, get_human_handoff, check_similarity_threshold
from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

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
        sections.append(
            f"[{doc['category'].upper()}] {doc['title']}\n{doc['content']}"
        )
    return "\n\n---\n\n".join(sections)


def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous messages."
    lines = []
    # Keep last 3 turns (6 messages)
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


class RAGPipeline:
    def __init__(self):
        self.retriever = HotelRetriever(top_k=4)
        self.intent_classifier = IntentClassifier()
        try:
            self.model = genai.GenerativeModel(GENERATION_MODEL)
            logger.info("RAGPipeline initialized with Gemini successfully.")
        except Exception as e:
            logger.error(f"Failed to load GenerativeModel in RAGPipeline: {e}")
            self.model = None

    def run(
        self,
        user_message: str,
        conversation_history: list[dict]
    ) -> dict:
        """
        Full RAG pipeline.
        Returns:
          {
            "response": str,
            "intent": str,
            "language": str,
            "retrieved_docs": list,
            "guardrail_triggered": bool,
            "guardrail_reason": str | None
          }
        """
        logger.info(f"Running pipeline for query: '{user_message[:40]}...'")

        # 1. Detect language
        language = detect_language(user_message)

        # 2. Classify intent
        intent = self.intent_classifier.classify(user_message)

        # 3. Retrieve relevant KB entries via FAISS
        retrieved_docs = self.retriever.retrieve(user_message)

        # 4. Pre-generation guardrail: check similarity threshold
        if not check_similarity_threshold(retrieved_docs):
            score = retrieved_docs[0]["similarity_score"] if retrieved_docs else 0.0
            logger.warning(
                f"Level 1 Guardrail triggered: Similarity score {score:.4f} is below threshold."
            )
            return {
                "response": get_human_handoff(language),
                "intent": intent,
                "language": language,
                "retrieved_docs": [
                    {"id": d["id"], "title": d["title"], "score": d["similarity_score"]}
                    for d in retrieved_docs
                ],
                "guardrail_triggered": True,
                "guardrail_reason": f"Low similarity ({score:.4f}) — topic not in KB"
            }

        # 5. Build prompt
        context = format_context(retrieved_docs)
        history = format_history(conversation_history)
        lang_instruction = get_language_instruction(language)

        prompt = SYSTEM_PROMPT.format(
            language_instruction=lang_instruction,
            context=context,
            history=history,
            intent=intent,
            user_message=user_message
        )

        # 6. Generate response
        if self.model is None:
            logger.error("GenerativeModel is uninitialized in run. Triggering human handoff.")
            return {
                "response": get_human_handoff(language),
                "intent": intent,
                "language": language,
                "retrieved_docs": [
                    {"id": d["id"], "title": d["title"], "score": d["similarity_score"]}
                    for d in retrieved_docs
                ],
                "guardrail_triggered": True,
                "guardrail_reason": "Gemini model uninitialized"
            }

        try:
            response = self.model.generate_content(prompt)
            generated_text = response.text.strip()
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            return {
                "response": get_human_handoff(language),
                "intent": intent,
                "language": language,
                "retrieved_docs": [
                    {"id": d["id"], "title": d["title"], "score": d["similarity_score"]}
                    for d in retrieved_docs
                ],
                "guardrail_triggered": True,
                "guardrail_reason": f"Gemini API failure: {str(e)}"
            }

        # 7. Post-generation guardrail: scan for hallucinations
        guardrail_result = apply_guardrail(generated_text, retrieved_docs, language)

        logger.info(
            f"Pipeline run complete. Guardrail triggered: {not guardrail_result['safe']}"
        )
        return {
            "response": guardrail_result["text"],
            "intent": intent,
            "language": language,
            "retrieved_docs": [
                {"id": d["id"], "title": d["title"], "score": d["similarity_score"]}
                for d in retrieved_docs
            ],
            "guardrail_triggered": not guardrail_result["safe"],
            "guardrail_reason": guardrail_result["reason"]
        }
