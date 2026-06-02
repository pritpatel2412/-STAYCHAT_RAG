"""
Main RAG pipeline wrapping the compiled LangGraph execution flow,
maintaining backwards compatibility.
"""

from src.graph import compiled_graph
from src.logger import logger


class RAGPipeline:
    def __init__(self):
        # Bind with compiled state workflow graph
        self.graph = compiled_graph
        # Mock attributes for backward compatibility
        self.model = True 
        self.retriever = self.graph.nodes.get("retriever") # dummy or direct lookup

    def run(
        self,
        user_message: str,
        conversation_history: list[dict]
    ) -> dict:
        """
        Runs the full LangGraph RAG pipeline.
        Returns:
          {
            "response": str,
            "intent": str,
            "language": str,
            "retrieved_docs": list,
            "guardrail_triggered": bool,
            "guardrail_reason": str | None,
            "failover_active": bool
          }
        """
        logger.info(f"Invoking LangGraph flow for user message...")

        initial_state = {
            "message": user_message,
            "session_id": "",
            "history": conversation_history,
            "language": "en",
            "intent": "other",
            "retrieved_docs": [],
            "response": "",
            "guardrail_triggered": False,
            "guardrail_reason": "",
            "failover_active": False
        }

        try:
            # Execute the LangGraph Stateflow Graph!
            final_state = self.graph.invoke(initial_state)
            
            logger.info(
                f"LangGraph execution finished successfully. "
                f"Language: {final_state['language']} | Intent: {final_state['intent']} | "
                f"Guardrail triggered: {final_state['guardrail_triggered']} | Failover: {final_state.get('failover_active', False)}"
            )

            # Build standardized response format
            return {
                "response": final_state["response"],
                "intent": final_state["intent"],
                "language": final_state["language"],
                "retrieved_docs": [
                    {"id": d["id"], "title": d["title"], "score": d.get("similarity_score", 0.0)}
                    for d in final_state["retrieved_docs"]
                ],
                "guardrail_triggered": final_state["guardrail_triggered"],
                "guardrail_reason": final_state["guardrail_reason"],
                "failover_active": final_state.get("failover_active", False)
            }
            
        except Exception as e:
            logger.error(f"Failed to execute compiled LangGraph stateflow: {e}")
            raise e

    def run_stream(self, user_message: str, conversation_history: list[dict]):
        """
        Executes a real-time streaming RAG pipeline.
        Yields JSON line-by-line event dicts:
          - Telemetry events (type: "telemetry", intent, language, retrieved_docs, failover)
          - Telemetry update events (type: "telemetry_update", failover_active)
          - Token events (type: "token", text)
          - Replacement events if guardrails block (type: "replacement", text)
        """
        logger.info(f"Invoking real-time streaming LangGraph flow for user message...")

        # 1. Run Input Guardrail Check (Input safety scanning)
        from src.input_guardrail import scan_input_safety
        import google.generativeai as genai
        from src.graph import groq_client, intent_classifier, retriever

        safety_model = None
        try:
            safety_model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            safety_model = groq_client

        safety_result = scan_input_safety(user_message, safety_model)
        if not safety_result["safe"]:
            logger.warning(f"Input Guardrail triggered in stream flow: {safety_result['reason']}")
            yield {
                "type": "telemetry",
                "intent": "adversarial",
                "language": "en",
                "retrieved_docs": [],
                "failover_active": False
            }
            yield {
                "type": "token",
                "text": safety_result["refusal_response"]
            }
            return

        # 2. Run Language Detector
        from src.language_detector import detect_language
        lang = detect_language(user_message)

        # 3. Run Intent Classifier Node (with Groq failover support)
        intent = "other"
        failover = False

        try:
            intent = intent_classifier.classify(user_message)
        except Exception as e:
            logger.warning(f"Gemini intent classification failed in stream: {e}. Attempting failover...")
            try:
                from src.intent_classifier import CLASSIFICATION_PROMPT, VALID_INTENTS
                prompt = CLASSIFICATION_PROMPT.format(message=user_message)
                resp = groq_client.generate_content(prompt)
                result_intent = resp.text.strip().lower()
                matched_intent = None
                for vi in VALID_INTENTS:
                    if vi in result_intent:
                        matched_intent = vi
                        break
                if matched_intent:
                    intent = matched_intent
                    failover = True
            except Exception as ge:
                logger.error(f"Groq failover intent classifier failed in stream: {ge}. Defaulting to 'other'")
                intent = "other"

        # 4. Run Retriever Node
        docs = retriever.retrieve(user_message)
        formatted_docs = [
            {"id": d["id"], "title": d["title"], "score": d.get("similarity_score", 0.0)}
            for d in docs
        ]

        # Yield structured RAG telemetry payload instantly!
        yield {
            "type": "telemetry",
            "intent": intent,
            "language": lang,
            "retrieved_docs": formatted_docs,
            "failover_active": failover
        }

        # 5. Run Level 1 Pre-generation Semantic Similarity check
        from src.guardrail import check_similarity_threshold, get_human_handoff
        similarity_ok = check_similarity_threshold(docs)
        if not similarity_ok:
            refusal = get_human_handoff(lang)
            yield {
                "type": "token",
                "text": refusal
            }
            return

        # 6. Format RAG prompt
        from src.graph import format_context, format_history, get_language_instruction, SYSTEM_PROMPT
        context_str = format_context(docs)
        history_str = format_history(conversation_history)
        lang_instruction = get_language_instruction(lang)

        prompt = SYSTEM_PROMPT.format(
            language_instruction=lang_instruction,
            context=context_str,
            history=history_str,
            intent=intent,
            user_message=user_message
        )

        # Inject clean business tone guidelines
        prompt_with_rules = prompt + "\n\n(IMPORTANT: Avoid using any emojis in your response. Keep the tone completely professional, executive, and direct.)"

        full_response = ""
        # 7. Execute Generation Streaming (with Groq failover redundancy)
        if not failover:
            try:
                # Stream via Gemini API
                model = genai.GenerativeModel("gemini-2.5-flash")
                response_stream = model.generate_content(prompt_with_rules, stream=True)
                for chunk in response_stream:
                    text_chunk = chunk.text
                    full_response += text_chunk
                    yield {
                        "type": "token",
                        "text": text_chunk
                    }
            except Exception as e:
                logger.warning(f"Gemini streaming generation failed: {e}. Switching to Groq failover stream...")
                failover = True
                yield {
                    "type": "telemetry_update",
                    "failover_active": True
                }

        if failover:
            try:
                # Stream via Groq completions SSE parser
                for text_chunk in groq_client.generate_content_stream(prompt_with_rules):
                    full_response += text_chunk
                    yield {
                        "type": "token",
                        "text": text_chunk
                    }
            except Exception as ge:
                logger.error(f"Groq streaming generation failed: {ge}. Defaulting to human handoff.")
                yield {
                    "type": "token",
                    "text": get_human_handoff(lang)
                }
                return

        # 8. Post-generation Level 2 Output Guardrail Check over the completed response text
        from src.guardrail import apply_guardrail
        guardrail_result = apply_guardrail(full_response, docs, lang)
        if not guardrail_result["safe"]:
            logger.warning(f"Post-gen Output Guardrail triggered: {guardrail_result['reason']}")
            # Instruct frontend to replace entire streamed content with safe refusal
            yield {
                "type": "replacement",
                "text": guardrail_result["text"]
            }

