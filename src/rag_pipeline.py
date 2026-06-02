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
