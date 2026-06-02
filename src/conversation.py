"""
Manages per-session conversation history for multi-turn context.
"""

from collections import defaultdict
from src.logger import logger


class ConversationManager:
    def __init__(self, max_turns: int = 10):
        self.sessions: dict[str, list[dict]] = defaultdict(list)
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content})
        # Trim to max_turns (each turn = 2 messages: 1 user, 1 assistant)
        if len(self.sessions[session_id]) > self.max_turns * 2:
            trimmed = self.sessions[session_id][-(self.max_turns * 2):]
            self.sessions[session_id] = trimmed
            logger.info(f"Trimmed session history for '{session_id}' to last {self.max_turns} turns.")
        else:
            logger.info(f"Added turn for session '{session_id}'. History length: {len(self.sessions[session_id])}")

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions.pop(session_id)
            logger.info(f"Session history cleared for '{session_id}'.")
        else:
            logger.warning(f"Attempted to clear non-existent session '{session_id}'.")
