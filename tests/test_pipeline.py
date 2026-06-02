import pytest
from src.conversation import ConversationManager
from src.graph import format_context, format_history, get_language_instruction


def test_conversation_manager():
    mgr = ConversationManager(max_turns=2)
    
    # Session 1 turns
    mgr.add_turn("session_123", "user", "Hello")
    mgr.add_turn("session_123", "assistant", "Hi there")
    mgr.add_turn("session_123", "user", "How are you?")
    mgr.add_turn("session_123", "assistant", "I am well")
    
    history = mgr.get_history("session_123")
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there"

    # Trimming test: Adding a 3rd turn (total 6 messages) should trim to 2 turns (4 messages)
    mgr.add_turn("session_123", "user", "What is check-in time?")
    mgr.add_turn("session_123", "assistant", "Check-in is at 2:00 PM")
    
    trimmed_history = mgr.get_history("session_123")
    assert len(trimmed_history) == 4  # Trims to max_turns * 2 (4 messages)
    assert trimmed_history[0]["content"] == "How are you?"  # Oldest turn "Hello/Hi" is purged!


def test_format_context():
    docs = [
        {"category": "amenity", "title": "Gym", "content": "24 hours gym."},
        {"category": "dining", "title": "Cafe", "content": "Open 7 AM."}
    ]
    formatted = format_context(docs)
    assert "[AMENITY] Gym" in formatted
    assert "24 hours gym." in formatted
    assert "[DINING] Cafe" in formatted
    
    formatted_empty = format_context([])
    assert "No relevant information found." in formatted_empty


def test_format_history():
    history = [
        {"role": "user", "content": "Can I check in early?"},
        {"role": "assistant", "content": "Subject to availability."}
    ]
    formatted = format_history(history)
    assert "Guest: Can I check in early?" in formatted
    assert "Concierge: Subject to availability." in formatted

    formatted_empty = format_history([])
    assert "No previous messages." in formatted_empty


def test_get_language_instruction():
    assert get_language_instruction("en") == "English"
    assert "Hinglish" in get_language_instruction("hinglish")
    assert "Hindi" in get_language_instruction("hi")
    assert get_language_instruction("fr") == "English"  # fallback
