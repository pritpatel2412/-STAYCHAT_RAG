"""
Classifies the guest's message into one of five intents using Gemini.
Intents: booking_inquiry | amenity_question | complaint | staff_command | other
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

VALID_INTENTS = {
    "booking_inquiry",
    "amenity_question",
    "complaint",
    "staff_command",
    "other"
}

CLASSIFICATION_PROMPT = """You are a hotel concierge intent classifier.

Classify the guest message into EXACTLY ONE of these intents:
- booking_inquiry: Questions about reservations, check-in, check-out, cancellation, room availability, prices, payment
- amenity_question: Questions about hotel facilities, services, dining, spa, pool, gym, Wi-Fi, parking, etc.
- complaint: Expressing dissatisfaction, reporting a problem, complaining about something
- staff_command: Requesting an action from hotel staff (e.g., "please send towels", "I need housekeeping")
- other: Anything that doesn't fit the above categories

Reply with ONLY the intent label, nothing else.

Guest message: {message}"""


class IntentClassifier:
    def __init__(self):
        # Instantiate the model once for optimization
        try:
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            logger.error(f"Failed to initialize GenerativeModel for intent classification: {e}")
            self.model = None

    def classify(self, message: str) -> str:
        """Classifies intent, with safe error fallbacks to guarantee uptime."""
        if not message.strip():
            return "other"
            
        if self.model is None:
            logger.warning("GenerativeModel is uninitialized. Defaulting to intent 'other'")
            return "other"

        try:
            prompt = CLASSIFICATION_PROMPT.format(message=message)
            response = self.model.generate_content(prompt)
            intent = response.text.strip().lower()

            # Sanitize — fall back to 'other' if model goes off-script
            if intent not in VALID_INTENTS:
                logger.warning(f"Gemini returned invalid intent '{intent}'. Defaulting to 'other'")
                return "other"
                
            logger.info(f"Intent classified successfully as: '{intent}'")
            return intent
        except Exception as e:
            logger.error(f"Gemini intent classification failed: {e}")
            raise e


# Wrapper function for functional call patterns
def classify_intent(message: str) -> str:
    classifier = IntentClassifier()
    return classifier.classify(message)
