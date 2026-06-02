"""
Detects language of guest message: 'en', 'hi', or 'hinglish'.
Uses a simple heuristic + Gemini fallback for ambiguous cases.
"""

import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Unicode range for Devanagari script (Hindi)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common Hinglish marker words (Excluding "please" to fix the English "please" bug!)
HINGLISH_MARKERS = {
    "kya", "hai", "mujhe", "aapka", "kab", "kaise", "hain",
    "chahiye", "batao", "bata", "yahan", "wahan",
    "kitne", "kitna", "accha", "theek", "mera", "meri", "humare",
    "hamare", "kamra", "room", "milega", "karna", "karne"
}


def detect_language(text: str) -> str:
    """
    Detects language: 'en', 'hi', or 'hinglish'.
    Uses local heuristics first. If ambiguous, falls back to Gemini to verify.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return "en"

    # 1. Unicode range check: contains Devanagari characters -> Hindi
    if DEVANAGARI_RE.search(text_stripped):
        logger.info(f"Devanagari detected: Language classified as 'hi'")
        return "hi"

    # 2. Local heuristic check for Hinglish markers
    text_lower = text_stripped.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    # Intersect with our Hinglish markers
    matching_markers = words & HINGLISH_MARKERS
    if matching_markers:
        logger.info(f"Hinglish markers matched: {matching_markers}. Language classified as 'hinglish'")
        return "hinglish"

    # 3. Fallback: If no markers match, use a lightweight Gemini classification fallback
    # to catch Romanized Hindi sentences that did not match our keyword list (e.g. "room service de do").
    logger.info("Local heuristics returned default English. Querying Gemini for confirmation fallback...")
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Analyze the following hotel guest message and classify its language into EXACTLY ONE of these labels: 'en', 'hi', or 'hinglish'.\n"
            "- 'en': standard English text (even with polite terms like please)\n"
            "- 'hi': Hindi text in Devanagari script\n"
            "- 'hinglish': Hindi text written in Roman/English script or a heavy blend of English and Hindi\n\n"
            f"Message: {text_stripped}\n\n"
            "Reply with only the label ('en', 'hi', or 'hinglish') and nothing else."
        )
        response = model.generate_content(prompt)
        lang = response.text.strip().lower()
        if lang in {"en", "hi", "hinglish"}:
            logger.info(f"Gemini fallback successfully classified language as '{lang}'")
            return lang
    except Exception as e:
        logger.error(f"Gemini language fallback classifier failed: {e}. Falling back to default 'en'")
        
    return "en"
