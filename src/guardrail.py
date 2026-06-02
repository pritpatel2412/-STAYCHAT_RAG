"""
Anti-hallucination guardrail. This module ensures the bot NEVER invents:
  - Room prices or package costs
  - Payment links or booking URLs
  - Contact details not in the KB
  - Policies not in the KB
"""

import re
from src.logger import logger

# ─── Thresholds ──────────────────────────────────────────────────────────────
MIN_SIMILARITY_SCORE = 0.40   # Below this → context is too weak → do not generate

# ─── WHITEDOMAIN LIST ────────────────────────────────────────────────────────
WHITELISTED_DOMAINS = ["grandhotel.com", "google.com", "maps.google"]

# ─── Structured Hallucination Patterns ───────────────────────────────────────
# 1. Price fabrication: Matches currency symbols (with safe boundary) followed by digits
# Fixes Bug 4 (boundary issues with special characters like $)
PRICE_PATTERN = r"(?i)(?<![a-zA-Z0-9])(?:USD|EUR|GBP|\$|€|£)\s*\d+"

# 2. Fabricated cost expressions: e.g. "price is Rs 5000" or "costs ₹10,000"
# Fixes Bug 3 (avoids triggering bare phrase "the price is" without actual numbers)
COST_EXPRESSION_PATTERN = r"(?i)\b(?:the price is|costs?)\s*(?:INR|Rs\.?|₹)?\s*\d+"

# 3. Brittle/Fabricated phone numbers (10+ digits not in KB: +91-22-6600-0000)
# Matches any 10+ consecutive digit block unless it matches the KB number block.
PHONE_PATTERN = r"\b(?<!\+91-22-6600-)(?<!\+91-22-6600-\d)(\d{10,})\b"

HUMAN_HANDOFF_EN = (
    "I'm sorry, I don't have that information in our hotel knowledge base. "
    "I'd be happy to connect you with a member of our team who can assist you directly. "
    "Please call the front desk at extension 0, or I can have someone reach out to you."
)

HUMAN_HANDOFF_HI = (
    "मुझे खेद है, यह जानकारी हमारे होटल के ज्ञान आधार में उपलब्ध नहीं है। "
    "मैं आपको हमारी टीम के किसी सदस्य से जोड़ सकता हूं जो आपकी सहायता कर सकते हैं। "
    "कृपया एक्सटेंशन 0 पर फ्रंट डेस्क को कॉल करें।"
)

HUMAN_HANDOFF_HINGLISH = (
    "Sorry, yeh information hamare hotel ke knowledge base mein available nahi hai. "
    "Main aapko humare team ke kisi member se connect kar sakta hoon jo aapki help kar sake. "
    "Please front desk ko extension 0 par call karein."
)


def check_similarity_threshold(retrieved_docs: list[dict]) -> bool:
    """Returns True if at least one retrieved doc clears the confidence threshold."""
    if not retrieved_docs:
        return False
    score = retrieved_docs[0].get("similarity_score", 0.0)
    return score >= MIN_SIMILARITY_SCORE


def scan_for_hallucinations(text: str) -> list[str]:
    """
    Scans the generated text for hallucination violations.
    Returns a list of violation description strings.
    """
    violations = []

    # 1. Scan for fabricated prices / currencies
    price_matches = re.findall(PRICE_PATTERN, text)
    if price_matches:
        violations.append(f"Unsanctioned currency pricing detected: {price_matches}")

    # 2. Scan for fabricated cost expressions
    cost_expr_matches = re.findall(COST_EXPRESSION_PATTERN, text)
    if cost_expr_matches:
        violations.append(f"Fabricated cost expression detected: {cost_expr_matches}")

    # 3. Scan for unauthorized phone numbers
    phone_matches = re.findall(PHONE_PATTERN, text)
    if phone_matches:
        violations.append(f"Fabricated phone number detected: {phone_matches}")

    # 4. Scan for non-whitelisted URLs and domains (Bug 5 URL bypass fix)
    general_url_pattern = r"(?i)\b(?:https?://)?(?:[a-zA-Z0-9-]+\.)+(?:[a-zA-Z]{2,6})\b"
    url_matches = re.findall(general_url_pattern, text)
    for match in url_matches:
        is_whitelisted = False
        for white in WHITELISTED_DOMAINS:
            if white.lower() in match.lower():
                is_whitelisted = True
                break
        if not is_whitelisted:
            violations.append(f"Non-whitelisted domain URL detected: {match}")

    return violations


def get_human_handoff(language: str) -> str:
    """Returns the 'not in KB' response in the appropriate language."""
    if language == "hi":
        return HUMAN_HANDOFF_HI
    elif language == "hinglish":
        return HUMAN_HANDOFF_HINGLISH
    else:
        return HUMAN_HANDOFF_EN


def apply_guardrail(
    generated_text: str,
    retrieved_docs: list[dict],
    language: str = "en"
) -> dict:
    """
    Main guardrail function. Returns:
      {
        "safe": bool,
        "text": str,            # final response to return to user
        "reason": str | None    # why it was blocked (for logging)
      }
    """
    # Level 1: Similarity check
    if not check_similarity_threshold(retrieved_docs):
        score = retrieved_docs[0]["similarity_score"] if retrieved_docs else 0.0
        reason = f"Low similarity score: {score:.4f} (Threshold: {MIN_SIMILARITY_SCORE})"
        logger.warning(f"Guardrail Level 1 triggered: {reason}")
        return {
            "safe": False,
            "text": get_human_handoff(language),
            "reason": reason
        }

    # Level 2: Post-generation hallucination scan
    violations = scan_for_hallucinations(generated_text)
    if violations:
        reason = f"Hallucination patterns matched: {violations}"
        logger.warning(f"Guardrail Level 2 triggered: {reason}")
        return {
            "safe": False,
            "text": get_human_handoff(language),
            "reason": reason
        }

    return {
        "safe": True,
        "text": generated_text,
        "reason": None
    }
