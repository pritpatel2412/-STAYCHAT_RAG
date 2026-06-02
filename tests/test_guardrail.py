import pytest
from src.guardrail import scan_for_hallucinations, check_similarity_threshold, get_human_handoff


def test_scan_for_hallucinations_valid_text():
    # Standard, safe texts should return zero violations
    violations = scan_for_hallucinations("Our check-in time is standard 2:00 PM and check-out is 12:00 noon.")
    assert len(violations) == 0

    violations = scan_for_hallucinations("Please call the front desk at extension 0 for room service.")
    assert len(violations) == 0


def test_scan_for_hallucinations_price_trap_currency_symbols():
    # Verify Bug 4 fix: Special currency symbols without word boundaries are captured!
    violations_usd = scan_for_hallucinations("The Deluxe Room costs $100 per night.")
    assert len(violations_usd) > 0
    assert any("currency pricing" in v.lower() for v in violations_usd)

    violations_eur = scan_for_hallucinations("Incidental charges can be settled up to €50.")
    assert len(violations_eur) > 0

    violations_gbp = scan_for_hallucinations("A deposit of £200 is required.")
    assert len(violations_gbp) > 0


def test_scan_for_hallucinations_price_trap_cost_phrases():
    # Verify Bug 3 fix: The bare phrase "the price is" without actual numbers should NOT violate guardrails!
    violations_safe = scan_for_hallucinations("I'm sorry, the price is not listed in my knowledge base.")
    assert len(violations_safe) == 0

    # But if there are numbers, it should violate!
    violations_unsafe = scan_for_hallucinations("The price is Rs. 15000.")
    assert len(violations_unsafe) > 0
    assert any("cost expression" in v.lower() for v in violations_unsafe)


def test_scan_for_hallucinations_unauthorized_urls():
    # Verify Bug 5 URL bypass fix: Custom/fabricated domains are flagged, even without http scheme
    violations = scan_for_hallucinations("To complete your booking, visit fake-grandhotel.net/pay.")
    assert len(violations) > 0
    assert any("whitelisted domain" in v.lower() for v in violations)

    # Whitelisted domains should be allowed
    violations_white = scan_for_hallucinations("You can search on google.com or visit our main site grandhotel.com/rewards.")
    assert len(violations_white) == 0


def test_scan_for_hallucinations_phone_numbers():
    # Authorized number in KB: +91-22-6600-0000
    violations_auth = scan_for_hallucinations("Call the emergency desk at +91-22-6600-0000.")
    assert len(violations_auth) == 0

    # Fabricated 10-digit number should trigger violation
    violations_fake = scan_for_hallucinations("Call reservations at 9876543210.")
    assert len(violations_fake) > 0


def test_similarity_threshold():
    # Mock documents
    doc_pass = [{"id": "1", "similarity_score": 0.55}]
    doc_fail = [{"id": "2", "similarity_score": 0.35}]
    doc_empty = []

    assert check_similarity_threshold(doc_pass) is True
    assert check_similarity_threshold(doc_fail) is False
    assert check_similarity_threshold(doc_empty) is False


def test_human_handoff_languages():
    assert "मुझे खेद है" in get_human_handoff("hi")
    assert "hamare hotel ke knowledge base" in get_human_handoff("hinglish")
    assert "hotel knowledge base" in get_human_handoff("en")
