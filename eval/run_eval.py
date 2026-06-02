"""
Runs the evaluation set and prints a structured report.
Checks: correct intent, correct language detection, guardrail behavior, no hallucination.
"""

import json
import sys
import os

# Ensure the root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import RAGPipeline
from src.guardrail import scan_for_hallucinations
from src.logger import logger

def run_evaluation():
    # Force UTF-8 console output to prevent CP1252 encoding crashes on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logger.info("Initializing RAG Pipeline for automated evaluation evaluation...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize evaluation runner: {e}. Ensure FAISS index is built and GEMINI_API_KEY is configured.")
        return

    eval_file = "eval/eval_questions.json"
    if not os.path.exists(eval_file):
        logger.error(f"Evaluation questions file not found: {eval_file}")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("\n" + "="*80)
    print("                      HOTEL RAG BOT — EVALUATION REPORT")
    print("="*80)

    passed = 0
    failed = 0

    for q in questions:
        print(f"\n[*] Testing ID: {q['id']}")
        print(f"   Question   : {q['question']}")
        print(f"   Category   : {q['notes']}")

        try:
            # Execute pipeline without conversation history
            result = pipeline.run(q["question"], [])
        except Exception as e:
            print(f"   [ERROR] PIPELINE ERROR: {e}")
            failed += 1
            continue

        # 1. Verify intent matching
        intent_ok = result["intent"] == q["expected_intent"]
        
        # 2. Verify language matching
        language_ok = result["language"] == q["expected_language"]
        
        # 3. Scan generated response for hallucination violations
        hallucination = scan_for_hallucinations(result["response"])
        no_hallucination = len(hallucination) == 0

        # 4. Check guardrail triggering:
        # For TRAP questions (should_answer = False):
        # The guardrail MUST trigger OR the response must contain off-limits warning words.
        if not q["should_answer"]:
            guardrail_ok = (
                result["guardrail_triggered"] or
                any(phrase in result["response"].lower() for phrase in [
                    "not in", "don't have", "nahi hai", "unavailable",
                    "connect you", "human", "front desk", "team member", "apologize"
                ])
            )
        else:
            # For normal questions, we expect NO guardrail trigger
            guardrail_ok = not result["guardrail_triggered"]

        overall_pass = intent_ok and language_ok and guardrail_ok and no_hallucination

        if overall_pass:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"   Status     : {status}")
        print(f"   Intent     : {result['intent']} (Expected: {q['expected_intent']}) {'[OK]' if intent_ok else '[FAIL]'}")
        print(f"   Language   : {result['language']} (Expected: {q['expected_language']}) {'[OK]' if language_ok else '[FAIL]'}")
        print(f"   Guardrail  : Triggered={result['guardrail_triggered']} | OK={guardrail_ok} {'[OK]' if guardrail_ok else '[FAIL]'}")
        print(f"   Hallucinate: {hallucination if hallucination else 'None detected'} {'[OK]' if no_hallucination else '[FAIL]'}")
        print(f"   Response   : {result['response'][:150]}...")

        # Pace requests to avoid free-tier 429 rate limits (5 RPM)
        import time
        if q != questions[-1]:
            logger.info("Pacing requests (sleeping 12 seconds) to respect free-tier RPM limits...")
            time.sleep(12)

    print("\n" + "="*80)
    print(f"FINAL RESULT: {passed}/{len(questions)} passed | {failed}/{len(questions)} failed")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_evaluation()
