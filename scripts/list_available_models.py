"""Utility script to list all models available in the configured Gemini API key account."""
import os
import sys
from dotenv import load_dotenv

# Ensure root dir is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_gemini_api_key_here":
    logger.error("GEMINI_API_KEY is not set or is still the default placeholder. Please configure it in your .env file.")
    sys.exit(1)

try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    logger.info("Connecting to Gemini API...")
    models = genai.list_models()
    
    print("\n" + "="*70)
    print("                    AVAILABLE GEMINI MODELS")
    print("="*70)
    
    print(f"{'Model Name':<35} | {'Supported Methods'}")
    print("-"*70)
    
    for m in models:
        methods = ", ".join(m.supported_generation_methods)
        print(f"{m.name:<35} | {methods}")
        
    print("="*70 + "\n")
    
except Exception as e:
    logger.error(f"Failed to query Gemini models list: {e}")
