"""
Lightweight, zero-dependency Groq client to provide ultra-fast LLM failover.
Directly invokes Groq's OpenAI-compatible completions endpoint.
"""

import os
import requests
from src.logger import logger


class GroqResponseMock:
    """Mock structure to mimic the Google Gemini response schema."""
    def __init__(self, text: str):
        self.text = text


class GroqClient:
    def __init__(self, default_model: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = default_model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # Validate API Key
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            logger.warning(
                "GROQ_API_KEY is not configured or is the default placeholder. "
                "Groq failover will be offline."
            )
            self.api_key = None

    def generate_content(self, prompt: str, system_instruction: str = None) -> GroqResponseMock:
        """
        Executes a completion request, mimicking the Gemini generate_content interface.
        """
        if not self.api_key:
            raise ValueError("Groq API Key is unconfigured. Cannot execute failover request.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024
        }

        logger.info(f"Routing failover query to Groq using model '{self.model}'...")
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                logger.info("Successfully received high-speed completion from Groq.")
                return GroqResponseMock(text)
            else:
                error_detail = resp.text
                logger.error(f"Groq API returned error status {resp.status_code}: {error_detail}")
                raise RuntimeError(f"Groq failure: {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to communicate with Groq API endpoint: {e}")
            raise e
