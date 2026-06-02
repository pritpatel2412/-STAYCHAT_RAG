"""
Ingestion pipeline: loads hotel KB, generates embeddings via Google Gemini in a single
optimized batch call, and builds a FAISS index. Run once before starting the server.
"""

import json
import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY is not set. Please set it in your .env file before running.")
else:
    genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/gemini-embedding-2"
INDEX_PATH = "faiss_index/hotel.index"
META_PATH  = "faiss_index/hotel_meta.pkl"
KB_PATH    = "data/hotel_kb.json"


def embed_texts_batched(texts: list[str]) -> np.ndarray:
    """Embed a list of texts in a single batched call using Gemini embeddings. Returns (N, D) float32 array."""
    logger.info(f"Sending batch request to embed {len(texts)} texts...")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="retrieval_document"
        )
        # result['embedding'] is a list of lists of floats
        embeddings = result["embedding"]
        return np.array(embeddings, dtype="float32")
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise e


def build_index():
    os.makedirs("faiss_index", exist_ok=True)

    if not os.path.exists(KB_PATH):
        logger.error(f"Knowledge base file not found at: {KB_PATH}")
        return

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse KB file {KB_PATH}: {e}")
        return

    # Build rich text representations for embedding
    texts = [
        f"Category: {item['category']}\nTitle: {item['title']}\n{item['content']}"
        for item in kb
    ]

    logger.info(f"Loaded {len(texts)} KB entries. Starting batch embedding generation...")
    try:
        embeddings = embed_texts_batched(texts)
    except Exception as e:
        logger.error(f"Aborting build_index due to embedding failure: {e}")
        return

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat inner-product index (equivalent to cosine after L2 normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    try:
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(kb, f)
    except Exception as e:
        logger.error(f"Failed to save FAISS index or metadata: {e}")
        return

    logger.info(f"Successfully built FAISS index with {index.ntotal} vectors.")
    logger.info(f"FAISS index saved to: {INDEX_PATH}")
    logger.info(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    build_index()
