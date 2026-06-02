"""
FAISS-based semantic retriever. ALWAYS uses real vector search —
never passes the full KB into the prompt.
"""

import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/gemini-embedding-2"
INDEX_PATH = "faiss_index/hotel.index"
META_PATH  = "faiss_index/hotel_meta.pkl"


class HotelRetriever:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.index = None
        self.kb = []
        self._load_index()

    def _load_index(self):
        """Load index and metadata from disk, handling missing files gracefully."""
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            logger.warning(
                f"FAISS index or metadata files missing. Please run 'python scripts/build_index.py' "
                f"to build them. Retriever will return empty lists until built."
            )
            return

        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.kb = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} documents successfully.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata: {e}")
            self.index = None
            self.kb = []

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            vec = np.array([result["embedding"]], dtype="float32")
            faiss.normalize_L2(vec)
            return vec
        except Exception as e:
            logger.error(f"Embedding query failed: {e}")
            raise e

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-K KB entries most relevant to the query.
        Includes similarity score for downstream confidence checks.
        """
        # Lazy reloading if index becomes available after initial load fail
        if self.index is None:
            self._load_index()
            if self.index is None:
                logger.warning("FAISS index is not built. Skipping retrieval.")
                return []

        try:
            query_vec = self._embed_query(query)
            scores, indices = self.index.search(query_vec, self.top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.kb):
                    continue
                entry = dict(self.kb[idx])
                entry["similarity_score"] = float(score)
                results.append(entry)

            logger.info(f"Retrieved {len(results)} relevant documents for query: '{query[:40]}...'")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
