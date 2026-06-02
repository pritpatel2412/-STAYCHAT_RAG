"""
Advanced Hybrid Search Retriever: combines FAISS dense semantic search
with custom local BM25 keyword search using Reciprocal Rank Fusion (RRF).
"""

import os
import pickle
import math
import re
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


class SimpleBM25:
    """Local, lightweight, zero-dependency implementation of BM25 Keyword Search."""
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _initialize(self):
        num_docs = len(self.corpus)
        if num_docs == 0:
            return
            
        total_len = 0
        term_doc_counts = {}
        
        for doc in self.corpus:
            # Combine fields to build search document representation
            text = f"{doc.get('category', '')} {doc.get('title', '')} {doc.get('content', '')}"
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_counts[token] = term_doc_counts.get(token, 0) + 1
                
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            
        self.avg_doc_len = total_len / num_docs
        
        for term, count in term_doc_counts.items():
            self.idf[term] = math.log((num_docs - count + 0.5) / (count + 0.5) + 1.0)

    def score(self, query: str, top_k: int = 4) -> list[tuple[int, float]]:
        """Scores all documents against the query and returns top-K sorted descending."""
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[doc_idx] if doc_idx < len(self.doc_len) else 0
            
            for token in query_tokens:
                if token not in freqs:
                    continue
                
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)
                
                # BM25 tf normalization formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len if self.avg_doc_len > 0 else 1))
                score += idf * (numerator / denominator)
                
            scores.append((doc_idx, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HotelRetriever:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.index = None
        self.kb = []
        self.bm25 = None
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
            # Initialize BM25 index over loaded KB corpus
            self.bm25 = SimpleBM25(self.kb)
            logger.info(f"Loaded Hybrid Retriever successfully. Corpus size: {len(self.kb)} docs.")
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata in retriever: {e}")
            self.index = None
            self.kb = []
            self.bm25 = None

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

    def retrieve_semantic(self, query: str) -> list[dict]:
        """Performs standard FAISS vector search."""
        if self.index is None:
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
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def retrieve_bm25(self, query: str) -> list[dict]:
        """Performs BM25 keyword search."""
        if self.bm25 is None:
            return []
        try:
            matches = self.bm25.score(query, self.top_k)
            results = []
            for doc_idx, score in matches:
                if score <= 0.0:
                    continue
                entry = dict(self.kb[doc_idx])
                entry["bm25_score"] = float(score)
                results.append(entry)
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def retrieve(self, query: str) -> list[dict]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF)
        to combine FAISS semantic matches and BM25 keyword matches.
        """
        # Lazy reloading if index becomes available after startup failure
        if self.index is None:
            self._load_index()
            if self.index is None:
                logger.warning("FAISS index is not built. Skipping retrieval.")
                return []

        # 1. Fetch semantic and sparse matches
        semantic_matches = self.retrieve_semantic(query)
        bm25_matches = self.retrieve_bm25(query)

        # 2. Combine ranks using Reciprocal Rank Fusion (RRF)
        # RRF_Score = Sum( 1.0 / (K + Rank) )
        K = 60
        rrf_scores = {}
        doc_map = {}

        # Load semantic matches and score ranks
        for rank, doc in enumerate(semantic_matches):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))

        # Load BM25 matches and score ranks
        for rank, doc in enumerate(bm25_matches):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (K + rank + 1))

        # Sort by unified RRF score descending
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_rrf[:self.top_k]:
            doc = dict(doc_map[doc_id])
            doc["rrf_score"] = float(rrf_score)
            
            # Ensure similarity_score is preserved for pre-gen guardrails check.
            # If a document only matched via BM25, default its similarity_score to 0.0
            if "similarity_score" not in doc:
                doc["similarity_score"] = 0.0
                
            final_results.append(doc)

        logger.info(
            f"Hybrid retrieval finished for '{query[:40]}...'. "
            f"Semantic hits: {len(semantic_matches)} | BM25 hits: {len(bm25_matches)} | Combined top-K: {len(final_results)}"
        )
        return final_results
