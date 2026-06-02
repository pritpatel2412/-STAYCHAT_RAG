import pytest
import os
from src.retriever import HotelRetriever


def test_retriever_initialization():
    # Verify the retriever can initialize even if index is not built yet
    # It should log warning and not crash
    retriever = HotelRetriever(top_k=3)
    assert retriever.top_k == 3
    assert hasattr(retriever, "retrieve")


def test_retrieve_empty_when_no_index(monkeypatch):
    # Force index to be None to simulate missing index file
    retriever = HotelRetriever(top_k=2)
    monkeypatch.setattr(retriever, "index", None)
    monkeypatch.setattr(retriever, "_load_index", lambda: None)
    
    results = retriever.retrieve("Any hotel query")
    assert isinstance(results, list)
    assert len(results) == 0
