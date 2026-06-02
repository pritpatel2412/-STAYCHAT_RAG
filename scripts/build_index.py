"""One-time script to build FAISS index from hotel KB."""
import sys
import os

# Ensure the root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import build_index
from src.logger import logger

if __name__ == "__main__":
    logger.info("Initializing vector index build script...")
    build_index()
