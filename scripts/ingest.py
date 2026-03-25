"""Data ingestion script — embeds movies into ChromaDB.

Run this once after placing the LLM-Redial Movie dataset files
in the ./data/movie/ directory:

    python -m scripts.ingest

This will:
1. Load item_map.json + final_data.jsonl from ./data/movie/Movie/
2. Extract all unique movies with usage statistics
3. Generate OpenAI embeddings
4. Store in ChromaDB for semantic search
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.vector_store import add_movies, get_movie_count
from app.data.loader import extract_movies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main ingestion pipeline."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("LLM-Redial Movie Data Ingestion")
    logger.info("=" * 60)
    logger.info("Data directory: %s", settings.data_path)
    logger.info("ChromaDB path:  %s", settings.chroma_path)
    logger.info("Embedding model: %s", settings.openai_embedding_model)

    # Check if data exists — auto-detect Movie/ subdirectory
    data_path = settings.data_path
    movie_subdir = data_path / "Movie"
    actual_path = movie_subdir if movie_subdir.exists() else data_path

    required_files = ["item_map.json", "final_data.jsonl"]
    missing = [f for f in required_files if not (actual_path / f).exists()]
    if missing:
        logger.error(
            "Missing data files: %s\n"
            "Expected in: %s\n"
            "Please download the LLM-Redial Movie dataset from:\n"
            "https://drive.google.com/drive/folders/"
            "1QxMxBgW8OpcVRchwbY45sFjeLQuPwuZf\n"
            "and place the Movie category files there.",
            ", ".join(missing),
            actual_path,
        )
        sys.exit(1)

    logger.info("Dataset found at: %s", actual_path)

    # Check existing count
    existing_count = get_movie_count()
    if existing_count > 0:
        logger.info("Vector store already contains %d movies", existing_count)
        response = input("Re-ingest? This will add duplicates. (y/N): ").strip().lower()
        if response != "y":
            logger.info("Skipping ingestion. Existing data retained.")
            return

    # Extract movies
    logger.info("Extracting movies from dataset...")
    movies = extract_movies()
    logger.info("Found %d unique movies", len(movies))

    if not movies:
        logger.error("No movies extracted from dataset")
        sys.exit(1)

    # Log sample
    logger.info("\nSample movies:")
    for movie in movies[:5]:
        logger.info("  - [%s] %s", movie["id"], movie["title"])

    # Embed and store
    logger.info("\nEmbedding and storing %d movies...", len(movies))
    count = add_movies(movies)
    logger.info("=" * 60)
    logger.info("✓ Successfully ingested %d movies into ChromaDB", count)
    logger.info("  Vector store path: %s", settings.chroma_path)
    logger.info("  Total movies in store: %d", get_movie_count())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
