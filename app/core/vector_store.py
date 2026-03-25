"""ChromaDB vector store operations for movie knowledge base.

Provides initialization, embedding storage, and semantic similarity
search over the movie catalog.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singleton
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Get or create the OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client with persistence."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialized at %s", settings.chroma_path)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the movies collection."""
    global _collection
    if _collection is None:
        settings = get_settings()
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d documents)",
            settings.chroma_collection_name,
            _collection.count(),
        )
    return _collection


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a text string.

    Args:
        text: The text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    settings = get_settings()
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def generate_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors.
    """
    settings = get_settings()
    client = get_openai_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info("Embedded batch %d-%d / %d", i, i + len(batch), len(texts))

    return all_embeddings


def add_movies(movies: list[dict[str, Any]]) -> int:
    """Add movies to the ChromaDB collection.

    Each movie's description + genre text is embedded and stored
    with full metadata for filtering.

    Args:
        movies: List of movie dicts with title, genres, year, description.

    Returns:
        Number of movies successfully added.
    """
    collection = get_collection()

    # Prepare texts for embedding
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str]] = []

    for i, movie in enumerate(movies):
        title = movie.get("title", "Unknown").strip()
        genres = movie.get("genres", [])
        year = movie.get("year", "")
        description = movie.get("description", "")

        # Build rich text for embedding
        genre_str = ", ".join(genres) if genres else ""
        embed_text = f"{title}. {genre_str}. {description}".strip()

        doc_id = f"movie_{i}_{title[:50].replace(' ', '_').lower()}"
        ids.append(doc_id)
        texts.append(embed_text)
        metadatas.append({
            "title": title,
            "genres": ", ".join(genres) if genres else "",
            "year": str(year) if year else "",
            "description": description[:500],
        })

    # Generate embeddings in batches
    logger.info("Generating embeddings for %d movies...", len(texts))
    embeddings = generate_embeddings_batch(texts)

    # Add to ChromaDB in batches
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("Added batch %d-%d to ChromaDB", i, end)

    logger.info("Successfully added %d movies to vector store", len(ids))
    return len(ids)


def search_movies(
    query: str,
    n_results: int = 10,
    genre_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search for movies similar to a query.

    Args:
        query: Natural language search query.
        n_results: Number of results to return.
        genre_filter: Optional genre to filter by.

    Returns:
        List of movie result dicts with title, genres, year, description, score.
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Vector store is empty — no movies to search")
        return []

    query_embedding = generate_embedding(query)

    where_filter = None
    if genre_filter:
        where_filter = {"genres": {"$contains": genre_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    movies: list[dict[str, Any]] = []
    if results and results["metadatas"]:
        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            movies.append({
                "title": metadata.get("title", "Unknown"),
                "genres": metadata.get("genres", "").split(", "),
                "year": metadata.get("year", ""),
                "description": metadata.get("description", ""),
                "similarity_score": round(1 - distance, 4),
            })

    return movies


def get_movie_count() -> int:
    """Get the number of movies in the vector store."""
    try:
        return get_collection().count()
    except Exception:
        return 0


def is_ready() -> bool:
    """Check if the vector store is initialized and has data."""
    try:
        return get_collection().count() > 0
    except Exception:
        return False
