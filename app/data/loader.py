"""LLM-Redial Movie dataset loader and parser.

Handles loading the actual LLM-Redial dataset files:
- item_map.json    → ASIN-to-movie-title mapping
- final_data.jsonl → Per-user structured data (history, conversations)
- Conversation.txt → Full conversation dialogues (User/Agent turns)
- user_ids.json    → User-ID-to-index mapping

Reference: Tools.py and read_me.py provided with the dataset.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Item map (ASIN → movie title) ──────────────────────────────────


def load_item_map(data_dir: Path | None = None) -> dict[str, str]:
    """Load item_map.json that maps ASINs to movie titles.

    Returns:
        Dict mapping ASIN strings to human-readable movie titles.
    """
    path = _resolve_data_path(data_dir) / "item_map.json"
    if not path.exists():
        logger.warning("item_map.json not found at %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        item_map: dict[str, str] = json.load(f)

    logger.info("Loaded item_map with %d movies", len(item_map))
    return item_map


# ── Final data (per-user records) ──────────────────────────────────


def load_final_data(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load final_data.jsonl — one JSON object per line, keyed by user ID.

    Each line: {"USER_ID": {"history_interaction": [...], "user_might_like": [...],
                             "Conversation": [...]}}

    Returns:
        List of parsed JSON objects (one per user).
    """
    path = _resolve_data_path(data_dir) / "final_data.jsonl"
    if not path.exists():
        logger.warning("final_data.jsonl not found at %s", path)
        return []

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line")

    logger.info("Loaded %d user records from final_data.jsonl", len(records))
    return records


# ── Conversation text ──────────────────────────────────────────────


def load_conversations(data_dir: Path | None = None) -> dict[int, str]:
    """Load Conversation.txt — dialogues indexed by conversation_id.

    Format: conversation_id on its own line (a digit), followed by
    User: / Agent: turn pairs, separated by blank lines.

    Returns:
        Dict mapping conversation_id (int) → full dialogue text.
    """
    path = _resolve_data_path(data_dir) / "Conversation.txt"
    if not path.exists():
        logger.warning("Conversation.txt not found at %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    dialogues: dict[int, str] = {}
    current_id: int | None = None
    current_text: list[str] = []

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.isdigit() and not line.startswith(" "):
            # Save previous conversation
            if current_id is not None:
                dialogues[current_id] = "\n".join(current_text).strip()
            current_id = int(stripped)
            current_text = []
        else:
            current_text.append(line)

    # Save last conversation
    if current_id is not None:
        dialogues[current_id] = "\n".join(current_text).strip()

    logger.info("Loaded %d conversations from Conversation.txt", len(dialogues))
    return dialogues


# ── Movie extraction (for vector store) ────────────────────────────


def extract_movies(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Extract all unique movies from item_map + final_data.

    Combines item_map titles with usage context from final_data
    (how many users interacted, liked, disliked, etc.).

    Returns:
        List of movie dicts: {id, title, description}.
    """
    item_map = load_item_map(data_dir)
    if not item_map:
        return []

    records = load_final_data(data_dir)

    # Count how many users interacted/liked/disliked each item
    interaction_count: dict[str, int] = {}
    like_count: dict[str, int] = {}
    rec_count: dict[str, int] = {}

    for record in records:
        _user_id, user_info = next(iter(record.items()))
        for asin in user_info.get("history_interaction", []):
            interaction_count[asin] = interaction_count.get(asin, 0) + 1
        for asin in user_info.get("user_might_like", []):
            rec_count[asin] = rec_count.get(asin, 0) + 1
        for conv in user_info.get("Conversation", []):
            conv_data = next(iter(conv.values()))
            for asin in conv_data.get("user_likes", []):
                like_count[asin] = like_count.get(asin, 0) + 1

    # Build movie list
    movies: list[dict[str, Any]] = []
    for asin, title in item_map.items():
        # Parse year from title if present (e.g., "Movie Name (2010)")
        year_match = re.search(r"\((\d{4})\)", title)
        year = year_match.group(1) if year_match else None

        clean_title = re.sub(r"\s*(VHS|DVD|Blu-ray|Region \d).*$", "", title, flags=re.IGNORECASE).strip()

        interactions = interaction_count.get(asin, 0)
        likes = like_count.get(asin, 0)
        recs = rec_count.get(asin, 0)

        desc = f"{clean_title}."
        if year:
            desc += f" Released in {year}."
        if interactions > 0:
            desc += f" Watched by {interactions} users."
        if likes > 0:
            desc += f" Liked by {likes} users."
        if recs > 0:
            desc += f" Recommended {recs} times."

        movies.append({
            "id": asin,
            "title": clean_title,
            "year": year,
            "genres": [],  # Not in dataset; LLM will infer
            "description": desc,
        })

    logger.info("Extracted %d movies for vector store", len(movies))
    return movies


# ── Few-shot examples (from real dialogues) ────────────────────────


def extract_few_shot_conversations(
    data_dir: Path | None = None,
    max_examples: int = 50,
) -> list[dict[str, str]]:
    """Extract real User/Agent conversation pairs for few-shot prompting.

    Reads Conversation.txt dialogues and extracts User→Agent pairs
    where the Agent actually recommends a movie.

    Returns:
        List of dicts: {user_message, assistant_response}.
    """
    dialogues = load_conversations(data_dir)
    examples: list[dict[str, str]] = []

    for _conv_id, text in sorted(dialogues.items()):
        turns = _parse_turns(text)
        for i in range(len(turns) - 1):
            role_a, text_a = turns[i]
            role_b, text_b = turns[i + 1]

            if role_a == "user" and role_b == "agent" and len(text_b) > 50:
                examples.append({
                    "user_message": text_a,
                    "assistant_response": text_b,
                })
                if len(examples) >= max_examples:
                    return examples

    logger.info("Extracted %d few-shot examples from dialogues", len(examples))
    return examples


def _parse_turns(text: str) -> list[tuple[str, str]]:
    """Parse a dialogue text into (role, content) turn pairs."""
    turns: list[tuple[str, str]] = []
    current_role = ""
    current_text: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("User:"):
            if current_role:
                turns.append((current_role, " ".join(current_text).strip()))
            current_role = "user"
            current_text = [line[len("User:"):].strip()]
        elif line.startswith("Agent:"):
            if current_role:
                turns.append((current_role, " ".join(current_text).strip()))
            current_role = "agent"
            current_text = [line[len("Agent:"):].strip()]
        else:
            current_text.append(line)

    if current_role and current_text:
        turns.append((current_role, " ".join(current_text).strip()))

    return turns


# ── Helpers ────────────────────────────────────────────────────────


def _resolve_data_path(data_dir: Path | None = None) -> Path:
    """Resolve the data directory, auto-detecting the Movie/ subdirectory."""
    settings = get_settings()
    base = Path(data_dir) if data_dir else settings.data_path

    # Auto-detect Movie/ subdirectory
    movie_subdir = base / "Movie"
    if movie_subdir.exists():
        return movie_subdir

    return base
