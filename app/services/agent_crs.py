"""Multi-Agent Conversational Recommender System using LangGraph.

Implements Approach #2: A graph of 3 specialized agents that cooperate:
1. Preference Analyzer → extracts user tastes from conversation
2. Movie Retriever + Recommender → searches DB & ranks results
3. Explainer → crafts a personalized, engaging response

Uses LangGraph for stateful orchestration with typed state management.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, TypedDict

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.prompts import (
    PREFERENCE_ANALYZER_PROMPT,
    RECOMMENDER_PROMPT,
    EXPLAINER_PROMPT,
    format_history,
    format_retrieved_movies,
)
from app.core.vector_store import search_movies
from app.models import Message
from app.services.base_crs import BaseCRS

logger = logging.getLogger(__name__)


# ── Agent State Definition ──────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Shared state passed between agents in the graph."""

    user_message: str
    history: list[dict[str, str]]
    preferences: dict[str, Any]
    retrieved_movies: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    final_response: str
    error: str | None


# ── Agent Node Functions ────────────────────────────────────────────

async def _call_llm(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    parse_json: bool = False,
) -> str:
    """Helper to call the LLM synchronously (non-streaming).

    Args:
        client: AsyncOpenAI client instance.
        model: Model name to use.
        system_prompt: System message.
        user_prompt: User message.
        parse_json: If True, request JSON response format.

    Returns:
        The LLM's response text.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    if parse_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def analyze_preferences(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
) -> AgentState:
    """Agent 1: Analyze user preferences from conversation.

    Extracts genres, themes, moods, and builds a refined search query.
    """
    logger.info("Agent[Analyzer]: Extracting user preferences...")

    history_str = format_history(state.get("history", []))
    prompt = PREFERENCE_ANALYZER_PROMPT.format(
        history=history_str,
        user_message=state["user_message"],
    )

    response = await _call_llm(
        client=client,
        model=model,
        system_prompt="You are a preference analysis agent. Output valid JSON only.",
        user_prompt=prompt,
        parse_json=True,
    )

    try:
        preferences = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Agent[Analyzer]: Failed to parse JSON, using defaults")
        preferences = {
            "liked_genres": [],
            "liked_themes": [],
            "liked_movies": [],
            "disliked": [],
            "mood": "looking for movie recommendations",
            "search_query": state["user_message"],
        }

    logger.info("Agent[Analyzer]: Preferences → %s", json.dumps(preferences, indent=2))
    state["preferences"] = preferences
    return state


async def retrieve_and_recommend(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
) -> AgentState:
    """Agent 2: Retrieve movies and rank recommendations.

    Uses the refined search query from the analyzer to search ChromaDB,
    then asks the LLM to rank and select the best matches.
    """
    preferences = state.get("preferences", {})
    search_query = preferences.get("search_query", state["user_message"])

    # Retrieve from vector store
    logger.info("Agent[Retriever]: Searching for: %s", search_query)
    retrieved = search_movies(query=search_query, n_results=15)
    state["retrieved_movies"] = retrieved
    logger.info("Agent[Retriever]: Found %d candidates", len(retrieved))

    # Ask LLM to rank and select top recommendations
    history_str = format_history(state.get("history", []))
    movies_str = format_retrieved_movies(retrieved)

    prompt = RECOMMENDER_PROMPT.replace("{preferences}", json.dumps(preferences))
    prompt = prompt.replace("{retrieved_movies}", movies_str)
    prompt = prompt.replace("{history}", history_str)
    prompt = prompt.replace("{user_message}", state["user_message"])

    response = await _call_llm(
        client=client,
        model=model,
        system_prompt="You are a movie ranking agent. Output valid JSON array only.",
        user_prompt=prompt,
        parse_json=True,
    )

    try:
        # Handle both array and object responses
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "recommendations" in parsed:
            recommendations = parsed["recommendations"]
        elif isinstance(parsed, list):
            recommendations = parsed
        else:
            recommendations = [parsed]
    except json.JSONDecodeError:
        logger.warning("Agent[Recommender]: Failed to parse JSON, using raw response")
        recommendations = [{"title": "Error", "reason": response[:200]}]

    logger.info("Agent[Recommender]: Selected %d recommendations", len(recommendations))
    state["recommendations"] = recommendations
    return state


async def explain_recommendations(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
) -> AgentState:
    """Agent 3: Generate a warm, engaging explanation of recommendations.

    Takes the ranked recommendations and crafts a natural conversational
    response with enthusiasm and personality.
    """
    history_str = format_history(state.get("history", []))
    recommendations_str = json.dumps(state.get("recommendations", []), indent=2)

    prompt = EXPLAINER_PROMPT.replace("{user_message}", state["user_message"])
    prompt = prompt.replace("{history}", history_str)
    prompt = prompt.replace("{recommendations}", recommendations_str)

    logger.info("Agent[Explainer]: Crafting response...")
    response = await _call_llm(
        client=client,
        model=model,
        system_prompt="You are a warm, enthusiastic movie recommender. Be conversational and engaging.",
        user_prompt=prompt,
    )

    state["final_response"] = response
    return state


# ── Multi-Agent CRS Class ───────────────────────────────────────────

class AgentCRS(BaseCRS):
    """Multi-Agent Conversational Recommender System.

    Uses LangGraph to orchestrate 3 specialized agents:
    1. Preference Analyzer: Extracts user preferences
    2. Retriever + Recommender: Searches DB and ranks movies
    3. Explainer: Generates engaging response

    The response is streamed character-by-character after the
    full agent pipeline completes (agent processing is sequential,
    but the final output is streamed for SSE compatibility).
    """

    def __init__(self) -> None:
        """Initialize the Multi-Agent CRS with LangGraph."""
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._graph = self._build_graph()

    @property
    def model_id(self) -> str:
        return "agent"

    @property
    def model_name(self) -> str:
        return "Multi-Agent Recommender"

    @property
    def model_description(self) -> str:
        return (
            "Multi-Agent system using LangGraph. Three specialized agents "
            "cooperate: Preference Analyzer → Movie Retriever & Recommender "
            "→ Explainer. Produces more structured, deeply personalized "
            "recommendations through step-by-step reasoning."
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow.

        Graph structure:
        START → analyzer → retriever_recommender → explainer → END
        """
        graph = StateGraph(AgentState)

        # Add nodes (wrapped to inject client + model)
        graph.add_node("analyzer", self._run_analyzer)
        graph.add_node("retriever_recommender", self._run_retriever_recommender)
        graph.add_node("explainer", self._run_explainer)

        # Define edges
        graph.set_entry_point("analyzer")
        graph.add_edge("analyzer", "retriever_recommender")
        graph.add_edge("retriever_recommender", "explainer")
        graph.add_edge("explainer", END)

        return graph

    async def _run_analyzer(self, state: AgentState) -> AgentState:
        """Wrapper for the preference analyzer agent."""
        return await analyze_preferences(
            state, self._client, self._settings.openai_model,
        )

    async def _run_retriever_recommender(self, state: AgentState) -> AgentState:
        """Wrapper for the retriever + recommender agent."""
        return await retrieve_and_recommend(
            state, self._client, self._settings.openai_model,
        )

    async def _run_explainer(self, state: AgentState) -> AgentState:
        """Wrapper for the explainer agent."""
        return await explain_recommendations(
            state, self._client, self._settings.openai_model,
        )

    async def recommend(
        self,
        message: str,
        history: list[Message],
    ) -> AsyncGenerator[str, None]:
        """Generate streaming movie recommendations using multi-agent pipeline.

        The agent pipeline runs to completion, then the final response
        is streamed token-by-token for SSE compatibility.

        Args:
            message: The user's current message.
            history: Previous conversation messages.

        Yields:
            Response text chunks for SSE streaming.
        """
        # Prepare initial state
        initial_state: AgentState = {
            "user_message": message,
            "history": [{"role": m.role.value, "content": m.content} for m in history],
            "preferences": {},
            "retrieved_movies": [],
            "recommendations": [],
            "final_response": "",
            "error": None,
        }

        # Compile and run the graph
        logger.info("Agent CRS: Running multi-agent pipeline...")
        compiled = self._graph.compile()

        try:
            result = await compiled.ainvoke(initial_state)
            final_response = result.get("final_response", "")

            if not final_response:
                final_response = (
                    "I apologize, but I had trouble processing your request. "
                    "Could you try rephrasing what kind of movie you're looking for?"
                )

            # Stream the final response word-by-word for natural SSE delivery
            words = final_response.split(" ")
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield word + " "
                else:
                    yield word

        except Exception as e:
            logger.error("Agent CRS pipeline failed: %s", str(e), exc_info=True)
            yield f"I encountered an error while processing your request: {str(e)}"
