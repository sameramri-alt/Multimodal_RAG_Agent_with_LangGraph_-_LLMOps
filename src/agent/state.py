"""
src/agent/state.py — Agent State Definition
============================================
In LangGraph, the "State" is the shared memory of the agent.
It is a typed dictionary (TypedDict) that flows through all nodes of the graph.

Think of it as a notebook the agent keeps open while working:
each node reads from it, writes to it, and passes it to the next node.
"""

from typing import TypedDict, List


class AgentState(TypedDict):
    """
    Defines the complete memory structure of the conversational agent.
    Every field below is available to every node (search, evaluate, rewrite, generate).
    """

    # The current user's question (may be rewritten by the rewriter node)
    query: str

    # Rolling conversation history (last N exchanges) as a plain string.
    # Used by the translator and rewriter nodes to resolve ambiguous follow-up
    # questions like "and John?" or "what about him?" without losing context.
    chat_history: str

    # How many search attempts have been made for the current question.
    # Capped at 2 to prevent infinite retry loops.
    search_attempts: int

    # Retrieved documents from Qdrant (text segments + OCR-enriched slide payloads).
    # Each item is a dict with keys: type, source, text, start, end.
    contexts: List[dict]

    # Boolean flag set by the evaluator node.
    # True  → the retrieved context is relevant → go to generator.
    # False → context is off-topic or empty     → go to rewriter (if attempts < 2).
    is_confident: bool

    # The final answer written by the generator node, ready to be printed to the user.
    final_response: str
