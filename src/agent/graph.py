"""
src/agent/graph.py — LangGraph Workflow Definition
===================================================
This file acts as the "Architect" of our conversational agent.
It stitches together the different nodes (Search, Evaluate, Rewrite, Generate)
using LangGraph, which allows us to create stateful, cyclic AI workflows.
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import search_node, evaluator_node, rewriter_node, generator_node

def should_generate(state: AgentState):
    """
    Conditional edge function (The "Router").
    It reads the agent's state after the evaluation step and decides which node to execute next.
    """
    # If the LLM is confident (it found the answer in the retrieved context)
    if state.get("is_confident"):
        print("-> Router: Confidence OK, proceeding to generate answer.")
        return "generate"
        
    # If not confident, but we've already tried 2 times, force generation.
    # This prevents infinite loops where the agent keeps rewriting the query forever.
    elif state.get("search_attempts", 0) >= 2:
        print("-> Router: Confidence KO but max attempts (2) reached. Generating with best available context.")
        return "generate"
        
    # If not confident and we still have attempts left, rewrite the query and search again.
    else:
        print("-> Router: Confidence KO, rewriting query for better retrieval.")
        return "rewrite"

def build_agent():
    """
    Assembles the Lego blocks (nodes) to create the final agent pipeline.
    Defines the entry point and the exact execution order.
    """
    
    # 1. Initialize the StateGraph with our custom memory structure (AgentState)
    workflow = StateGraph(AgentState)
    
    # 2. Register all available nodes
    workflow.add_node("search", search_node)
    workflow.add_node("evaluate", evaluator_node)
    workflow.add_node("rewrite", rewriter_node)
    workflow.add_node("generate", generator_node)
    
    # 3. Define the base execution flow (the directed edges)
    # The agent always starts by searching Qdrant
    workflow.set_entry_point("search")
    
    # After a search, it MUST always evaluate the results
    workflow.add_edge("search", "evaluate")
    
    # After evaluation, the Router decides the next step conditionally:
    # It will either go to 'generate' OR 'rewrite'
    workflow.add_conditional_edges(
        "evaluate",
        should_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    # If routed to rewrite, we MUST loop back to search again with the new query
    workflow.add_edge("rewrite", "search")
    
    # After generation, the task is complete.
    workflow.add_edge("generate", END)
    
    # 4. Compile the graph into an executable application
    app = workflow.compile()
    
    return app
