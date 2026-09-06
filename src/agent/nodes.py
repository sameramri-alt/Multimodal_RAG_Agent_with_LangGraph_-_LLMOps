from langchain_ollama import OllamaLLM
from src.agent.state import AgentState
from src.database.qdrant_manager import QdrantManager
from src.ingestion.embedder import MultiModalEmbedder

# We instantiate our tools ONLY ONCE at the module level.
# Thanks to the Singleton pattern, QdrantManager() always returns the same connection.
llm = OllamaLLM(model="llama3:8b")
embedder = MultiModalEmbedder()
qdrant = QdrantManager()

def search_node(state: AgentState) -> dict:
    """
    Step 1: The Search Node.
    Takes the question (query), transforms it into vectors (dense + sparse),
    and queries Qdrant to find the most relevant information.
    """
    print(f"--- SEARCH NODE (Attempt {state.get('search_attempts', 0) + 1}) ---")
    
    query = state["query"]
    chat_history = state.get("chat_history", "")
    
    # ---------------------------------------------------------
    # QUERY TRANSLATION (French to English) FOR CLIP
    # ---------------------------------------------------------
    # CLIP and our audio transcriptions are primarily in English.
    # If the user asks in French, the dense search will fail due to language mismatch.
    # We ask the LLM to translate the question before searching.
    translate_prompt = f"""Translate the following question into English so it can be used for a vector search.
Only return the translated text, nothing else. Do not answer the question.
If the question is short (like "and John?"), use the conversation history to understand the context and write a complete standalone question (e.g. "What about John's interest?").

Conversation History:
{chat_history}

Question: '{query}'
Translation:"""
    
    search_query = llm.invoke(translate_prompt).strip()
    # Clean up in case Llama adds quotes around the answer
    search_query = search_query.strip("'\"") 
    print(f"Optimized Search Query (translated): '{search_query}'")
    
    # Vectorize the translated question to capture both "meaning" (dense) and "keywords" (sparse)
    dense_vec = embedder.embed_text_dense(search_query)
    sparse_vec = embedder.embed_text_sparse(search_query)
    
    # Run the hybrid search in Qdrant (Limit increased to 15 to capture broader narrative context)
    results = qdrant.hybrid_search(query_dense=dense_vec, query_sparse=sparse_vec, limit=15)
    
    # Update state: save the retrieved contexts and increment the attempt counter
    return {
        "contexts": results,
        "search_attempts": state.get("search_attempts", 0) + 1
    }

def evaluator_node(state: AgentState) -> dict:
    """
    Step 2: The Evaluator Node.
    The AI (Llama 3) reads the retrieved documents and decides if the answer is actually there.
    It acts strictly: if the context doesn't directly address the question, it says NO.
    """
    print("--- EVALUATOR NODE ---")
    
    query = state["query"]
    # Sort contexts chronologically so Llama reads the story/presentation in the correct order
    contexts = sorted(state["contexts"], key=lambda x: x.get("start", 0))
    
    # Prepare the context as plain text for the LLM
    context_lines = []
    for c in contexts:
        text = c.get("text", "")
        if c.get("type") == "image_slide":
            text += " (Note: The visual search engine certifies this image matches the user's query)"
        context_lines.append(text)
    
    context_text = "\n".join(context_lines)
    
    # Strict prompt: the LLM must evaluate honestly without hallucinating.
    prompt = f"""You are a VERY STRICT evaluator.

User's question: {query}

Documents retrieved from the database:
{context_text}

Evaluation Rules:
- Answer YES ONLY if the documents contain information DIRECTLY related to the question.
- Answer NO if the documents are off-topic, empty, or if the information is missing.
- DO NOT GUESS. DO NOT USE your own general knowledge to fill in the blanks.

Answer ONLY with 'YES' or 'NO'."""
    
    # The LLM reads the prompt and evaluates
    evaluation = llm.invoke(prompt).strip().upper()
    
    # Update the state with the confidence boolean
    is_confident = "YES" in evaluation or "OUI" in evaluation
    print(f"AI Confidence: {'Sufficient' if is_confident else 'Insufficient'}")
    
    return {"is_confident": is_confident}

def rewriter_node(state: AgentState) -> dict:
    """
    Step 3: The Rewriter Node.
    If the AI did not find the answer, it rewrites the question to try again.
    """
    print("--- REWRITER NODE (Self-Correction) ---")
    
    query = state["query"]
    chat_history = state.get("chat_history", "")
    
    prompt = f"""The previous search for this question yielded no useful results: '{query}'.

Conversation History:
{chat_history}

Rewrite this question using DIFFERENT and SIMPLER keywords to improve the vector search.
Take the history into account to preserve the original intent (e.g. 'gain' means financial gain, not weight gain).
Return ONLY the new rewritten question, without any extra text."""

    new_query = llm.invoke(prompt).strip()
    print(f"New rewritten query: {new_query}")
    
    # Update state with the new query
    return {"query": new_query}

def generator_node(state: AgentState) -> dict:
    """
    Step 4: The Generator Node.
    Drafts the final response based STRICTLY on the retrieved documents.
    Forbidden from making things up or using general external knowledge.
    """
    print("--- GENERATOR NODE ---")
    
    query = state["query"]
    # Sort contexts chronologically
    contexts = sorted(state["contexts"], key=lambda x: x.get("start", 0))
    is_confident = state.get("is_confident", False)
    
    # If the AI is still not confident after max attempts, tell the user clearly
    if not is_confident and not contexts:
        return {"final_response": "❌ I could not find relevant information regarding this topic in the source materials. Please try rephrasing."}
    
    # Extract text, source, and timestamp for each retrieved document
    context_text = ""
    for c in contexts:
        text = c.get("text", "")
        source = c.get("source", "Unknown Source")
        
        if c.get("type") == "image_slide":
            text += " (The visual search engine certifies this image matches the user's query)"

        start_time = c.get("start", 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        time_str = f"[{minutes:02d}:{seconds:02d}]"
        context_text += f"[Source: {source}] {time_str} : {text}\n"

    chat_history = state.get("chat_history", "")
    
    # Strict anti-hallucination prompt + source citation + conversational history
    prompt = f"""You are an assistant answering questions about videos, audios, and presentations.

Conversation History:
{chat_history}

Current Question: '{query}'

Available Excerpts:
{context_text}

ABSOLUTE RULES:
1. Answer ONLY based on the excerpts provided above.
2. If the information is NOT in the excerpts, state clearly: "This information is not available in the sources."
3. Do NOT invent ANY facts, names, or numbers that are not explicitly stated in the excerpts.
4. ALWAYS cite the source using [Source: name] and the timestamp [MM:SS] for every piece of information.
5. Answer in the same language as the user's question (usually French).

Answer:"""
    
    final_response = llm.invoke(prompt)
    
    # Save the final generated text into the state
    return {"final_response": final_response}
