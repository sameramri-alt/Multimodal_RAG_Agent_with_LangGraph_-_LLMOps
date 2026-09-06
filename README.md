# Multimodal RAG Agent with LangGraph & LLMOps 🚀

An advanced, **100% local**, and privacy-first Multimodal RAG (Retrieval-Augmented Generation) agent. It processes audio, video, and image presentation slides to answer your questions accurately using an intelligent, state-driven workflow.

This project integrates a local vector database (**Qdrant**), an orchestration graph (**LangGraph**), local speech-to-text (**Whisper**), multimodal embeddings (**CLIP**), and full LLMOps observability (**Arize Phoenix**). 

Everything runs locally on your machine with **zero cloud dependencies** and **zero API costs**.

---

## 🌟 Key Features

*   **Multimodal Ingestion Pipeline:** 
    *   Automatically extracts audio and key frames (slide changes) from `.mp4` video presentations.
    *   Transcribes audio using OpenAI's **Whisper** (runs locally).
    *   Extracts text from presentation slides using **EasyOCR**.
*   **Hybrid Vector Search (Qdrant):** Combines Dense semantic search (via `clip-ViT-B-32`) and Sparse keyword search (via `FastEmbed BM25`) using Reciprocal Rank Fusion (RRF) to retrieve the best context regardless of modality (text vs. image).
*   **Stateful Agentic Workflow (LangGraph):** The agent follows a defined graph logic:
    *   **Search Node:** Retrieves relevant audio transcripts and slides.
    *   **Evaluator Node:** A strict LLM judge checks if the retrieved context actually answers the question.
    *   **Rewriter Node:** If the context is poor, the agent intelligently rewrites the user's query and searches again (auto-correction loop).
    *   **Generator Node:** Drafts the final answer, strictly citing sources and timestamps (e.g., `[Source: Video1] [01:15]`).
*   **Conversational Memory:** Remembers chat history to resolve ambiguous follow-up questions (e.g., *"And what about John?"*).
*   **LLMOps Supervision (Arize Phoenix):** Real-time tracing of every agent step, LLM prompt, and latency metric in a local dashboard.

---

## 🏗️ Architecture

![Architecture](https://img.shields.io/badge/Architecture-LangGraph-blue) ![DB](https://img.shields.io/badge/VectorDB-Qdrant-red) ![LLM](https://img.shields.io/badge/LLM-Ollama_(Llama_3)-orange) ![Observability](https://img.shields.io/badge/LLMOps-Arize_Phoenix-green)

1.  **Ingestion & Embeddings:** Media files -> Whisper (Text) + CLIP (Images) -> Qdrant Local.
2.  **Orchestration:** LangGraph controls the decision-making loop (`Search` → `Evaluate` ⇄ `Rewrite` → `Generate`).
3.  **Observability:** OpenTelemetry hooks capture LangChain traces and send them to the Arize Phoenix UI.

---

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **Ollama:** You must have [Ollama](https://ollama.com/) installed and running locally.
*   **Llama 3 Model:** Pull the required model via terminal:
    ```bash
    ollama run llama3:8b
    ```

*(Note: FFmpeg is automatically bundled and configured via Python, no system-level installation is required!)*

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/multimodal-agent.git
    cd multimodal-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add your data:**
    Place your media files in the `data/raw/` directory. Supported formats:
    *   `.mp4` videos (placed at root or in subfolders)
    *   `.mp3` / `.wav` audio files
    *   Subfolders containing separated audio files and `.png`/`.jpg` slides.

---

## 🚀 Usage

The system operates using two separate terminals for maximum observability.

### Terminal 1: Start LLMOps Supervision (Optional but Recommended)
Start the Arize Phoenix tracing server to monitor the agent's thought process.
```bash
python -m phoenix.server.main serve
```
*Open your browser to [http://localhost:6006](http://localhost:6006)*

### Terminal 2: Run the Agent
Execute the main pipeline. It will automatically detect new files, ingest them into Qdrant, and start the interactive chat.
```bash
python run.py
```

### Example Interaction:
```text
You: What is the total gain of Lisa after 30 years?

Agent is thinking...
--- SEARCH NODE (Attempt 1) ---
Optimized Search Query (translated): 'What is the total gain of Lisa after 30 years?'
--- EVALUATOR NODE ---
AI Confidence: Sufficient
--- GENERATOR NODE ---

--- FINAL RESPONSE ---
According to the excerpts, after year 30, Lisa will have $17,449 and 40 cents.
[Source: Compound Interest] [01:19]
```

To see exactly what prompt was sent to the LLM during evaluation or generation, check the Phoenix dashboard!

---

## 📁 Project Structure

```text
multimodal_agent/
├── data/
│   ├── raw/                 # Put your source files here (.mp4, .mp3, .png)
│   └── processed/           # Auto-generated transcripts and extracted frames
├── qdrant_data/             # Local Qdrant vector database storage
├── src/
│   ├── agent/
│   │   ├── graph.py         # LangGraph workflow definition (edges/nodes)
│   │   ├── nodes.py         # Core logic for search, evaluate, rewrite, generate
│   │   └── state.py         # AgentState typed dictionary (memory)
│   ├── database/
│   │   └── qdrant_manager.py# Hybrid search and DB management
│   ├── ingestion/
│   │   ├── embedder.py      # CLIP and FastEmbed BM25 integrations
│   │   ├── media_processor.py # File hashing, scene detection, audio extraction
│   │   └── transcriber.py   # Whisper AI transcription and FFmpeg conversion
│   └── main.py              # Ingestion pipeline and chat loop
├── run.py                   # Main entry point and Phoenix tracing setup
└── requirements.txt
```

---

## 🧠 Why Hybrid Search?
Standard vector databases struggle when comparing text queries to image embeddings (Dense search). We implemented **Reciprocal Rank Fusion (RRF)**. Qdrant performs a semantic search (CLIP) and an exact keyword search (BM25 over EasyOCR extracted text), fusing the scores so text-heavy slides are never ignored.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
