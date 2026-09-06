"""
run.py — Main Entry Point
==========================
This is the single script you need to run the entire pipeline.
Launch it from the project root directory:

    python run.py

It performs 3 setup tasks before starting the application:
    1. Adds the project root to sys.path so that 'src' imports work correctly.
    2. Configures FFmpeg (via imageio_ffmpeg) so that Whisper and MoviePy
       can decode any audio/video file without a system-level FFmpeg install.
    3. (Optional) Connects to the local Arize Phoenix LLMOps dashboard
       to trace all LLM calls in real time. Phoenix must be running separately:
           python -m phoenix.server.main serve
       then open http://localhost:6006 in your browser.
"""

import sys
import os

# ── Setup 1: Python Path ──────────────────────────────────────────────────────
# Insert the project root into Python's module search path.
# This allows all `from src.xxx import yyy` statements to work correctly
# regardless of which directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Setup 2: FFmpeg Path ──────────────────────────────────────────────────────
# Whisper (speech-to-text) and MoviePy (video processing) both require FFmpeg
# to read and convert audio/video files.
# Instead of asking the user to install FFmpeg system-wide, we use the binary
# bundled inside the `imageio_ffmpeg` Python package (already installed).
# We expose it to the current session by prepending its directory to PATH.
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"FFmpeg configured: {ffmpeg_exe}")
except Exception as e:
    print(f"WARNING: Could not configure FFmpeg automatically: {e}")
    print("         Install FFmpeg manually if audio/video errors occur.")

# ── Setup 3: LLMOps Observability — Arize Phoenix (Optional) ─────────────────
# Arize Phoenix is a 100% local, open-source LLMOps observability platform.
# When running, it provides a real-time dashboard at http://localhost:6006
# showing every LLM call made by the agent, including:
#   - The exact prompt sent to Llama 3
#   - The model's raw response
#   - Latency per node (Search, Evaluate, Rewrite, Generate)
#   - Token counts
#
# PREREQUISITE: Start Phoenix server in a separate terminal first:
#   python -m phoenix.server.main serve
#
# This block is wrapped in try/except so the app still works if Phoenix
# is not running (the agent simply runs without tracing).
try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # Register the OpenTelemetry tracer provider pointing to the local Phoenix server.
    # All LangChain/LangGraph/Ollama calls will be automatically captured.
    tracer_provider = register(
        project_name="multimodal-agent",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=False,
        verbose=False,
    )

    # Instrument LangChain: this single call patches all LangChain LLM wrappers
    # so that every llm.invoke(...) is automatically recorded as a Phoenix span.
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("LLMOps tracing enabled (Phoenix at http://localhost:6006)")

except Exception as e:
    print(f"LLMOps tracing unavailable (optional): {e}")

# ── Application Launch ────────────────────────────────────────────────────────
# Import and run the two main pipeline stages from src/main.py
from src.main import run_ingestion_pipeline, run_agent_chat

if __name__ == "__main__":
    DATA_DIR = "./data/raw"

    # Stage 1: Ingest data (transcribe audio, vectorize slides, store in Qdrant)
    # This step is skipped automatically if no files have changed (MD5 cache check).
    success = run_ingestion_pipeline(DATA_DIR)

    # Stage 2: Launch the interactive chat agent if ingestion succeeded
    if success:
        run_agent_chat()
