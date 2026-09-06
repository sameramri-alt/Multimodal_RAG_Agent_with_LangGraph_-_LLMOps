import os
import glob
import json
from pathlib import Path

from src.ingestion.media_processor import (
    scan_sessions,
    process_video_to_audio_and_frames,
    compute_directory_state
)
from src.ingestion.transcriber import transcribe_audio
from src.ingestion.embedder import MultiModalEmbedder
from src.database.qdrant_manager import QdrantManager
from src.agent.graph import build_agent

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
STATE_FILE = "./data/processed/ingestion_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT (Change Detection)
# ─────────────────────────────────────────────────────────────────────────────

def load_ingestion_state() -> dict:
    """
    Loads the state saved during the last ingestion run.
    The state is a dictionary: { file_path: md5_hash }.
    If no state is found (first run), returns an empty dictionary.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_ingestion_state(state: dict):
    """
    Saves the current state (MD5 hashes of all source files)
    into a JSON file after a successful ingestion.
    """
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion_pipeline(data_dir: str = DATA_RAW_DIR):
    """
    Multi-format and multi-file ingestion pipeline.

    STEPS:
    1. Scan: Identify all media sources in data/raw/
    2. Detect: Compare MD5 hashes to see if anything changed
    3. If changes detected → Reset Qdrant + Re-ingest ALL sources
    4. Save the new state

    SUPPORTED FORMATS:
    - .mp4 file → Audio extracted + slides (auto scene change detection)
    - .mp3 / .wav file → Audio transcription only (no images)
    - Subdirectory with audio + images → Transcribes audio AND vectorizes slides
    - Subdirectory with images only → Vectorizes slides (CLIP + OCR)
    """
    print("\n" + "="*50)
    print("STAGE 1: DATA INGESTION")
    print("="*50)

    # ── 1. Scan available sources ─────────────────────
    sessions = scan_sessions(data_dir)

    if not sessions:
        print("No data found in data/raw/")
        print("Please drop .mp4, .mp3, .wav files or folders with audio+images inside.")
        return False

    print(f"\n{len(sessions)} source(s) detected:")
    for s in sessions:
        print(f"  - [{s['type'].upper()}] {s['name']}")

    # ── 2. Change Detection (MD5 Hashing) ─────────────
    print(f"\nChecking for changes...")
    current_state = compute_directory_state(data_dir)
    previous_state = load_ingestion_state()

    # Detailed difference analysis
    new_files = set(current_state.keys()) - set(previous_state.keys())
    modified_files = {
        f for f in current_state
        if f in previous_state and current_state[f] != previous_state[f]
    }
    deleted_files = set(previous_state.keys()) - set(current_state.keys())

    has_changes = bool(new_files or modified_files or deleted_files)

    if not has_changes:
        print("No changes detected. The database is up to date!")
        return True

    # Display detected changes
    if new_files:
        print(f"  New files: {len(new_files)}")
        for f in new_files:
            print(f"    + {Path(f).name}")
    if modified_files:
        print(f"  Modified files: {len(modified_files)}")
        for f in modified_files:
            print(f"    ~ {Path(f).name}")
    if deleted_files:
        print(f"  Deleted files: {len(deleted_files)}")
        for f in deleted_files:
            print(f"    - {Path(f).name}")

    # ── 3. Qdrant Reset ───────────────────────────────
    # When source data changes, we start fresh to prevent stale/duplicate data.
    qdrant = QdrantManager()
    qdrant.recreate_collection()
    embedder = MultiModalEmbedder()

    all_documents = []

    # ── 4. Process each session ───────────────────────
    for session in sessions:
        print(f"\n{'─'*40}")
        print(f"Processing: {session['name']} [{session['type']}]")
        print(f"{'─'*40}")

        # Dedicated folder for temporary files of this session
        session_processed_dir = os.path.join(PROCESSED_DIR, session['name'])
        Path(session_processed_dir).mkdir(parents=True, exist_ok=True)

        audio_file = session['audio']
        image_files = list(session['images'])  # Copy the list
        # Dictionary to recover an image's timestamp from its path
        image_timestamps: dict[str, float] = {}

        # ── 4a. Extraction from MP4 ───────────────────
        if session['type'] == 'video' and session['video']:
            audio_file = os.path.join(session_processed_dir, "extracted_audio.wav")
            frames_dir = os.path.join(session_processed_dir, "frames")

            if os.path.exists(audio_file) and os.path.isdir(frames_dir):
                print(f"  Audio and frames already extracted, skipping...")
                image_files = sorted(
                    glob.glob(f"{frames_dir}/*.jpg") +
                    glob.glob(f"{frames_dir}/*.png")
                )
                # For already extracted frames, we reconstruct approximate timestamps
                # from the filenames (e.g., frame_0003.jpg → frame number 3)
                import re
                for img_path in image_files:
                    m = re.search(r"frame_(\d+)", img_path)
                    if m:
                        # Note: exact timestamps are lost on resume, we approximate 
                        # with 5s per saved frame based on standard intervals
                        image_timestamps[img_path] = int(m.group(1)) * 5.0
            else:
                saved_frames = process_video_to_audio_and_frames(
                    session['video'], audio_file, frames_dir
                )
                for fpath, ts in saved_frames:
                    image_timestamps[fpath] = ts
                image_files = [f for f, _ in saved_frames]

        # ── 4b. Audio Transcription ────────────────────
        segments = []
        if audio_file and os.path.exists(audio_file):
            transcript_file = os.path.join(session_processed_dir, "transcript.txt")
            audio_16k_file = os.path.join(session_processed_dir, "audio_16k.wav")
            print(f"\n  Transcribing '{session['name']}'...")
            segments = transcribe_audio(audio_file, output_16k_path=audio_16k_file, model_size="base")

            # Save the transcription into a readable text file
            with open(transcript_file, 'w', encoding='utf-8') as f:
                for seg in segments:
                    m_min = int(seg['start'] // 60)
                    m_sec = int(seg['start'] % 60)
                    f.write(f"[{m_min:02d}:{m_sec:02d}] {seg['text']}\n")
            print(f"  {len(segments)} segments transcribed.")

        # ── 4c. Vectorizing Audio segments ─────────────
        for seg in segments:
            text = seg['text']
            all_documents.append({
                "dense_vector": embedder.embed_text_dense(text),
                "sparse_vector": embedder.embed_text_sparse(text),
                "payload": {
                    "type": "audio_transcript",
                    "source": session['name'],  # SOURCE NAME!
                    "text": text,
                    "start": seg['start'],
                    "end": seg['end']
                }
            })

        # ── 4d. Vectorizing Images / Slides ────────────
        for img_path in image_files:
            time_sec = image_timestamps.get(img_path, 0.0)
            m_min = int(time_sec // 60)
            m_sec = int(time_sec % 60)
            time_str = f"{m_min:02d}:{m_sec:02d}"

            # Base text for this image (timestamp + source)
            frame_text = f"Visual slide from '{session['name']}' at [{time_str}]"

            # Attempt OCR to read visible text on the slide
            try:
                import easyocr
                if not hasattr(run_ingestion_pipeline, '_ocr_reader'):
                    print("  Loading OCR engine...")
                    run_ingestion_pipeline._ocr_reader = easyocr.Reader(
                        ['fr', 'en'], gpu=False, verbose=False
                    )
                ocr_results = run_ingestion_pipeline._ocr_reader.readtext(img_path, detail=0)
                if ocr_results:
                    ocr_text = " ".join(ocr_results).strip()
                    frame_text = f"[{time_str}]['{session['name']}'] Text on screen: {ocr_text}"
            except ImportError:
                pass  # easyocr not installed, fallback to basic text

            all_documents.append({
                "dense_vector": embedder.embed_image_dense(img_path),
                "sparse_vector": embedder.embed_text_sparse(frame_text),
                "payload": {
                    "type": "image_slide",
                    "source": session['name'],  # SOURCE NAME!
                    "path": img_path,
                    "text": frame_text,
                    "start": float(time_sec)
                }
            })

        print(f"  Session '{session['name']}': {len(segments)} audio segments + {len(image_files)} slides")

    # ── 5. Insertion into Qdrant ──────────────────────
    if all_documents:
        print(f"\nInserting {len(all_documents)} documents into Qdrant...")
        qdrant.insert_documents(all_documents)
    else:
        print("\nNo documents to ingest.")

    # ── 6. Save state for next run ────────────────────
    save_ingestion_state(current_state)
    print(f"\nAll data ingested successfully!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATIONAL AGENT
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_chat(data_dir: str = DATA_RAW_DIR):
    """
    Starts the interactive chat with the LangGraph agent.
    Preloads all available transcriptions to quickly answer full-transcript
    requests without going through the vector search agent.
    """
    print("\n" + "="*50)
    print("STAGE 2: MULTI-MODAL AGENT READY!")
    print("="*50)

    # Load all available full transcriptions
    all_transcripts = {}
    if os.path.exists(PROCESSED_DIR):
        for session_dir in glob.glob(f"{PROCESSED_DIR}/*/"):
            session_name = Path(session_dir).name
            transcript_file = os.path.join(session_dir, "transcript.txt")
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    all_transcripts[session_name] = f.read()

    if all_transcripts:
        sources = ", ".join(all_transcripts.keys())
        print(f"Loaded sources: {sources}")
        total_chars = sum(len(t) for t in all_transcripts.values())
        print(f"Loaded transcriptions ({total_chars} total characters).")

    # Build the LangGraph agent
    app = build_agent()

    print("\nAsk your questions (or type 'quit' to exit).")
    print("You can query all your sources simultaneously!\n")

    chat_history = ""

    while True:
        user_query = input("\nYou: ")

        if user_query.lower() in ["quit", "exit", "quitter"]:
            print("Goodbye!")
            break

        print("\nAgent is thinking...")

        # Detect requests for a full transcript
        keywords_transcription = [
            "transcri", "tout ce qui", "tout ce qu",
            "ce qui a ete dit", "ecri", "dit dans",
            "transcript", "everything that was said"
        ]
        is_full_transcript = any(kw in user_query.lower() for kw in keywords_transcription)

        if is_full_transcript and all_transcripts:
            print("\n--- FINAL RESPONSE ---")
            if len(all_transcripts) == 1:
                source_name = list(all_transcripts.keys())[0]
                print(f"Full transcript of '{source_name}':\n")
                print(list(all_transcripts.values())[0])
            else:
                print("Available transcripts:\n")
                for name, transcript in all_transcripts.items():
                    print(f"=== {name} ===")
                    print(transcript)
                    print()
            continue

        # Invoke the LangGraph agent with the current state
        initial_state = {
            "query": user_query, 
            "search_attempts": 0,
            "chat_history": chat_history
        }
        final_state = app.invoke(initial_state)

        response = final_state.get("final_response", "Error.")
        print("\n--- FINAL RESPONSE ---")
        print(response)
        
        # Append the exchange to the rolling chat history for context
        chat_history += f"User: {user_query}\nAssistant: {response}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_ingestion_pipeline(DATA_RAW_DIR)
    if success:
        run_agent_chat(DATA_RAW_DIR)
