import glob
import os
import cv2
import hashlib
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE DETECTION (MD5 Hashing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """
    Calculates the digital fingerprint (MD5 Hash) of a file.
    If the file changes by even a single byte, its hash changes entirely.
    This allows us to detect when source data has been added, modified, or removed.
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read the file in chunks to avoid memory overload on large video files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_state(data_dir: str) -> dict:
    """
    Scans the data/raw directory and calculates the hash of every media file.
    Returns a dictionary mapping { filepath: md5_hash }.
    Used to detect if any source files changed since the last ingestion run.
    """
    state = {}
    # Search for all supported media files, recursively
    for ext in ('*.mp4', '*.mp3', '*.wav', '*.png', '*.jpg', '*.jpeg'):
        for filepath in glob.glob(f"{data_dir}/**/{ext}", recursive=True):
            # Ignore generated temporary files and processed folders
            norm_path = filepath.replace("\\", "/")
            if "processed" not in norm_path and "extracted" not in norm_path:
                state[norm_path] = compute_file_hash(filepath)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# AUTOMATIC SCENE / SLIDE CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_scene_changes(video_path: str, frames_dir: str, threshold: float = 25.0) -> list:
    """
    Analyzes a video frame by frame and automatically saves a screenshot 
    whenever a significant visual change is detected (e.g., a new presentation slide).

    HOW IT WORKS:
    1. Reads the video frame by frame.
    2. Compares each frame with the previous one (converted to grayscale for speed).
    3. If the average pixel difference exceeds the threshold, it is flagged as a scene change.
    4. Enforces a minimum delay of 3 seconds between saves to avoid capturing
       50 frames of a single slide transition animation.

    Parameters:
    - video_path: Path to the source MP4 file.
    - frames_dir: Output folder where extracted images will be saved.
    - threshold: Sensitivity of detection. Higher = less sensitive.

    Returns: A list of tuples containing (image_filepath, timestamp_in_seconds).
    """
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # Fallback default if FPS cannot be read from the video file

    saved_frames = []
    prev_gray = None
    frame_count = 0
    saved_count = 0
    last_saved_time = -10.0  # Force saving the very first frame
    MIN_INTERVAL_SECONDS = 3.0  # Minimum delay between two saved frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        # Convert to grayscale for faster comparison (color data isn't needed for this)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        should_save = False

        if prev_gray is None:
            # Always save the very first frame of the video
            should_save = True
        elif (current_time - last_saved_time) >= MIN_INTERVAL_SECONDS:
            # Calculate the absolute difference between current and previous frame
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(diff.mean())
            if mean_diff > threshold:
                should_save = True

        if should_save:
            filepath = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filepath, frame)
            saved_frames.append((filepath, current_time))
            saved_count += 1
            last_saved_time = current_time

        prev_gray = gray
        frame_count += 1

    cap.release()
    return saved_frames


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO & FRAME EXTRACTION (From MP4)
# ─────────────────────────────────────────────────────────────────────────────

def process_video_to_audio_and_frames(video_path: str, output_audio_path: str, frames_dir: str) -> list:
    """
    Extracts the audio track and key frames (slide changes) from an MP4 file.

    Returns: A list of tuples containing (image_filepath, timestamp_in_seconds).
    """
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    # Audio extraction using MoviePy
    print(f"  Extracting audio from video...")
    from moviepy import VideoFileClip
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(output_audio_path, logger=None)
    print(f"  Audio successfully extracted: {output_audio_path}")

    # Automatic scene change detection
    print(f"  Detecting slide/scene changes automatically...")
    saved_frames = detect_scene_changes(video_path, frames_dir)
    print(f"  {len(saved_frames)} unique slides/scenes detected and saved.")
    return saved_frames


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FORMAT INTELLIGENT SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan_sessions(data_dir: str) -> list:
    """
    Scans the 'raw' directory and identifies all data 'sessions' to ingest.

    A SESSION = a logical source of data (a video, a podcast, a lecture).
    It can take 3 forms:

    FORM 1: MP4 file directly in data/raw/
      → Contains both audio and video in a single file.

    FORM 2: MP3/WAV file directly in data/raw/
      → Audio only (e.g., podcast, voice memo).

    FORM 3: Subdirectory in data/raw/
      → Can contain: an audio file (.mp3/.wav) + slides (.png/.jpg).
      → All files in the folder are treated together as a single lecture/session.

    Returns: A list of dictionaries defining each session: {name, type, video, audio, images}.
    """
    sessions = []

    # --- FORM 1: MP4 files at the root of data/raw/ ---
    for mp4_file in sorted(glob.glob(f"{data_dir}/*.mp4")):
        name = Path(mp4_file).stem
        sessions.append({
            "name": name,
            "type": "video",
            "video": mp4_file,
            "audio": None,
            "images": []
        })

    # --- FORM 2: Audio files at the root of data/raw/ ---
    for audio_file in sorted(
        glob.glob(f"{data_dir}/*.mp3") + glob.glob(f"{data_dir}/*.wav")
    ):
        name = Path(audio_file).stem
        sessions.append({
            "name": name,
            "type": "audio_only",
            "video": None,
            "audio": audio_file,
            "images": []
        })

    # --- FORM 3: Subdirectories containing separated media ---
    for subdir in sorted(glob.glob(f"{data_dir}/*/")):
        # Ignore internal folders generated by our pipeline
        subdir_name = Path(subdir).name
        if subdir_name in ("processed", "extracted_frames"):
            continue

        audio_in_subdir = sorted(
            glob.glob(f"{subdir}*.mp3") + glob.glob(f"{subdir}*.wav")
        )
        images_in_subdir = sorted(
            glob.glob(f"{subdir}*.png") +
            glob.glob(f"{subdir}*.jpg") +
            glob.glob(f"{subdir}*.jpeg")
        )

        if not audio_in_subdir and not images_in_subdir:
            continue  # Empty folder or no recognized media

        # Determine the session type based on its contents
        if audio_in_subdir and images_in_subdir:
            session_type = "audio_with_slides"
        elif audio_in_subdir:
            session_type = "audio_only"
        else:
            session_type = "slides_only"

        sessions.append({
            "name": subdir_name,
            "type": session_type,
            "video": None,
            "audio": audio_in_subdir[0] if audio_in_subdir else None,
            "images": images_in_subdir
        })

    return sessions
