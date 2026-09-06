import whisper
import warnings
import subprocess
import numpy as np


def convert_audio_to_16k_mono(audio_path: str, output_path: str = None) -> str:
    """
    Converts an audio file to 16 kHz mono using the bundled FFmpeg binary.

    WHY THIS APPROACH:
    - Standard scipy resampling (resample, resample_poly) distorts long audio files.
    - Whisper absolutely requires audio at precisely 16,000 Hz.
    - FFmpeg is the industry standard for audio conversion. We already have it
      installed via `imageio_ffmpeg` (used by MoviePy). We reuse it here to ensure stability.

    Parameters:
    - audio_path: The path of the source audio file (e.g., extracted_audio.wav at 44100 Hz).
    - output_path: Optional. The exact path where the converted file should be saved.

    Returns: The path of the new audio file converted to 16,000 Hz mono.
    """
    import imageio_ffmpeg

    # Retrieve the bundled FFmpeg executable path (e.g., ffmpeg-win-x86_64-v7.1.exe)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Define the output file if not explicitly provided
    if not output_path:
        output_path = audio_path.rsplit('.', 1)[0] + '_16k.wav'

    print(f"Converting audio to 16 kHz mono using FFmpeg...")
    cmd = [
        ffmpeg_exe,
        '-y',            # Overwrite output file if it already exists
        '-i', audio_path, # Source file
        '-ar', '16000',  # Target sample rate: 16,000 Hz (required by Whisper)
        '-ac', '1',      # Target channels: 1 (mono)
        '-f', 'wav',     # Output format: Uncompressed WAV
        output_path
    ]

    # Run FFmpeg and capture any error messages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    print(f"Conversion successful: {output_path}")
    return output_path


def transcribe_audio(audio_path: str, output_16k_path: str = None, model_size: str = "base"):
    """
    This function is responsible for listening to an audio file and transforming it into text.
    It uses Whisper (by OpenAI), which is capable of recognizing speech and pinpointing
    the exact second a sentence was spoken.

    Steps:
    1. Convert the audio to 16 kHz mono using FFmpeg (highly reliable method).
    2. Load the Whisper AI model into memory.
    3. Run the transcription on the converted file.
    4. Return a list of segments containing {start, end, text}.

    Parameters:
    - audio_path: The path to the source audio file (WAV format recommended).
    - output_16k_path: The path to save the temporary 16k audio file.
    - model_size: The size of the Whisper model to use. "base" is fast; "small" or "medium" 
                  offer higher accuracy but require more computational power.

    Returns: A list of dictionaries representing spoken segments.
    """

    # Step 1: Clean audio conversion via FFmpeg
    converted_path = convert_audio_to_16k_mono(audio_path, output_16k_path)

    print(f"Loading Whisper Artificial Intelligence model ({model_size}) into memory...")

    # Disable unnecessary technical warnings (e.g., FP16 execution on CPU)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the Whisper model
    model = whisper.load_model(model_size)

    # We read the converted WAV file DIRECTLY with scipy.
    # No need to resample here: FFmpeg has already produced a perfect 16,000 Hz mono file.
    # We obtain a numpy array that we pass to Whisper. Whisper accepts either:
    # a file path OR a numpy array. We choose numpy to prevent Whisper from calling
    # its own internal ffmpeg subprocess, which can sometimes fail or hang on Windows.
    print(f"Reading converted audio file...")
    from scipy.io import wavfile
    sample_rate, audio_data = wavfile.read(converted_path)

    # Normalize the signal to float32 between -1.0 and 1.0
    if audio_data.dtype == np.int16:
        audio_array = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_array = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_array = audio_data.astype(np.float32)

    print(f"Transcription in progress ({len(audio_array)/sample_rate:.1f} seconds of audio)...")

    # Pass the numpy array to Whisper — Whisper transcribes directly without calling ffmpeg again
    result = model.transcribe(audio_array)

    # Extract only the useful information: start time, end time, and transcribed text
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],      # In seconds, e.g., 12.5
            "end": seg["end"],          # In seconds, e.g., 15.0
            "text": seg["text"].strip() # The spoken text, without trailing/leading spaces
        })

    print(f"Transcription completed successfully: {len(segments)} segments found.")
    return segments
