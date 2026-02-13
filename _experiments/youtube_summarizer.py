import torch
# from sentence_transformers import SentenceTransformer # Not strictly needed
from transformers import T5ForConditionalGeneration, T5Tokenizer
import subprocess
import os
import re
import shutil
from typing import Optional
import sys
import json # Import json for potential future use if needed

# --- IMPORT YOUR CUSTOM MODULES ---
from topic_pipeline import generate_topic_name
# --- FIX: Import the correct worker function name ---
from test_trained_model import summarize_with_custom_model_worker

# -------------------------------
# Helper functions for YouTube transcript
# -------------------------------
def get_video_id(url: str) -> Optional[str]:
    # (function remains the same)
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def parse_vtt(filepath: str) -> str:
    # (function remains the same)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    text_lines = []
    for line in lines:
        if '-->' in line or line.strip().isdigit() or 'WEBVTT' in line or line.strip() == '':
            continue
        cleaned_line = re.sub(r'<[^>]+>', '', line).strip()
        text_lines.append(cleaned_line)
    return " ".join(dict.fromkeys(text_lines))

def get_youtube_transcript_with_yt_dlp(video_url: str) -> Optional[str]:
    # (function remains largely the same, added print statements)
    temp_dir = None
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            print(f"Error: Could not extract video ID from URL: {video_url}")
            return None

        temp_dir = f"yt_dlp_temp_{video_id}"
        os.makedirs(temp_dir, exist_ok=True)
        output_filename = os.path.join(temp_dir, f"{video_id}.%(ext)s")

        print(f"Attempting to download transcript for {video_id}...")
        command = [
            "yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--sub-format", "vtt",
            "--skip-download", "-o", output_filename, video_url
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
             print(f"yt-dlp error for {video_id}: {result.stderr}")
             return None

        downloaded_vtt = None
        for fname in os.listdir(temp_dir):
            if fname.startswith(video_id) and fname.endswith(".vtt"):
                downloaded_vtt = os.path.join(temp_dir, fname)
                print(f"Found transcript file: {downloaded_vtt}")
                break

        if not downloaded_vtt or not os.path.exists(downloaded_vtt):
            print(f"Error: Transcript VTT file not found for {video_id} in {temp_dir}")
            return None

        transcript = parse_vtt(downloaded_vtt)
        print(f"Successfully parsed transcript for {video_id} (Length: {len(transcript)} chars)")
        return transcript

    except subprocess.TimeoutExpired:
        print(f"Error: yt-dlp command timed out for {video_url}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred retrieving transcript for {video_url}: {e}")
        return None
    finally:
         if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                # print(f"Cleaned up temporary directory: {temp_dir}") # Optional: less verbose
            except Exception as cleanup_error:
                 print(f"Warning: Failed to cleanup temp directory {temp_dir}: {cleanup_error}")


# --- Standalone Test Block (REMOVED / COMMENTED OUT) ---
# It's better to test via the API server now.
# if __name__ == "__main__":
#    ... (code removed for clarity as it won't work easily with worker pattern)

