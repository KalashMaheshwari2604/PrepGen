import torch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import subprocess
import os
import re
import shutil
from typing import Optional
import sys

# --- IMPORT YOUR CUSTOM MODULES ---
# Summarization is now handled by ensemble_engine.py in main.py
# from summarization_engine import summarize_with_custom_model  # Not used in production

# -------------------------------
# Helper functions for YouTube transcript
# -------------------------------
def get_video_id(url: str) -> Optional[str]:
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def parse_vtt(filepath: str) -> str:
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
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            return None

        temp_dir = "yt_dlp_temp"
        os.makedirs(temp_dir, exist_ok=True)
        output_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")

        command = [
            "yt-dlp",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "-o", output_template,
            video_url
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        downloaded_vtt = None
        for fname in os.listdir(temp_dir):
            if fname.startswith(video_id) and fname.endswith(".en.vtt"):
                downloaded_vtt = os.path.join(temp_dir, fname)
                break

        if not downloaded_vtt:
            shutil.rmtree(temp_dir)
            return None

        transcript = parse_vtt(downloaded_vtt)
        shutil.rmtree(temp_dir)
        return transcript

    except subprocess.CalledProcessError:
        return None
    except Exception:
        return None
