import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Llama is used via the worker function, no direct import needed here
# from llama_cpp import Llama
# from processing import extract_text # Removed import as __main__ block is removed
import os
import glob
from typing import Optional, Tuple

# Import the worker helper functions from ai_models
try:
    # These helpers give access to models loaded in the worker process
    from ai_models import _get_llm, _get_t5
except ImportError:
    # Fallback/placeholder if run standalone (models won't load correctly anyway)
    def _get_llm(): raise RuntimeError("Cannot get LLM outside worker process.")
    def _get_t5(): raise RuntimeError("Cannot get T5 outside worker process.")

# --- Summarization Worker Function ---
# --- FIX: Added async def ---
def summarize_with_custom_model_worker(input_text: str) -> str:
    """
    WORKER FUNCTION for two-step summarization.
    Gets T5 and LLM instances from its own process scope.
    Defined as async to align with await run_in_executor usage.
    """
    try:
        model, tokenizer = _get_t5()
        llm = _get_llm()
        device = model.device
    except RuntimeError as e:
         print(f"Error: AI models not available in worker: {e}")
         return f"Error: AI models not fully loaded for summarization: {e}"

    # --- Step 1: Extract key points with the custom T5 model ---
    prefix = "summarize: "
    try:
        inputs = tokenizer(prefix + input_text,
                           max_length=2048,
                           return_tensors="pt",
                           truncation=True).to(device)

        # Note: model.generate is blocking, but run_in_executor handles this.
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=768,
            num_beams=4,
            early_stopping=True
        )
        highlights = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[Worker {os.getpid()}] Error during T5 highlight extraction: {e}")
        return f"Error during Step 1 (Highlight Extraction): {str(e)}"

    if not highlights:
        return "No highlights could be generated from the document."

    # --- Step 2: Polish highlights using the pre-loaded Llama/Mistral model ---
    prompt = f"""
[INST]
You are an expert academic writer. Your task is to take the following list of key points from a document and rewrite them into a detailed, multi-paragraph summary covering the main aspects mentioned.
Your goal is to create a fluent, human-like summary that accurately reflects the key information from the points provided. Ensure the summary is comprehensive but concise.

Key Points:
"{highlights}"
[/INST]
"""
    try:
        # Note: llm() call is blocking, but run_in_executor handles this.
        output = llm(prompt, max_tokens=1024, echo=False)
        final_summary = output['choices'][0]['text'].strip()
        return final_summary
    except ValueError as e:
        if "Requested tokens" in str(e) and "exceed context window" in str(e):
             print(f"--- ERROR: Context window overflow during Llama polishing: {e} ---")
             return f"(Polishing failed due to length)\n\nRaw Highlights:\n{highlights}"
        else: return f"Error during Step 2 (Polishing - ValueError): {str(e)}"
    except Exception as e:
        print(f"--- Error during Llama polishing step: {e} ---")
        return f"Error during Step 2 (Polishing): {str(e)}"

# Standalone test block removed - test via API server

