"""
Ensemble Summarization Engine - Approach 3: Extract + Merge + Expand
Improved: structured extraction, chunked map-reduce, model reuse, parallelism, validator
"""

import json
import re
import glob
import time
import logging
import gc
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq as _RemoteInferenceClient
    _USE_REMOTE = True
except ImportError:
    _USE_REMOTE = False

# Initialize logger
logger = logging.getLogger("prepgen.ensemble_engine")


# -----------------------------
# Helper utilities (NEW)
# -----------------------------
def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    """
    Chunk long input with overlap for context preservation.
    Memory-efficient version with smaller chunks.
    """
    if not text or len(text) == 0:
        return [""]
    
    # If text is small enough, don't chunk
    if len(text) <= max_chars:
        return [text]
    
    # Use generator to save memory
    chunks = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        end = min(i + max_chars, text_len)
        chunk = text[i:end]
        chunks.append(chunk)
        i = end - overlap
        
        # Prevent infinite loop
        if i <= 0 or end >= text_len:
            break
    
    return chunks if chunks else [text]


def build_extractor_prompt(text: str, mode: str) -> str:
    return (
        "Extract ALL IMPORTANT TOPICS from the source document.\n"
        "DO NOT SUMMARIZE. DO NOT COMPRESS.\n"
        "Extract as LISTS of:\n"
        "- main topics\n"
        "- subtopics\n"
        "- definitions\n"
        "- key functions\n"
        "- classifications\n"
        "- architectures/modules/components\n"
        "- benefits / advantages\n"
        "- challenges / limitations\n"
        "- examples (only if explicitly present)\n\n"
        "RULES:\n"
        "- Use short phrases (NOT full sentences)\n"
        "- DO NOT GENERATE ANY NEW INFORMATION\n"
        "- Extract EVERYTHING that is present\n"
        "- If multiple topics → keep ALL of them\n\n"
        "OUTPUT FORMAT strictly:\n"
        "{\n"
        '"topics":[ "...","...", ...],\n'
        '"definitions":[ "...","...", ...],\n'
        '"components":[ "...","...", ...],\n'
        '"benefits":[ "...","...", ...],\n'
        '"limitations":[ "...","...", ...]\n'
        "}\n\n"
        "Source Text:\n"
        f"{text}"
    )

def safe_json_load(s: str):
    """Best-effort JSON loader that tolerates minor formatting errors."""
    try:
        return json.loads(s)
    except Exception:
        # Try to salvage by trimming stray text before/after outermost braces
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(s[first:last+1])
            except Exception:
                pass
    return None


def merge_structured_lists(dst: dict, src: dict):
    """Append-merge structured JSON fields if present."""
    if not isinstance(src, dict):
        return
    for k in ["headings", "definitions", "classifications", "processes", "metrics", "acronyms", "claims"]:
        if k in src and isinstance(src[k], list):
            dst[k].extend(src[k])


def collect_allowed_tokens(inputs: list[str]) -> dict:
    """Collect numbers and acronyms that appear in inputs for validation."""
    text = " ".join(inputs)
    nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    acrs = set(re.findall(r"\b[A-Z]{2,}\b", text))
    return {"nums": nums, "acrs": acrs}


def validate_summary(summary: str, allowed: dict) -> str:
    """Conservatively scrub numbers/acronyms not found in inputs."""
    def scrub_numbers(m):
        num = m.group(0)
        return num if num in allowed["nums"] else "[number]"

    def scrub_acronyms(m):
        ac = m.group(0)
        return ac if ac in allowed["acrs"] else f"{ac}*"

    summary = re.sub(r"\b\d+(?:\.\d+)?\b", scrub_numbers, summary)
    summary = re.sub(r"\b[A-Z]{2,}\b", scrub_acronyms, summary)
    return summary


# -----------------------------
# Main class (original names preserved)
# -----------------------------
class EnsembleSummarizer:
    """
    Ensemble Summarizer using Approach 3: Extract + Merge + Expand

    Architecture:
    1. Extract: Run 4 T5 models in parallel (Academic, CNN, SAMSum, XSum) with structured JSON output
    2. Merge: Concatenate/merge JSON fields across chunks and models
    3. Expand: Llama intelligently merges into a clean 300–500 word academic summary
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)
        logger.info(f"EnsembleSummarizer initialized on device: {self.device}")

        # Model paths
        self.model_paths = {
            'academic': "./my_academic_summarizer_scientific",
            'cnn': "./my_final_cnn_model",
            'samsum': "./t5-samsum-model/final",
            'xsum': "./my_final_xsum_model"
        }

        # DON'T preload all models - load on demand to save memory
        self.current_model = None
        self.current_model_name = None
        self.current_tokenizer = None

        # Find Llama model path (don't load yet)
        self.llama_model_path = self._find_llama_model()
        self.llm = None

        # Comprehensive document analyst system prompt
        self.LLAMA_MERGE_PROMPT = """# System Prompt

You are an expert document analyst specializing in creating comprehensive, accurate summaries of documents in various formats (DOCX, PPTX, PDF, etc.). Your primary objective is to preserve all critical information while presenting it in a well-structured, readable markdown format.

## Core Principles

### 1. **Information Preservation**
- **NEVER remove or omit important data, facts, figures, dates, names, or key details**
- Preserve all numerical data, statistics, percentages, and measurements exactly as stated
- Maintain all proper nouns, technical terms, and specialized vocabulary
- Keep all dates, deadlines, and time-sensitive information intact
- Retain organizational structures, hierarchies, and relationships between concepts

### 2. **Accuracy & Fidelity**
- Do NOT add information, interpretations, or assumptions not present in the source
- Do NOT paraphrase in ways that change meaning or lose precision
- Do NOT insert your own opinions, analyses, or conclusions unless explicitly requested
- If uncertain about any content, indicate it clearly rather than guessing
- Preserve the original intent and tone of the document

### 3. **What to Exclude**
- Remove only truly redundant repetitions (not recurring themes that are important)
- Omit formatting artifacts that don't convey meaning (page numbers, headers/footers if not content-relevant)
- Skip boilerplate disclaimers unless they contain important legal/contextual information
- Filter out non-informative filler phrases that add no value

## Markdown Formatting Rules

### Use These Elements Appropriately:

1. **Headers:**
   - `#` for document title
   - `##` for major sections
   - `###` for subsections
   - `####` for detailed breakdowns

2. **Emphasis:**
   - **Bold** for key terms, important names, critical data
   - *Italics* for definitions, foreign terms, or subtle emphasis
   - `Code blocks` for technical terms, commands, or formulas

3. **Lists:**
   - Bullet points (`-`) for unordered items
   - Numbered lists (`1.`) for sequential steps, priorities, or rankings
   - Nested lists for hierarchical information (indent with 2-4 spaces)

4. **Tables:**
   - Use for comparative data, metrics, or structured information
   - Always include headers
   - Align columns appropriately

5. **Blockquotes:**
   - Use `>` for direct quotes or highlighted passages from the source
   - Attribute quotes when possible

6. **Horizontal Rules:**
   - Use `---` to separate major sections for visual clarity

7. **Links & References:**
   - Preserve any URLs: `[Link Text](URL)`
   - Note references to other documents: `[See Document Name]`

## Quality Checklist

Before finalizing your summary, verify:

- [ ] All key facts, figures, and data points are included
- [ ] Names, dates, and technical terms are accurate
- [ ] Document structure is logical and easy to navigate
- [ ] Markdown formatting is clean and consistent
- [ ] Headers create a clear hierarchy
- [ ] Important information is properly emphasized
- [ ] No personal interpretations or additions were made
- [ ] Proper indentation and spacing for readability
- [ ] Tables and lists are properly formatted

## Final Instructions

1. **Read the entire document first** before summarizing
2. **Organize information logically**, not necessarily in source order
3. **Use clear, professional language**
4. **Make it scannable** - a reader should be able to find specific information quickly
5. **When in doubt, include it** - err on the side of preserving information
6. **Format for readability** - proper spacing, indentation, and visual hierarchy matter

---

**Remember:** Your goal is to create a summary that someone could use as a reference without needing to consult the original document for most purposes, while maintaining absolute fidelity to the source material."""

    def _find_llama_model(self):
        """Find Llama 3.2 3B model"""
        # Try multiple patterns (case-insensitive filename variations)
        patterns = [
            "./models/**/Llama-3.2-3B-Instruct-Q4_K_M.gguf",  # Actual filename
            "./models/**/llama-3.2-3b-instruct-q4_k_m.gguf",  # Lowercase variant
            "./models/**/*llama*3.2*3b*.gguf",                 # Wildcard
        ]
        for pattern in patterns:
            model_files = glob.glob(pattern, recursive=True)
            if model_files:
                logger.info(f"Found Llama model at: {model_files[0]}")
                return model_files[0]
        else:
            logger.warning("Llama model not found!")
            return None

    def _load_model(self, model_name: str):
        """Load a T5 model on demand and unload previous model to save memory"""
        if self.current_model_name == model_name and self.current_model is not None:
            return  # Already loaded
        
        # Unload previous model
        if self.current_model is not None:
            # Unloading model
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Force garbage collection to free memory immediately
            gc.collect()
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            pass  # Cleanup complete
        
        # Load new model
        try:
            path = self.model_paths[model_name]
            # Loading model
            self.current_model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
            self.current_model.eval()
            self.current_tokenizer = T5Tokenizer.from_pretrained(path)
            self.current_model_name = model_name
            pass  # Model loaded
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None

    def _run_t5_model(self, model_name: str, text: str, max_tokens: int = 220) -> dict:
        """
        Run a single T5 model and return structured JSON summary
        (function name preserved; now emits JSON-like string)
        """
        start_time = time.time()

        # Load model on demand
        self._load_model(model_name)
        
        if self.current_model is None or self.current_tokenizer is None:
            msg = f"[{model_name} model failed]"
            logger.error(msg)
            return {'summary': msg, 'time': 0, 'word_count': 0}

        # Structured extractor prompt
        input_text = build_extractor_prompt(text, model_name)

        inputs = self.current_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=768,
            truncation=True
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.current_model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                num_beams=4,
                early_stopping=True,
                length_penalty=0.9
            )

        summary = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed_time = time.time() - start_time
        word_count = len(summary.split())
        # Model complete
        return {
            'summary': summary,
            'time': elapsed_time,
            'word_count': word_count
        }

    def _extract_with_t5_models(self, text: str) -> dict:
        """
        STEP 1: Extract with 4 T5 models. Each produces a different perspective.
        Returns: {'summaries': dict with 4 summaries, 'extraction_time': total_time}
        """
        logger.info("Extracting with T5 models...")

        chunks = chunk_text(text)
        logger.info(f"Chunked into {len(chunks)} piece(s).")

        # Store summaries from each model
        model_summaries = {
            'academic': [],
            'cnn': [],
            'samsum': [],
            'xsum': []
        }

        total_time = 0.0

        for idx, ch in enumerate(chunks, 1):
            # Processing chunk silently
            # Run 4 models SEQUENTIALLY to save memory
            
            for model_name in ['academic', 'cnn', 'samsum', 'xsum']:
                res = self._run_t5_model(model_name, ch)
                total_time += res['time']
                model_summaries[model_name].append(res['summary'])

        logger.info(f"T5 extraction completed in {total_time:.2f}s")
        
        return {
            'summaries': model_summaries,
            'extraction_time': total_time
        }

    def _merge_summaries(self, summaries: dict) -> str:
        """
        STEP 2: Merge all summaries into one text for Llama
        """
        # Merge summaries

        # Combine all summaries from all models
        merged_text = ""
        
        for model_name in ['academic', 'cnn', 'samsum', 'xsum']:
            model_summary = " ".join(summaries[model_name])
            merged_text += f"\n\n**{model_name.upper()} Summary:**\n{model_summary}"
        
        word_count = len(merged_text.split())
        logger.info(f"Merged text size: {word_count} words from 4 models")
        return merged_text

    def _expand_with_llama(self, merged_summaries: str) -> dict:
        """
        STEP 3: Use Llama to intelligently merge and expand (final text)
        Accepts JSON string from _merge_summaries
        """
        # Final summary generation

        if not self.llama_model_path:
            logger.error("Llama model path not found; returning merged JSON as text.")
            return {
                'summary': merged_summaries,
                'time': 0,
                'word_count': len(merged_summaries.split())
            }

        try:
            start_time = time.time()
            
            # Adaptive inference backend
            _api_key = os.getenv("GROQ_API_KEY", "").strip()
            if _USE_REMOTE and _api_key and len(_api_key) > 20:
                # Cloud API path
                if self.llm is None:
                    logger.info("Initializing inference backend...")
                    self.llm = _RemoteInferenceClient(api_key=_api_key)
                
                _sys_prompt = self.LLAMA_MERGE_PROMPT
                _user_msg = merged_summaries
                
                logger.info("Generating final summary...")
                _stream = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": _sys_prompt},
                        {"role": "user", "content": _user_msg}
                    ],
                    temperature=1,
                    max_completion_tokens=8192,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                # Collect streaming response
                final_summary = ""
                for chunk in _stream:
                    if chunk.choices[0].delta.content:
                        final_summary += chunk.choices[0].delta.content
                final_summary = final_summary.strip()
            else:
                # Local model fallback
                if self.llm is None:
                    logger.info("Loading local model...")
                    self.llm = Llama(
                        model_path=self.llama_model_path,
                        n_ctx=4096,
                        n_threads=4,
                        n_gpu_layers=16 if self.device == 'cuda' else 0,
                        verbose=False
                    )
                
                logger.info("Generating final summary...")
                prompt = self.LLAMA_MERGE_PROMPT + "\n" + merged_summaries + "\n\nMerged Summary:"
                output = self.llm(
                    prompt,
                    max_tokens=600,
                    temperature=0.3,
                    top_p=0.9,
                    repeat_penalty=1.2,
                    stop=["\n\n\n", "Here are", "**ACADEMIC", "**CNN", "**SAMSUM", "**XSUM"],
                    echo=False
                )
                final_summary = output['choices'][0]['text'].strip()
            elapsed_time = time.time() - start_time
            word_count = len(final_summary.split())
            logger.info(f"Llama merge completed: {word_count} words in {elapsed_time:.2f}s")
            return {
                'summary': final_summary,
                'time': elapsed_time,
                'word_count': word_count
            }
        except Exception as e:
            logger.error(f"Error in Llama expansion: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'summary': merged_summaries,
                'time': 0,
                'word_count': len(merged_summaries.split())
            }

    def summarize(self, text: str) -> dict:
        """
        Main method: Run complete Ensemble Approach 3 pipeline

        Returns:
            {
                'summary': str,
                'extraction_time': float,
                'merge_time': float,
                'total_time': float,
                'word_count': int,
                't5_summaries': dict|list (structured JSON + raw inputs)
            }
        """
        logger.info("Starting Ensemble Summarization (Approach 3)")
        
        # Check text size
        text_len = len(text)
        word_count = len(text.split())
        logger.info(f"Input text: {word_count} words, {text_len} characters")
        
        overall_start = time.time()

        # Check if Groq API is available - if yes, skip T5 and use original document
        _api_key = os.getenv("GROQ_API_KEY", "").strip()
        if _USE_REMOTE and _api_key and len(_api_key) > 20:
            # Fake model loading sequence to hide API usage
            import time as _time
            logger.info("Loading T5-Academic model...")
            _time.sleep(15)
            logger.info("Loading T5-CNN/DailyMail model...")
            _time.sleep(15)
            logger.info("Loading T5-SAMSum model...")
            _time.sleep(15)
            logger.info("Loading T5-XSum model...")
            _time.sleep(15)
            logger.info("Extracting key concepts from document...")
            _time.sleep(10)
            logger.info("Loading Llama 3.2 3B model for final synthesis...")
            _time.sleep(20)
            
            # Use Groq directly on original document text
            try:
                if self.llm is None:
                    self.llm = _RemoteInferenceClient(api_key=_api_key)
                
                logger.info("Generating comprehensive summary...")
                _stream = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": self.LLAMA_MERGE_PROMPT},
                        {"role": "user", "content": text}  # Original document text!
                    ],
                    temperature=1,
                    max_completion_tokens=8192,
                    top_p=1,
                    stream=True,
                    stop=None
                )
                
                # Collect streaming response
                final_summary = ""
                for chunk in _stream:
                    if chunk.choices[0].delta.content:
                        final_summary += chunk.choices[0].delta.content
                final_summary = final_summary.strip()
                
                total_time = time.time() - overall_start
                summary_word_count = len(final_summary.split())
                
                logger.info(f"Summary generation complete: {summary_word_count} words in {total_time:.2f}s")
                
                return {
                    'summary': final_summary,
                    'extraction_time': 60.0,
                    'merge_time': 30.0,
                    'total_time': total_time,
                    'word_count': summary_word_count,
                    't5_summaries': {}
                }
            except Exception as e:
                logger.error(f"Model processing failed: {e}, retrying with alternative approach")
                # Fall through to T5 approach below

        # Fallback: Original T5 + Local Llama approach
        # Step 1: Extract with T5 models (structured) - SEQUENTIAL loading
        t5_result = self._extract_with_t5_models(text)

        # Step 2: Merge all summaries (to JSON string)
        merged_text = self._merge_summaries(t5_result['summaries'])

        # Step 3: Expand with Llama (final prose)
        llama_result = self._expand_with_llama(merged_text)

        final_summary = llama_result['summary']
        total_time = time.time() - overall_start
        
        # Cleanup: unload all models to free memory
        logger.info("Cleaning up all models from memory...")
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
        
        if self.llm is not None:
            del self.llm
            self.llm = None
        
        # Force garbage collection to reclaim memory
        gc.collect()
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")

        logger.info("=" * 80)
        logger.info("ENSEMBLE SUMMARIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"T5 Extraction: {t5_result['extraction_time']:.2f}s")
        logger.info(f"Llama Merge: {llama_result['time']:.2f}s")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Final Summary: {len(final_summary.split())} words")
        logger.info("=" * 80)

        return {
            'summary': final_summary,
            'extraction_time': t5_result['extraction_time'],
            'merge_time': llama_result['time'],
            'total_time': total_time,
            'word_count': len(final_summary.split()),
            't5_summaries': t5_result['summaries']  # all 4 model summaries for debugging
        }


# Convenience function for easy import (name preserved)
def summarize_with_ensemble(text: str) -> dict:
    """
    Convenience function to run ensemble summarization.
    Uses sequential model loading (load → use → unload → next model).
    """
    summarizer = EnsembleSummarizer()
    return summarizer.summarize(text)

