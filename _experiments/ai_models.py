import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_cpp import Llama
import os
import functools
import glob
from concurrent.futures import ProcessPoolExecutor
import asyncio
from typing import Optional, Tuple

# --- Global Variables for Worker Processes ---
# These will hold the model instances within each worker process
_llm_instance: Optional[Llama] = None
_t5_model: Optional[T5ForConditionalGeneration] = None
_t5_tokenizer: Optional[T5Tokenizer] = None
_llm_model_path_global: Optional[str] = None # To store the path for worker init
_t5_model_name_global: Optional[str] = None # To store the name for worker init
_n_gpu_layers_global: int = 0 # GPU layers for worker init

# --- Initializer Functions for Worker Processes ---
# These functions run ONCE when each worker process starts
def _initialize_models_in_process(llm_path: str, t5_name: str, n_gpu_layers: int):
    """Loads both LLM and T5 models within a worker process."""
    global _llm_instance, _t5_model, _t5_tokenizer, _llm_model_path_global, _t5_model_name_global, _n_gpu_layers_global
    
    _llm_model_path_global = llm_path
    _t5_model_name_global = t5_name
    _n_gpu_layers_global = n_gpu_layers
    
    # Load LLM (Llama/Mistral)
    if _llm_instance is None and llm_path and os.path.exists(llm_path):
        try:
            print(f"[Worker {os.getpid()}] Loading Llama model from {llm_path}...")
            _llm_instance = Llama(
                model_path=llm_path,
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers, # Use the passed GPU layer count
                verbose=False
            )
            print(f"[Worker {os.getpid()}] Llama model loaded successfully.")
        except Exception as e:
            print(f"[Worker {os.getpid()}] ❌ Failed to load Llama model: {e}")
            _llm_instance = None # Ensure it's None on failure

    # Load T5 Summarizer
    if (_t5_model is None or _t5_tokenizer is None) and t5_name:
         try:
            print(f"[Worker {os.getpid()}] Loading T5 model ({t5_name})...")
            # Determine device - use CUDA if available IN THE WORKER
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[Worker {os.getpid()}] Using device: {device}")
            _t5_model = T5ForConditionalGeneration.from_pretrained(t5_name).to(device)
            _t5_tokenizer = T5Tokenizer.from_pretrained(t5_name)
            print(f"[Worker {os.getpid()}] T5 model loaded successfully.")
         except Exception as e:
            print(f"[Worker {os.getpid()}] ❌ Failed to load T5 model: {e}")
            _t5_model = None
            _t5_tokenizer = None

# --- Helper Functions to Access Models within Worker ---
# These are called by the actual task functions (e.g., generate_quiz_worker)
def _get_llm() -> Llama:
    """Gets the LLM instance for the current worker process."""
    if _llm_instance is None:
        # Attempt to lazy load if not initialized (e.g., if initializer failed)
        if _llm_model_path_global:
             print(f"[Worker {os.getpid()}] Lazy loading LLM...")
             _initialize_models_in_process(_llm_model_path_global, _t5_model_name_global, _n_gpu_layers_global)
        if _llm_instance is None: # Still None after trying lazy load
            raise RuntimeError(f"[Worker {os.getpid()}] LLM instance not initialized.")
    return _llm_instance

def _get_t5() -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Gets the T5 model and tokenizer for the current worker process."""
    if _t5_model is None or _t5_tokenizer is None:
        if _t5_model_name_global:
             print(f"[Worker {os.getpid()}] Lazy loading T5...")
             _initialize_models_in_process(_llm_model_path_global, _t5_model_name_global, _n_gpu_layers_global)
        if _t5_model is None or _t5_tokenizer is None: # Still None after trying
            raise RuntimeError(f"[Worker {os.getpid()}] T5 model/tokenizer not initialized.")
    # Add type assertion for clarity, though lazy load should ensure they are not None
    assert _t5_model is not None
    assert _t5_tokenizer is not None
    return _t5_model, _t5_tokenizer


# --- Executor Management ---
executor: Optional[ProcessPoolExecutor] = None

def init_executor(max_workers: int, llm_path: Optional[str], t5_name: Optional[str], n_gpu_layers: int):
    """Initializes the ProcessPoolExecutor."""
    global executor
    if executor is None and llm_path and os.path.exists(llm_path): # Only init if LLM path is valid
        print(f"Initializing ProcessPoolExecutor with {max_workers} workers.")
        try:
            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_models_in_process,
                initargs=(llm_path, t5_name, n_gpu_layers) # Pass paths/names to worker initializer
            )
            print("✅ ProcessPoolExecutor initialized.")
        except Exception as e:
             print(f"❌ FAILED to initialize ProcessPoolExecutor: {e}")
             executor = None # Ensure executor is None if init fails
    elif not llm_path or not os.path.exists(llm_path):
        print("⚠️ LLM model path invalid or not found. Executor not started. LLM features disabled.")
    else:
        print("Executor already initialized.")


def shutdown_executor():
    """Shuts down the ProcessPoolExecutor."""
    global executor
    if executor:
        print("Shutting down ProcessPoolExecutor...")
        executor.shutdown(wait=True)
        executor = None
        print("✅ ProcessPoolExecutor shut down.")

async def run_in_executor(func, *args, **kwargs):
    """Submits a function to run in the executor pool."""
    if executor is None:
        # Fallback for environments where executor might not start (e.g., missing LLM)
        # Or potentially run locally if preferred (but defeats concurrency)
        raise RuntimeError("ProcessPoolExecutor not available. Cannot process AI task.")
        # Alternatively, could try running synchronously here, but that brings back the blocking issue.
        # print("⚠️ Executor not running, attempting synchronous execution (will block).")
        # return func(*args, **kwargs) # This would run it in the main thread

    loop = asyncio.get_running_loop()
    # Use functools.partial to wrap the function call if needed, passing necessary args
    # Ensure the function called ('func') knows how to get its model instance via _get_llm() or _get_t5()
    blocking_task = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, blocking_task)

# --- Central Model Path Finding (Run Once on Startup) ---
def find_llm_model_path() -> Optional[str]:
    """Finds the largest GGUF model file in the cache."""
    print("--- Searching for Llama/Mistral GGUF model ---")
    try:
        # Check environment variable first
        env_path = os.getenv("LLAMA_MODEL_PATH")
        if env_path and os.path.exists(env_path):
             print(f"Found GGUF model via LLAMA_MODEL_PATH: {env_path}")
             return env_path
        
        # Fallback to searching cache
        huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub")
        model_files = glob.glob(os.path.join(huggingface_cache, "**", "*.gguf"), recursive=True)
        
        if not model_files:
            print("❌ No .gguf model found in Hugging Face cache or via LLAMA_MODEL_PATH.")
            return None

        # Find the largest file (heuristic for best quality)
        latest_model_path = max(model_files, key=os.path.getsize)
        print(f"Found GGUF model in cache: {latest_model_path}")
        # Optionally set the env var here if found in cache for consistency
        # os.environ["LLAMA_MODEL_PATH"] = latest_model_path
        return latest_model_path
    except Exception as e:
        print(f"❌ Error finding Llama/Mistral model: {e}")
        return None

# --- Configuration (Consider moving to a config file or env vars) ---
LLM_MODEL_PATH = find_llm_model_path() # Find path on import
T5_MODEL_NAME = "./my_final_cnn_model" # Your fine-tuned T5
# --- GPU Configuration ---
# Set n_gpu_layers based on your RTX 3050 (4GB VRAM). Start low, e.g., 20.
# -1 tries to offload all layers, likely too much for 4GB.
N_GPU_LAYERS = 20 # <<< ADJUST THIS VALUE (0 for CPU, 20-30 for RTX 3050 might work)
MAX_WORKERS = 1# Use number of CPU cores or default to 2

# Note: The global 'models' object is removed. Endpoints will call run_in_executor.
# The T5 model and tokenizer are loaded via the initializer in worker processes.

