import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_cpp import Llama
import os
import glob
from src.academic_summarizer import AcademicSummarizer

# This class will find and load all the necessary AI models when the server starts.
# All other files can import the 'models' object from this file to access them.
class ModelLoader:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Using device: {self.device} ---")

        # Load the Academic Summarizer (fine-tuned on 70% Scientific + 20% BookSum + 10% WikiHow)
        # DISABLED: Ensemble engine loads this on-demand to save memory
        print("--- Academic Summarizer will be loaded on-demand by ensemble engine ---")
        self.academic_summarizer = None
        self.custom_summary_tokenizer = None
        self.custom_summary_model = None
        print(" Memory optimized: Models load on-demand during summarization")

        # Load the Llama/Mistral GGUF model for RAG and quizzes
        print("--- Searching for and loading Llama/Mistral GGUF model ---")
        self.llm = self._find_and_load_llm()

    def _find_and_load_llm(self):
        """
        Finds and loads GGUF model with priority for Llama 3.2 3B.
        Llama 3.2 3B is 2-3x faster than Mistral 7B on CPU.
        """
        try:
            # Priority 1: Llama 3.2 3B (fastest, optimized for CPU)
            llama32_path = "./models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
            if os.path.exists(llama32_path):
                print(f" Using Llama 3.2 3B (optimized for speed): {llama32_path}")
                llm = Llama(
                    model_path=llama32_path, 
                    n_ctx=4096,
                    n_gpu_layers=0,  # CPU only for now
                    n_threads=8,
                    n_batch=512,
                    verbose=False
                )
                print(" Llama 3.2 3B loaded successfully (2-3x faster than Mistral 7B)")
                return llm
            
            # Priority 2: Fallback to any GGUF in cache
            huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub")
            model_files = glob.glob(os.path.join(huggingface_cache, "**", "*.gguf"), recursive=True)
            
            if not model_files:
                print(" No .gguf model found. Please download Llama 3.2 3B.")
                return None

            latest_model_path = max(model_files, key=os.path.getsize)
            print(f"Found GGUF model: {latest_model_path}")

            # Optimized settings for faster quiz generation
            llm = Llama(
                model_path=latest_model_path, 
                n_ctx=4096,
                n_gpu_layers=0,  # CPU only for now
                n_threads=8,
                n_batch=512,
                verbose=False
            )
            print(" Llama/Mistral model loaded successfully.")
            return llm
        except Exception as e:
            print(f" Failed to load Llama/Mistral model. Error: {e}")
            return None

# Create a single instance of the models that the rest of the app will import and use.
# This is the "singleton" pattern.
models = ModelLoader()
