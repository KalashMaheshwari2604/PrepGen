import os
from llama_cpp import Llama
import json
from serpapi import GoogleSearch
from typing import List, Dict, Any, Tuple
# Note: This file should not import `ai_models` directly to avoid circular dependencies
# Instead, the worker functions will implicitly use the Llama instance in their own process.

# --- Helper function to prevent context overflow ---
def truncate_text(text: str, max_chars: int = 15000) -> str:
    """Limits text to a max number of characters for LLM prompts."""
    if len(text) > max_chars:
        print(f"--- Warning: Context truncated from {len(text)} to {max_chars} chars. ---")
        return text[:max_chars]
    return text

# --- Internal function to get the LLM instance for a worker process ---
# This function must be defined here so the worker can access it.
def _get_llm_instance_for_worker() -> Llama:
    # This relies on ai_models._llm_instance being set in the worker's global scope
    # by the ProcessPoolExecutor's initializer.
    from ai_models import _llm_instance # Import dynamically to avoid circular issues
    if _llm_instance is None:
        raise RuntimeError("LLM instance not initialized in worker process.")
    return _llm_instance

# --- 1. Web Search Function (Unchanged, not using LLM directly) ---
def search_web_with_serpapi(question: str) -> str:
    print("--- Performing web search fallback ---")
    try:
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            return "Web search not configured. Set SERPAPI_API_KEY env variable."
        params = {"api_key": serpapi_key, "q": question, "engine": "google"}
        search = GoogleSearch(params)
        results = search.get_dict()
        answer = results.get("answer_box", {}).get("snippet") or \
                 results.get("organic_results", [{}])[0].get("snippet") or \
                 "Could not find a direct answer on the web."
        return answer
    except Exception as e:
        print(f"Error during web search: {e}")
        return "An error occurred while searching the web."

# --- 2. RAG Chatbot Worker Function ---
def generate_rag_answer_worker(question: str, context: str) -> Tuple[str, str]:
    """
    Worker function for RAG that gets the LLM instance from its own process scope.
    """
    try:
        llm = _get_llm_instance_for_worker()
    except RuntimeError as e:
        return f"Error: {e}", "error" # LLM not loaded in worker
    
    safe_context = truncate_text(context) 

    try:
        critique_prompt = f"""
[INST]
Based *only* on the context below, can you answer the following question? Answer only with a single word: YES or NO.
Context: "{safe_context}"
Question: "{question}"
[/INST]
"""
        critique_output = llm(critique_prompt, max_tokens=10, echo=False)
        decision = critique_output['choices'][0]['text'].strip().upper()

        if "YES" in decision:
            print("--- Answer found in document context. Generating... ---")
            answer_prompt = f"""
[INST]
Based only on the context provided, please answer the following question.
Context: {safe_context}
Question: {question}
[/INST]
"""
            answer_output = llm(answer_prompt, max_tokens=512, echo=False)
            answer = answer_output['choices'][0]['text'].strip()
            return answer, "document"
        else:
            print("--- Context is insufficient. Switching to web search. ---")
            web_answer = search_web_with_serpapi(question)
            final_answer = f"{web_answer}\n\n(Note: This result is from a web search and was not found in the provided document.)"
            return final_answer, "web"
    except ValueError as e:
         if "Requested tokens" in str(e) and "exceed context window" in str(e):
              print(f"--- ERROR: Context window overflow during RAG critique/answer generation: {e} ---")
              return "Error: The document content is too large for the AI model's context window, even after truncation.", "error"
         else:
              print(f"--- An unexpected ValueError occurred during RAG: {e} ---")
              return f"An unexpected error occurred during RAG generation: {str(e)}", "error" # Return error string
    except Exception as e:
         print(f"--- An unexpected error occurred during RAG generation: {e} ---")
         return f"An unexpected error occurred during RAG generation: {str(e)}", "error"


# --- 3. Quiz Generation Worker Function ---
def generate_quiz_questions_for_chunk_worker(context_chunk: str, num_questions: int) -> List[Dict[str, Any]]:
    """
    Worker function for quiz generation from a single text chunk.
    It gets the LLM instance from its own process scope.
    """
    # Inside generate_quiz_questions_for_chunk_worker, within the try block:
    try:
        llm = _get_llm_instance_for_worker()
        print(f"[Worker {os.getpid()}] Got LLM instance for chunk.") # <-- Add this
        safe_context = truncate_text(context_chunk)
        # ... (prompt definition) ...
        print(f"[Worker {os.getpid()}] Making LLM call for quiz chunk...") # <-- Add this
        output = llm(prompt, max_tokens=1024, echo=False)
        print(f"[Worker {os.getpid()}] LLM call completed for quiz chunk.") # <-- Add this
        raw_quiz_json = output['choices'][0]['text']
        # ... (rest of the function) ...
    except Exception as e:
        print(f"[Worker {os.getpid()}] EXCEPTION during quiz chunk generation: {e}") # <-- Add process ID here
        # ... (rest of except block) ...
        safe_context = truncate_text(context_chunk)
        
    prompt = f"""
[INST]
You are a quiz generation expert. Based ONLY on the context chunk provided below, generate exactly {num_questions} unique multiple-choice questions.
For each question, provide:
1. The question text.
2. A list of 4 options.
3. The correct answer text.
4. A brief explanation for why the answer is correct based on the context chunk.

Format your entire response as a single, valid JSON array of objects. Each object must have the keys "question", "options", "correct_answer", and "explanation".

Context Chunk:
"{safe_context}"
[/INST]
"""
    
    for attempt in range(2):
        print(f"Attempt {attempt + 1} for chunk...")
        try:
            output = llm(prompt, max_tokens=1024, echo=False) 
            raw_quiz_json = output['choices'][0]['text']

            clean_json_string = raw_quiz_json.strip().replace('```json', '').replace('```', '')
            parsed_quiz = json.loads(clean_json_string)
            
            if isinstance(parsed_quiz, list) and len(parsed_quiz) > 0:
                validated_quiz = []
                for q in parsed_quiz:
                    if all(k in q for k in ["question", "options", "correct_answer", "explanation"]) and \
                       isinstance(q.get("options"), list) and len(q["options"]) == 4:
                        if isinstance(q.get("correct_answer"), dict):
                            q["correct_answer"] = q["correct_answer"].get("text", "")
                        validated_quiz.append(q)
                
                if validated_quiz:
                    return validated_quiz 
                else:
                    print(f"--- Warning: Model output structure invalid on attempt {attempt + 1}. Retrying. ---")
                    continue 

        except ValueError as e:
            if "Requested tokens" in str(e) and "exceed context window" in str(e):
                 print(f"--- ERROR: Context window overflow for chunk on attempt {attempt + 1}: {e} ---")
                 return [{"error": f"Context chunk too large/dense for model: {e}"}]
            else:
                 print(f"--- An unexpected ValueError occurred during quiz generation: {e} ---")
                 return [{"error": f"An unexpected ValueError during quiz generation: {str(e)}"}]
        except json.JSONDecodeError as e:
            print(f"--- Warning: Model did not return valid JSON on attempt {attempt + 1}. Error: {e} ---")
            continue 
        except Exception as e:
            print(f"--- An unexpected error occurred during quiz generation for chunk (Attempt {attempt+1}): {e} ---")
            break 

    print("--- Failed to generate questions for this chunk after multiple attempts. ---")
    return [{"error": "Failed to generate quiz questions for this chunk."}]
