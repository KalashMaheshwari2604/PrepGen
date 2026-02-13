from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
import shutil
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import uuid
import math
import asyncio
import json

# --- Import from your existing project files ---
from processing import extract_text, chunk_text
from test_mistral import generate_rag_answer_worker, generate_quiz_questions_for_chunk_worker
from test_trained_model import summarize_with_custom_model_worker
from youtube_summarizer import get_youtube_transcript_with_yt_dlp
from ai_models import (
    init_executor,
    shutdown_executor,
    run_in_executor,
    LLM_MODEL_PATH,
    T5_MODEL_NAME,
    N_GPU_LAYERS,
    MAX_WORKERS
)

# --- App Initialization ---
app = FastAPI(
    title="PrepGen AI API",
    description="An API for processing documents and YouTube videos.",
    version="1.0.0"
)

# --- CORS MIDDLEWARE ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session-based storage ---
document_store: Dict[str, Dict[str, Any]] = {}

# --- Constants ---
QUESTIONS_PER_PART = 15
QUESTIONS_PER_CHUNK = 2

# --- Pydantic Models (Unchanged) ---
class SessionRequest(BaseModel): session_id: str
class AskRequest(SessionRequest): question: str
class UploadResponse(BaseModel): message: str; filename: str; session_id: str; character_count: int
class AskResponse(BaseModel): question: str; answer: str; source: str
class SummaryResponse(BaseModel): source: str; summary: str
class QuizQuestion(BaseModel): question: str; options: List[str]; correct_answer: str; explanation: str
class QuizResponse(BaseModel): filename: str; quiz_part: List[QuizQuestion]; current_part: int; total_parts: int
class YouTubeRequest(BaseModel): url: HttpUrl

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    init_executor(
        max_workers=MAX_WORKERS,
        llm_path=LLM_MODEL_PATH,
        t5_name=T5_MODEL_NAME,
        n_gpu_layers=N_GPU_LAYERS
    )
    print("✅ FastAPI application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    shutdown_executor()
    print("✅ FastAPI application shutdown complete.")


@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # (Upload logic remains the same)
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text = extract_text(file_path)
        if not text: raise HTTPException(status_code=400, detail="Could not extract text.")
        session_id = str(uuid.uuid4())
        document_store[session_id] = {
            "text": text, "filename": file.filename, "full_quiz": None,
            "quiz_parts": [], "total_quiz_parts": 0, "quiz_generating": False
        }
        return {"message": "File processed.", "filename": file.filename,
                "session_id": session_id, "character_count": len(text)}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Upload error: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    session_data = document_store.get(request.session_id)
    if not session_data: raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        answer, source = await run_in_executor(
            generate_rag_answer_worker,
            request.question,
            session_data["text"]
        )
        if source == "error": raise HTTPException(status_code=500, detail=f"AI RAG failed: {answer}")
        return {"question": request.question, "answer": answer, "source": source}
    except RuntimeError as e: raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")


@app.post("/summarize", response_model=SummaryResponse)
async def get_summary_for_document(request: SessionRequest):
    session_data = document_store.get(request.session_id)
    if not session_data: raise HTTPException(status_code=404, detail="Session not found.")
    
    try:
        summary = await run_in_executor(
            summarize_with_custom_model_worker,
            session_data["text"]
        )
        if summary.startswith("Error:"): raise HTTPException(status_code=500, detail=f"AI summarization failed: {summary}")
        return {"source": session_data["filename"], "summary": summary}
    except RuntimeError as e: raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")

# --- FULL QUIZ GENERATION LOGIC (Runs in Background) ---
async def _generate_and_store_full_quiz_background(session_id: str):
    session_data = document_store.get(session_id)
    if not session_data or not session_data.get("quiz_generating"): return # Check flag

    print(f"--- Starting background full quiz generation for session {session_id} ---")
    full_quiz: List[Dict[str, Any]] = []
    text_chunks = chunk_text(session_data["text"], chunk_size=1500, chunk_overlap=150)

    if not text_chunks:
        print(f"Error: Doc text empty/short for session {session_id}.")
        session_data["quiz_generating"] = False; return

    chunk_tasks = [
        run_in_executor(generate_quiz_questions_for_chunk_worker, chunk, QUESTIONS_PER_CHUNK)
        for chunk in text_chunks
    ]
    
    try:
        results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        for i, questions_from_chunk in enumerate(results):
            if isinstance(questions_from_chunk, Exception):
                print(f"Error generating quiz for chunk {i+1} (Sess:{session_id}): {questions_from_chunk}")
                continue
            
            if isinstance(questions_from_chunk, list) and questions_from_chunk:
                valid_questions = [q for q in questions_from_chunk if isinstance(q, dict) and "error" not in q]
                full_quiz.extend(valid_questions)
            else:
                 print(f"Warning: No valid questions for chunk {i+1} (Sess:{session_id}).")

    except RuntimeError as e:
        print(f"CRITICAL ERROR during quiz generation (Executor Unavailable?) for session {session_id}: {e}")
        session_data["full_quiz"] = [{"error": "Executor failed"}] # Indicate failure
    except Exception as e:
        print(f"CRITICAL ERROR during full quiz generation for session {session_id}: {e}")
        session_data["full_quiz"] = [{"error": "Generation failed"}] # Indicate failure
    
    if not full_quiz and "full_quiz" not in session_data: # If no questions and no error flagged
        print(f"Error: Failed to generate ANY quiz questions for session {session_id}.")
        session_data["full_quiz"] = [] # Mark as generated but empty

    elif isinstance(session_data.get("full_quiz"), list): # Only process if no critical error occurred
        # Paginate and store
        quiz_parts = [full_quiz[i:i + QUESTIONS_PER_PART] for i in range(0, len(full_quiz), QUESTIONS_PER_PART)]
        session_data["full_quiz"] = full_quiz
        session_data["quiz_parts"] = quiz_parts
        session_data["total_quiz_parts"] = len(quiz_parts)
        print(f"--- Full quiz generated (Sess:{session_id}): {len(full_quiz)} questions in {len(quiz_parts)} parts ---")

    session_data["quiz_generating"] = False # Mark as complete

@app.post("/quiz", response_model=QuizResponse, status_code=200)
async def get_quiz_part(request: SessionRequest, part: int = 1, background_tasks: BackgroundTasks = BackgroundTasks()):
    session_data = document_store.get(request.session_id)
    if not session_data: raise HTTPException(status_code=404, detail="Session not found.")

    if session_data.get("quiz_generating"):
        # --- FIX: Use json.dumps ---
        return Response(status_code=202, content=json.dumps({"detail": "Quiz generation is in progress. Please try again shortly."}), media_type="application/json")


    if session_data.get("full_quiz") is None:
        session_data["quiz_generating"] = True
        background_tasks.add_task(_generate_and_store_full_quiz_background, request.session_id)
        # --- FIX: Use json.dumps ---
        return Response(status_code=202, content=json.dumps({"detail": "Quiz generation initiated. Please request the quiz again in a few moments."}), media_type="application/json")
        
    total_parts = session_data.get("total_quiz_parts", 0)
    quiz_parts = session_data.get("quiz_parts", [])

    # Check if generation completed but resulted in an error state stored in full_quiz
    if isinstance(session_data.get("full_quiz"), list) and session_data["full_quiz"] and "error" in session_data["full_quiz"][0]:
         error_msg = session_data["full_quiz"][0]["error"]
         raise HTTPException(status_code=500, detail=f"Quiz generation failed: {error_msg}")

    if total_parts == 0 or not quiz_parts:
         if isinstance(session_data.get("full_quiz"), list): # Check if generation completed (even if empty)
              return {"filename": session_data["filename"], "quiz_part": [], "current_part": 0, "total_parts": 0}
         else:
             raise HTTPException(status_code=500, detail="Quiz generation failed or is inconsistent.")

    if not (1 <= part <= total_parts):
        raise HTTPException(status_code=400, detail=f"Invalid part number. Available: 1-{total_parts}.")

    return {"filename": session_data["filename"], "quiz_part": quiz_parts[part - 1],
            "current_part": part, "total_parts": total_parts}

@app.post("/summarize-youtube", response_model=SummaryResponse)
async def summarize_youtube_video(request: YouTubeRequest):
    try:
        transcript = get_youtube_transcript_with_yt_dlp(str(request.url))
        if not transcript: raise HTTPException(status_code=404, detail="Could not retrieve transcript.")
        
        summary = await run_in_executor(summarize_with_custom_model_worker, transcript)
        if summary.startswith("Error:"): raise HTTPException(status_code=500, detail=f"AI summarization failed: {summary}")
        return {"source": f"YouTube: {request.url}", "summary": summary}
    except RuntimeError as e: raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"YouTube summarization error: {str(e)}")

if __name__ == "__main__":
    if not LLM_MODEL_PATH:
         print("\n\n" + "="*60); print("⚠️ WARNING: LLM_MODEL_PATH could not be automatically found."); print("   Please set the LLAMA_MODEL_PATH environment variable."); print("="*60 + "\n\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

