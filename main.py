# Load environment variables FIRST before any imports
from dotenv import load_dotenv
load_dotenv()

# Now import everything else
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
import shutil
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uuid # Used to generate unique session IDs
import asyncio
import logging

# Suppress harmless Windows ConnectionResetError warnings
import warnings
from functools import wraps

# Monkey-patch asyncio to suppress ConnectionResetError
_original_call = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost

def _silent_call_connection_lost(self, exc):
    try:
        _original_call(self, exc)
    except (ConnectionResetError, OSError):
        pass  # Silently ignore client disconnects

asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = _silent_call_connection_lost

# Initialize logging
from logger import PrepGenLogger
prepgen_logger = PrepGenLogger("prepgen", log_dir="./logs")
logger = logging.getLogger("prepgen.api")

# --- Import from your existing project files ---
from processing import extract_text
from quiz_rag_engine import generate_rag_answer, generate_quiz_questions
# from summarization_engine import summarize_with_custom_model  # Old approach - kept as backup
from ensemble_engine import summarize_with_ensemble  # NEW: Approach 3 (default)
from youtube_summarizer import get_youtube_transcript_with_yt_dlp
from ai_models import models
from session_manager import session_manager

# --- App Initialization ---
app = FastAPI(
    title="PrepGen AI API",
    description="An API for processing documents and YouTube videos to generate summaries, quizzes, and answers to questions.",
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

# --- Session-based storage REMOVED - Now using SessionManager ---
# Old: document_store = {}
# New: Using session_manager from session_manager.py

# --- Pydantic Models ---

# These request models now require the session_id to be sent explicitly
class SessionRequest(BaseModel):
    session_id: str

class AskRequest(SessionRequest):
    question: str

# The upload response will now return the session_id
class UploadResponse(BaseModel):
    message: str
    filename: str
    session_id: str
    character_count: int

# (Other response models remain the same)
class AskResponse(BaseModel):
    question: str
    answer: str
    source: str
class SummaryResponse(BaseModel):
    source: str
    summary: str
    extraction_time: float  # Time for T5 models
    merge_time: float       # Time for Llama
    total_time: float       # Total time
    word_count: int         # Final word count

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str  # "easy", "medium", or "hard"
class QuizResponse(BaseModel):
    filename: str
    quiz: List[QuizQuestion]
class YouTubeRequest(BaseModel):
    url: HttpUrl

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Initialize AI models and start background cleanup task"""
    if models.llm:
        logger.info("AI models ready (summarization models load on-demand)")
        logger.info("Server is ready to accept requests")
    else:
        logger.warning("Llama model failed to load - quiz generation may not work")
    
    # Start background session cleanup task
    asyncio.create_task(periodic_session_cleanup())


async def periodic_session_cleanup():
    """Background task to cleanup expired sessions every 15 minutes"""
    while True:
        await asyncio.sleep(900)  # 15 minutes
        cleaned = session_manager.cleanup_expired_sessions()
        if cleaned > 0:
            logger.info("Cleaned up %d expired sessions", cleaned)


@app.get("/health", status_code=200)
async def health_check():
    """A simple endpoint to confirm the API is running correctly."""
    return {
        "status": "ok",
        "ai_models_loaded": models.llm is not None and models.custom_summary_model is not None,
        "active_sessions": session_manager.get_session_count()
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a document, extract its text, and return a unique session_id.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.pptx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (50MB max)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    
    upload_dir = "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        # Save file and check size
        file_size = 0
        with open(file_path, "wb") as buffer:
            chunk = await file.read(8192)  # Read in 8KB chunks
            while chunk:
                file_size += len(chunk)
                if file_size > max_size:
                    raise HTTPException(
                        status_code=400, 
                        detail="File size exceeds 50MB limit"
                    )
                buffer.write(chunk)
                chunk = await file.read(8192)
        
        # Extract text
        text = extract_text(file_path)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")
        
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Store in session manager (pickle-safe)
        session_manager.create_session(session_id, text, file.filename)
        
        logger.info("Created session %s for %s (%d characters)", session_id, file.filename, len(text))
        
        return {
            "message": "File processed successfully. Use the session_id for other actions.",
            "filename": file.filename,
            "session_id": session_id,
            "character_count": len(text)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question using a specific session_id.
    Now includes conversation history for context awareness.
    """
    session_data = session_manager.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload the document again.")
    
    # Initialize conversation history if not exists
    if "conversation_history" not in session_data:
        session_data["conversation_history"] = []
    
    # Get answer with conversation context
    answer, source = generate_rag_answer(
        request.question, 
        session_data["text"], 
        llm=models.llm,
        conversation_history=session_data["conversation_history"]
    )
    
    # Store this exchange in history
    session_data["conversation_history"].append({
        "question": request.question,
        "answer": answer,
        "source": source
    })
    
    # Keep only last 5 exchanges to manage memory
    if len(session_data["conversation_history"]) > 5:
        session_data["conversation_history"] = session_data["conversation_history"][-5:]
    
    # Session is already updated in memory (get_session returns the actual dict, not a copy)
    # No need to reassign
    
    return {"question": request.question, "answer": answer, "source": source}

@app.post("/summarize", response_model=SummaryResponse)
async def get_summary_for_document(request: SessionRequest):
    """
    Generate a comprehensive summary using Ensemble Approach 3:
    - Extracts summaries from 4 fine-tuned T5 models (Academic, CNN, SAMSum, XSum)
    - Merges all perspectives into one combined text
    - Uses Llama 3.2 3B to intelligently merge and expand
    
    This produces HIGH QUALITY comprehensive summaries (~5 minutes processing).
    Best for: Academic documents, research papers, study materials.
    
    Metrics returned:
    - extraction_time: Time for T5 models (~30s)
    - merge_time: Time for Llama merge (~80s)
    - total_time: Total processing time (~5 min)
    - word_count: Final summary word count (typically 400-700 words)
    """
    session_data = session_manager.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload the document again.")
    
    try:
        logger.info("Starting ensemble summarization for session %s", request.session_id)
        
        # Run Ensemble Approach 3
        result = summarize_with_ensemble(session_data["text"])
        
        logger.info("Ensemble summarization completed: %d words in %.2fs", 
                   result['word_count'], result['total_time'])
        
        return {
            "source": session_data["filename"],
            "summary": result['summary'],
            "extraction_time": result['extraction_time'],
            "merge_time": result['merge_time'],
            "total_time": result['total_time'],
            "word_count": result['word_count']
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error("Ensemble summarization error: %s", str(e))
        logger.error("Traceback: %s", error_details)
        raise HTTPException(status_code=500, detail=f"Ensemble summarization failed: {str(e)}")


@app.post("/quiz", response_model=QuizResponse)
async def get_quiz_for_document(request: SessionRequest):
    """
    Generate a quiz for the document associated with a session_id.
    Number of questions is dynamically calculated based on document size (max 20).
    """
    session_data = session_manager.get_session(request.session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload the document again.")
    
    # Calculate number of questions based on document size
    # 1 question per ~1000 characters, minimum 3, maximum 20
    text_length = len(session_data["text"])
    num_questions = min(20, max(3, text_length // 1000))
    
    logger.info("Generating %d questions for document with %d characters", num_questions, text_length)
    
    new_questions = generate_quiz_questions(
        session_data["text"], 
        num_questions=num_questions, 
        llm=models.llm,
        previously_asked=session_data["asked_quiz_questions"]
    )
    
    if not new_questions or (isinstance(new_questions, list) and "error" in new_questions[0]):
         raise HTTPException(status_code=500, detail="Failed to generate new quiz questions.")
    
    # Add new questions to the session's history
    for q in new_questions:
        if "question" in q and q["question"] != "Quiz Complete!":
            session_manager.add_quiz_question(request.session_id, q["question"])
    
    return {"filename": session_data["filename"], "quiz": new_questions}

# The YouTube summarizer is stateless, so it doesn't need a session_id
@app.post("/summarize-youtube", response_model=SummaryResponse)
async def summarize_youtube_video(request: YouTubeRequest):
    try:
        transcript = get_youtube_transcript_with_yt_dlp(str(request.url))
        if not transcript:
            raise HTTPException(status_code=404, detail="Could not retrieve a transcript.")
        
        # Use Ensemble Approach 3 for comprehensive YouTube summaries
        result = summarize_with_ensemble(transcript)
        
        return {
            "source": f"YouTube: {request.url}",
            "summary": result['summary'],
            "extraction_time": result['extraction_time'],
            "merge_time": result['merge_time'],
            "total_time": result['total_time'],
            "word_count": result['word_count']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("YouTube summarization error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- Admin/Debug Endpoints ---

@app.get("/sessions/stats")
async def get_session_stats():
    """Get statistics about active sessions (for debugging/monitoring)"""
    return session_manager.get_stats()


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Manually delete a session"""
    if session_manager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    # Groq API verification removed for cleaner output
    pass
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep-alive
        timeout_graceful_shutdown=30  # 30 seconds for graceful shutdown
    )



