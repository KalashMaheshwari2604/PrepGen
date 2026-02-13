"""
Session Manager for PrepGen AI Service
Handles session storage with proper serialization to fix pickle errors
"""
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading


class SessionManager:
    """Thread-safe session manager with automatic expiration"""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.session_timeout = session_timeout_minutes * 60  # Convert to seconds
        
    def create_session(self, session_id: str, text: str, filename: str) -> None:
        """
        Create a new session with serializable data only.
        
        Args:
            session_id: Unique session identifier
            text: Extracted document text
            filename: Original filename
        """
        with self._lock:
            self._sessions[session_id] = {
                "text": text,
                "filename": filename,
                "asked_quiz_questions": [],
                "created_at": time.time(),
                "last_accessed": time.time(),
                # Store embeddings and FAISS index metadata separately
                "has_embeddings": False,
                "chunk_count": 0
            }
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data and update last accessed time.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dict or None if not found/expired
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            
            # Check if session has expired
            if time.time() - session["last_accessed"] > self.session_timeout:
                del self._sessions[session_id]
                return None
            
            # Update last accessed time
            session["last_accessed"] = time.time()
            return session
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session.update(updates)
            session["last_accessed"] = time.time()
            return True
    
    def add_quiz_question(self, session_id: str, question: str) -> bool:
        """Add a quiz question to the asked questions list"""
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session["asked_quiz_questions"].append(question)
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            current_time = time.time()
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if current_time - session["last_accessed"] > self.session_timeout
            ]
            
            for sid in expired_sessions:
                del self._sessions[sid]
            
            return len(expired_sessions)
    
    def get_session_count(self) -> int:
        """Get the number of active sessions"""
        with self._lock:
            return len(self._sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "session_ids": list(self._sessions.keys()),
                "oldest_session": min(
                    (s["created_at"] for s in self._sessions.values()),
                    default=None
                ),
                "newest_session": max(
                    (s["created_at"] for s in self._sessions.values()),
                    default=None
                )
            }


# Global session manager instance
session_manager = SessionManager(session_timeout_minutes=60)
