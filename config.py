"""
Configuration management for PrepGen AI Service
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "PrepGen AI Service"
    app_version: str = "2.0.0"
    debug_mode: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # File Upload
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_extensions: list = [".pdf", ".docx", ".pptx", ".txt"]
    upload_dir: str = Field(default="./temp_uploads", env="UPLOAD_DIR")
    
    # Session Management
    session_timeout_minutes: int = Field(default=60, env="SESSION_TIMEOUT_MINUTES")
    cleanup_interval_minutes: int = Field(default=15, env="CLEANUP_INTERVAL_MINUTES")
    
    # Caching
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    cache_dir: str = Field(default="./cache", env="CACHE_DIR")
    cache_max_age_hours: int = Field(default=24, env="CACHE_MAX_AGE_HOURS")
    
    # AI Models
    model_device: Optional[str] = Field(default=None, env="MODEL_DEVICE")  # None = auto-detect
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL_NAME"
    )
    custom_summary_model_path: str = Field(
        default="./my_final_cnn_model",
        env="CUSTOM_SUMMARY_MODEL_PATH"
    )
    
    # Text Processing
    chunk_size: int = Field(default=768, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    
    # Quiz Generation
    default_num_questions: int = Field(default=3, env="DEFAULT_NUM_QUESTIONS")
    max_num_questions: int = Field(default=10, env="MAX_NUM_QUESTIONS")
    
    # RAG Parameters
    rag_top_k_chunks: int = Field(default=5, env="RAG_TOP_K_CHUNKS")
    rag_max_context_chars: int = Field(default=12000, env="RAG_MAX_CONTEXT_CHARS")
    
    # LLM Generation
    llm_max_tokens_summary: int = Field(default=768, env="LLM_MAX_TOKENS_SUMMARY")
    llm_max_tokens_quiz: int = Field(default=2048, env="LLM_MAX_TOKENS_QUIZ")
    llm_max_tokens_answer: int = Field(default=512, env="LLM_MAX_TOKENS_ANSWER")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # Error Handling
    enable_retry: bool = Field(default=True, env="ENABLE_RETRY")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_initial_delay: float = Field(default=1.0, env="RETRY_INITIAL_DELAY")
    retry_max_delay: float = Field(default=60.0, env="RETRY_MAX_DELAY")
    
    # Circuit Breaker
    enable_circuit_breaker: bool = Field(default=True, env="ENABLE_CIRCUIT_BREAKER")
    circuit_breaker_failure_threshold: int = Field(default=5, env="CB_FAILURE_THRESHOLD")
    circuit_breaker_timeout: float = Field(default=60.0, env="CB_TIMEOUT")
    
    # Logging
    log_dir: str = Field(default="./logs", env="LOG_DIR")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Performance
    enable_performance_monitoring: bool = Field(
        default=True, 
        env="ENABLE_PERFORMANCE_MONITORING"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def print_settings():
    """Print current settings (for debugging)"""
    print("=" * 50)
    print(f"{settings.app_name} v{settings.app_version}")
    print("=" * 50)
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Debug Mode: {settings.debug_mode}")
    print(f"Upload Directory: {settings.upload_dir}")
    print(f"Cache Directory: {settings.cache_dir}")
    print(f"Cache Enabled: {settings.enable_embedding_cache}")
    print(f"Session Timeout: {settings.session_timeout_minutes} minutes")
    print(f"Max File Size: {settings.max_file_size_mb} MB")
    print(f"Retry Enabled: {settings.enable_retry}")
    print(f"Circuit Breaker Enabled: {settings.enable_circuit_breaker}")
    print("=" * 50)
