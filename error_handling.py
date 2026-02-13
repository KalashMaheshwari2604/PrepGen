"""
Error handling and retry logic utilities for PrepGen AI Service
"""
import asyncio
import time
from typing import Callable, Any, Optional, TypeVar, Dict
from functools import wraps
import traceback

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class AIServiceError(Exception):
    """Base exception for AI service errors"""
    pass


class AIServiceUnavailableError(AIServiceError):
    """Raised when AI service is completely unavailable"""
    pass


class AIServiceTimeoutError(AIServiceError):
    """Raised when AI service request times out"""
    pass


class ModelNotLoadedError(AIServiceError):
    """Raised when required AI model is not loaded"""
    pass


async def retry_with_exponential_backoff(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        config: RetryConfig instance (uses default if None)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result from successful function call
        
    Raises:
        Last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            
            # Log success if it's a retry
            if attempt > 0:
                print(f"✅ Retry succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Don't retry on last attempt
            if attempt >= config.max_retries:
                print(f"❌ All {config.max_retries} retries failed")
                break
            
            # Calculate delay with exponential backoff
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                import random
                delay = delay * (0.5 + random.random())
            
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            print(f"⏳ Retrying in {delay:.2f} seconds...")
            
            await asyncio.sleep(delay)
    
    # All retries failed
    raise last_exception


def safe_execute(default_value: Any = None, log_errors: bool = True):
    """
    Decorator to safely execute functions with error handling.
    
    Args:
        default_value: Value to return if function raises exception
        log_errors: Whether to print error messages
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    print(f" Error in {func.__name__}: {str(e)}")
                    if kwargs.get('debug', False):
                        traceback.print_exc()
                return default_value
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    print(f" Error in {func.__name__}: {str(e)}")
                    if kwargs.get('debug', False):
                        traceback.print_exc()
                return default_value
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, block requests immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        
        # If circuit is OPEN, check if timeout has passed
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                print("🔄 Circuit breaker moving to HALF_OPEN state")
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise AIServiceUnavailableError(
                    f"Circuit breaker is OPEN. Service unavailable for "
                    f"{self.timeout - (time.time() - self.last_failure_time):.0f} more seconds"
                )
        
        # Try to execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection"""
        
        # If circuit is OPEN, check if timeout has passed
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout:
                print(" Circuit breaker moving to HALF_OPEN state")
                self.state = "HALF_OPEN"
                self.success_count = 0
            else:
                raise AIServiceUnavailableError(
                    f"Circuit breaker is OPEN. Service unavailable for "
                    f"{self.timeout - (time.time() - self.last_failure_time):.0f} more seconds"
                )
        
        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful request"""
        self.failure_count = 0
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                print(" Circuit breaker CLOSED - service recovered")
                self.state = "CLOSED"
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            print(" Circuit breaker OPEN - service still failing")
            self.state = "OPEN"
            self.failure_count = 0
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            print(f" Circuit breaker OPEN - {self.failure_count} failures detected")
            self.state = "OPEN"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


# Global circuit breaker for AI model operations
ai_model_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0
)
