"""
Logging and monitoring system for PrepGen AI Service
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import time
from typing import Dict, Any, Optional
from functools import wraps


class PrepGenLogger:
    """Custom logger for PrepGen with structured logging support"""
    
    def __init__(self, name: str = "prepgen", log_dir: str = "./logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging with file and console handlers"""
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler with rotation (DEBUG and above)
        log_file = os.path.join(self.log_dir, f"{self.name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Error file handler (ERROR and above)
        error_file = os.path.join(self.log_dir, f"{self.name}_errors.log")
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data"""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data"""
        self.logger.critical(self._format_message(message, kwargs))
    
    def _format_message(self, message: str, data: Dict[str, Any]) -> str:
        """Format message with structured data"""
        if not data:
            return message
        
        # Filter out None values
        filtered_data = {k: v for k, v in data.items() if v is not None}
        
        if filtered_data:
            return f"{message} | {json.dumps(filtered_data, default=str)}"
        return message


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, logger: PrepGenLogger):
        self.logger = logger
        self.metrics: Dict[str, list] = {}
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "tags": tags or {}
        }
        self.metrics[name].append(metric_data)
        
        # Keep only last 1000 metrics per name
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = [m["value"] for m in self.metrics[name]]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "last": values[-1]
        }
    
    def log_stats(self):
        """Log all metric statistics"""
        for name in self.metrics:
            stats = self.get_stats(name)
            if stats:
                self.logger.info(f"Metric: {name}", **stats)


def log_execution_time(logger: PrepGenLogger, operation_name: str):
    """Decorator to log execution time of functions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation_name} completed",
                    duration_seconds=round(duration, 2),
                    function=func.__name__
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    duration_seconds=round(duration, 2),
                    function=func.__name__,
                    error=str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation_name} completed",
                    duration_seconds=round(duration, 2),
                    function=func.__name__
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    duration_seconds=round(duration, 2),
                    function=func.__name__,
                    error=str(e)
                )
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RequestLogger:
    """Log API requests and responses"""
    
    def __init__(self, logger: PrepGenLogger):
        self.logger = logger
        self.request_count = 0
        self.error_count = 0
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log incoming API request"""
        self.request_count += 1
        self.logger.info(
            f"API Request: {method} {endpoint}",
            session_id=session_id,
            user_id=user_id,
            request_count=self.request_count
        )
    
    def log_response(
        self,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        session_id: Optional[str] = None
    ):
        """Log API response"""
        log_func = self.logger.info if status_code < 400 else self.logger.error
        
        if status_code >= 400:
            self.error_count += 1
        
        log_func(
            f"API Response: {endpoint}",
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            session_id=session_id,
            error_count=self.error_count
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get request statistics"""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": round(
                (1 - self.error_count / max(self.request_count, 1)) * 100, 2
            )
        }


# Global logger instances
prepgen_logger = PrepGenLogger("prepgen")
performance_monitor = PerformanceMonitor(prepgen_logger)
request_logger = RequestLogger(prepgen_logger)


# Convenience functions
def log_info(message: str, **kwargs):
    """Quick access to info logging"""
    prepgen_logger.info(message, **kwargs)


def log_error(message: str, **kwargs):
    """Quick access to error logging"""
    prepgen_logger.error(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Quick access to warning logging"""
    prepgen_logger.warning(message, **kwargs)
