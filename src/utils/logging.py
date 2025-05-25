"""Logging utilities for the project."""
from __future__ import annotations
import logging
import sys
from typing import Any, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

class StructuredLogger:
    """Logger that supports both text and structured JSON logging.
    
    Attributes:
        name: Logger name
        logger: Python logger instance
    """

    def __init__(self, name: str):
        """Initialize logger.
        
        Args:
            name: Logger name (usually __name__)
        """
        self.name = name
        self.logger = logging.getLogger(name)

    def _format_json(self, level: str, msg: str, **kwargs: Any) -> str:
        """Format log entry as JSON.
        
        Args:
            level: Log level
            msg: Log message
            **kwargs: Additional fields to log
            
        Returns:
            JSON string
        """
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "name": self.name,
            "message": msg,
            **kwargs
        }
        return json.dumps(data)

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message.
        
        Args:
            msg: Message to log
            **kwargs: Additional fields to include
        """
        self.logger.debug(self._format_json("DEBUG", msg, **kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message.
        
        Args:
            msg: Message to log
            **kwargs: Additional fields to include
        """
        self.logger.info(self._format_json("INFO", msg, **kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message.
        
        Args:
            msg: Message to log
            **kwargs: Additional fields to include
        """
        self.logger.warning(self._format_json("WARNING", msg, **kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message.
        
        Args:
            msg: Message to log
            **kwargs: Additional fields to include
        """
        self.logger.error(self._format_json("ERROR", msg, **kwargs))

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log exception with traceback.
        
        Args:
            msg: Message to log
            **kwargs: Additional fields to include
        """
        self.logger.exception(self._format_json("ERROR", msg, **kwargs))

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> None:
    """Configure logging for the project.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        json_format: Whether to use JSON formatting
    """
    # Create formatter
    if json_format:
        formatter = logging.Formatter('%(message)s')  # Raw JSON
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Configure handlers
    handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers and add new ones
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return StructuredLogger(name) 