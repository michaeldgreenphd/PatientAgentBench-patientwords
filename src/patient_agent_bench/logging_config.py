# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Centralized logging configuration for PatientAgentBench.

Provides color-coded logging with configurable log levels.
Default level is INFO to show progress during benchmark execution.
"""

import logging
import sys
from typing import Optional


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # Map log levels to colors
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_CYAN,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }
    
    # Special color for header-style INFO logs (stage announcements, etc.)
    HEADER_COLOR = Colors.BRIGHT_GREEN + Colors.BOLD
    
    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True, show_timestamps: bool = True):
        """
        Initialize the colored formatter.
        
        Args:
            fmt: Log format string
            use_colors: Whether to use colors (disable for non-TTY)
            show_timestamps: Whether format includes timestamps/logger names
        """
        super().__init__(fmt)
        self.use_colors = use_colors and sys.stderr.isatty()
        self.show_timestamps = show_timestamps
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        if self.use_colors:
            # Check if this is a header-style message (starts with STAGE, contains ===, etc.)
            is_header = (
                record.levelno == logging.INFO and 
                (record.msg.startswith('STAGE') or 
                 record.msg.startswith('=') or
                 'Checking AWS credentials' in record.msg or
                 'AWS credentials are valid' in record.msg or
                 'AWS credentials refreshed successfully' in record.msg)
            )
            
            # Get color for this level
            if is_header:
                color = self.HEADER_COLOR
            else:
                color = self.LEVEL_COLORS.get(record.levelno, "")
            
            # Format the message
            result = super().format(record)
            
            if self.show_timestamps:
                # Split into timestamp/logger and level/message parts
                # Format: "2026-01-08 11:22:16,923 - patient_agent_bench.run - INFO - message"
                parts = result.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp_logger = f"{parts[0]} - {parts[1]}"
                    level = parts[2]
                    message = parts[3]
                    
                    # Gray timestamp/logger, colored level and message
                    return (f"{Colors.BRIGHT_BLACK}{timestamp_logger}{Colors.RESET} - "
                           f"{color}{level} - {message}{Colors.RESET}")
                else:
                    # Fallback: color entire message
                    return f"{color}{result}{Colors.RESET}"
            else:
                # Simple format: just color the level and message
                # Format: "INFO - message"
                parts = result.split(' - ', 1)
                if len(parts) >= 2:
                    level = parts[0]
                    message = parts[1]
                    return f"{color}{level} - {message}{Colors.RESET}"
                else:
                    # Fallback: color entire message
                    return f"{color}{result}{Colors.RESET}"
        else:
            return super().format(record)


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    show_timestamps: Optional[bool] = None
) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
        format_string: Custom format string (uses default if None)
        show_timestamps: Whether to show timestamps and logger names.
                        Default: None (auto - True for DEBUG, False otherwise)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    
    # Auto-determine show_timestamps if not explicitly set
    if show_timestamps is None:
        show_timestamps = (numeric_level == logging.DEBUG)
    
    # Choose format based on show_timestamps
    if format_string is None:
        if show_timestamps:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(levelname)s - %(message)s"
    
    # Create formatter with colors always enabled
    formatter = ColoredFormatter(format_string, use_colors=True, show_timestamps=show_timestamps)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set level for patient_agent_bench loggers
    logging.getLogger("patient_agent_bench").setLevel(numeric_level)
    
    # Suppress verbose third-party loggers. These emit high-volume noise that
    # is unrelated to debugging the benchmark pipeline itself. Note we do NOT
    # mute the HTTP/LLM clients (httpx, anthropic, openai): their DEBUG output
    # is the full request/response body, which is genuinely useful under -v.
    logging.getLogger("langchain_aws").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # matplotlib/PIL spew per-glyph "findfont:" DEBUG lines on every chart
    # render, drowning out real logs; they carry no debugging value here.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
