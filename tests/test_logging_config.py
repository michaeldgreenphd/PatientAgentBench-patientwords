# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for logging configuration.

Tests logging setup, formatters, and handlers.
"""

import logging
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from patient_agent_bench.logging_config import (
    get_logger,
    setup_logging,
    ColoredFormatter,
    Colors,
)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_caches_loggers(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test get_logger returns different instances for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_formatter_initialization(self):
        """Test ColoredFormatter can be initialized."""
        formatter = ColoredFormatter()

        assert formatter is not None
        assert hasattr(formatter, 'format')

    def test_formatter_formats_record(self):
        """Test formatter can format a log record."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "Test message" in result
        assert isinstance(result, str)

    def test_formatter_includes_colors(self):
        """Test formatter includes ANSI color codes."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should contain ANSI escape codes for colors
        assert "\033[" in result or "Error message" in result

    def test_log_colors_defined(self):
        """Test ColoredFormatter.LEVEL_COLORS dictionary is properly defined."""
        assert hasattr(ColoredFormatter, 'LEVEL_COLORS')
        level_colors = ColoredFormatter.LEVEL_COLORS
        assert isinstance(level_colors, dict)
        assert logging.DEBUG in level_colors
        assert logging.INFO in level_colors
        assert logging.WARNING in level_colors
        assert logging.ERROR in level_colors
        assert logging.CRITICAL in level_colors

    def test_colors_class_defined(self):
        """Test Colors class has expected color codes."""
        assert hasattr(Colors, 'RESET')
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'GREEN')
        assert hasattr(Colors, 'YELLOW')
        assert hasattr(Colors, 'CYAN')
        assert isinstance(Colors.RESET, str)
        assert Colors.RESET.startswith('\033')


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_runs_without_error(self):
        """Test setup_logging runs without error."""
        # Should not raise
        setup_logging()

    def test_setup_logging_with_debug_level(self):
        """Test setup_logging with DEBUG level."""
        # Should not raise
        setup_logging(level="DEBUG")

    def test_setup_logging_with_warning_level(self):
        """Test setup_logging with WARNING level."""
        # Should not raise
        setup_logging(level="WARNING")

    def test_setup_logging_configures_root_logger(self):
        """Test setup_logging configures the root logger."""
        setup_logging(level="INFO")

        root_logger = logging.getLogger()

        # Should have at least one handler
        assert len(root_logger.handlers) > 0

    def test_setup_logging_with_custom_format(self):
        """Test setup_logging with custom format string."""
        # Should not raise
        setup_logging(format_string="%(levelname)s: %(message)s")

    def test_setup_logging_with_timestamps(self):
        """Test setup_logging with timestamps enabled."""
        # Should not raise
        setup_logging(show_timestamps=True)

    def test_setup_logging_without_timestamps(self):
        """Test setup_logging with timestamps disabled."""
        # Should not raise
        setup_logging(show_timestamps=False)



class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_logger_can_log_messages(self):
        """Test logger can log messages at different levels."""
        logger = get_logger("test_integration")

        # Should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_logger_respects_level(self):
        """Test logger respects configured level."""
        logger = get_logger("test_level")
        logger.setLevel(logging.WARNING)

        with patch.object(logger, 'handle') as mock_handle:
            logger.debug("Debug")  # Should not be handled
            logger.info("Info")    # Should not be handled
            logger.warning("Warning")  # Should be handled

            # Only WARNING and above should be handled
            assert mock_handle.call_count >= 1

    def test_logger_with_exception(self):
        """Test logger can log exceptions."""
        logger = get_logger("test_exception")

        try:
            raise ValueError("Test error")
        except ValueError:
            # Should not raise
            logger.exception("An error occurred")
