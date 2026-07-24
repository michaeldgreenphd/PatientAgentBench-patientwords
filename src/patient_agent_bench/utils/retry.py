# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Retry handling with exponential backoff and jitter.

Provides robust retry mechanisms for LLM calls that may fail due to
throttling or transient errors. Also handles expired AWS credentials
by triggering a credential refresh before retrying.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Tuple, Type, TypeVar

from botocore.exceptions import ClientError, NoCredentialsError

from patient_agent_bench.config import ensure_credentials
from patient_agent_bench.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


from openai import APIConnectionError as OpenAIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError
from openai import RateLimitError as OpenAIRateLimitError


class ThrottlingException(Exception):
    """Exception raised when AWS throttles requests."""


class ServiceUnavailableException(Exception):
    """Exception raised when AWS service is unavailable."""


def is_expired_token_error(error: Exception) -> bool:
    """
    Check if an exception is an expired token/credentials error.

    These errors require credential refresh (via the credential hook) before retrying,
    not just a simple backoff.
    """
    # NoCredentialsError means credentials are completely missing
    if isinstance(error, NoCredentialsError):
        return True

    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "")
        return error_code in {
            "ExpiredTokenException",
            "ExpiredToken",
            "RequestExpired",
            "UnrecognizedClientException",
        }

    error_str = str(error)
    # Bedrock/boto surface expired creds as ExpiredToken*. The Bedrock Mantle
    # gateway (SigV4 behind an OpenAI-protocol client) instead surfaces a 401
    # AuthenticationError whose body reads "The security token included in the
    # request is expired" — the AWS SigV4 phrasing, distinct from a plain
    # OpenAI/LiteLLM bad-key 401 ("Incorrect API key provided"). Matching only
    # the SigV4 phrase means Mantle (auth="sigv4") triggers a credential refresh
    # — which recovers it, since _MantleSigV4Auth re-reads credentials per request
    # — WITHOUT futilely refreshing for api_key providers (OpenAI/LiteLLM) whose
    # static key a credential refresh cannot fix.
    return (
        "ExpiredTokenException" in error_str
        or "ExpiredToken" in error_str
        or "Unable to locate credentials" in error_str
        or "security token included in the request is expired" in error_str
    )


def is_throttling_error(error: Exception) -> bool:
    """
    Check if an exception is a throttling error (transient, worth retrying).

    Handles botocore ClientError with throttling codes, HTTP 429 status codes,
    and OpenAI API rate limit / transient errors.

    Note: ValidationException and parameter validation errors are NOT retryable -
    they indicate a permanent issue with the request format (e.g., unsupported
    message content, malformed tool calls).

    Note: ExpiredTokenException is NOT a throttling error - it requires credential
    refresh and is handled separately by the retry loop.
    """
    if isinstance(error, (ThrottlingException, ServiceUnavailableException)):
        return True

    # OpenAI rate limit and transient errors are retryable
    if isinstance(error, OpenAIRateLimitError):
        return True
    if isinstance(error, OpenAIConnectionError):
        return True
    if isinstance(error, OpenAITimeoutError):
        return True
    if isinstance(error, OpenAIAPIError):
        # 5xx server errors are retryable, 4xx (except 429) are not
        status = getattr(error, "status_code", None)
        if status and status >= 500:
            return True

    # Expired token errors are not throttling - handled separately
    if is_expired_token_error(error):
        return False

    error_str = str(error)

    # Check for validation errors in error string (catches wrapped exceptions)
    # These are NOT retryable - they indicate permanent request format issues
    validation_patterns = [
        "ValidationException",
        "Parameter validation failed",
        "Invalid length for parameter",
    ]
    for pattern in validation_patterns:
        if pattern in error_str:
            logger.error(
                "Validation error detected (not retryable): %s",
                error_str[:200],
            )
            return False

    if isinstance(error, ClientError):
        error_code = error.response.get("Error", {}).get("Code", "")

        # Non-retryable error codes
        non_retryable_codes = {
            "ValidationException",
            "InvalidParameterException",
            "InvalidRequestException",
        }
        if error_code in non_retryable_codes:
            return False

        throttling_codes = {
            "ThrottlingException",
            "Throttling",
            "TooManyRequestsException",
            "ProvisionedThroughputExceededException",
            "RequestLimitExceeded",
            "ServiceUnavailable",
            "ModelStreamErrorException",
        }
        return error_code in throttling_codes

    # Check for HTTP 429 in the error message or attributes
    error_str_lower = error_str.lower()
    if "429" in error_str_lower or "throttl" in error_str_lower or "rate limit" in error_str_lower:
        return True

    return False


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        jitter_factor: Factor for random jitter, delay * random(1-jitter, 1+jitter) (default: 0.5)
        retryable_exceptions: Tuple of exception types that trigger retry
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.5
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: _build_retryable_exceptions()
    )


def _build_retryable_exceptions() -> Tuple[Type[Exception], ...]:
    """Build the default tuple of retryable exception types."""
    return (
        ThrottlingException,
        ServiceUnavailableException,
        ClientError,
        NoCredentialsError,
        OpenAIRateLimitError,
        OpenAIAPIError,
        OpenAIConnectionError,
        OpenAITimeoutError,
    )


def calculate_delay_with_jitter(attempt: int, config: RetryConfig) -> float:
    """
    Calculate retry delay with exponential backoff and jitter.

    Formula: min(base_delay * 2^attempt, max_delay) * random(1-jitter, 1+jitter)
    """
    base_exp_delay = config.base_delay * (2**attempt)
    capped_delay = min(base_exp_delay, config.max_delay)
    jitter_multiplier = random.uniform(1 - config.jitter_factor, 1 + config.jitter_factor)
    return float(max(0.0, capped_delay * jitter_multiplier))


def _should_retry(error: Exception, config: RetryConfig, attempt: int = 0) -> bool:
    """Check if an exception should trigger a retry."""
    # Expired token errors are retryable (after credential refresh)
    if is_expired_token_error(error):
        return True
    # Connection errors: retry only once
    if isinstance(error, OpenAIConnectionError):
        return attempt < 1
    is_throttle = is_throttling_error(error)
    return is_throttle or (
        isinstance(error, config.retryable_exceptions)
        and "ValidationException" not in str(error)
    )


def _log_retry(attempt: int, config: RetryConfig, error: Exception, delay: float) -> None:
    """Log a retry attempt."""
    logger.warning(
        "Attempt %d/%d failed with %s: %s. Retrying in %.2fs",
        attempt + 1,
        config.max_retries + 1,
        type(error).__name__,
        str(error)[:100],
        delay,
    )


def _log_exhausted(config: RetryConfig, func_name: str, error: Exception) -> None:
    """Log when all retries are exhausted."""
    logger.error(
        "All %d attempts failed for %s. Last error: %s",
        config.max_retries + 1,
        func_name,
        str(error)[:200],
    )


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args,
    config: RetryConfig | None = None,
    on_credential_refresh: Callable[[], None] | None = None,
    **kwargs,
) -> T:
    """
    Execute an async function with retry and exponential backoff.

    Retries the function when it raises a retryable exception (e.g., throttling).
    Uses exponential backoff with jitter to avoid thundering herd problems.

    For expired token errors, triggers a credential refresh via ensure_credentials()
    before retrying. If on_credential_refresh is provided, it is called after
    successful credential refresh so callers can recreate clients/LLMs that hold
    stale session tokens.
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if not _should_retry(e, config, attempt):
                raise

            last_exception = e

            # If expired token, refresh credentials before retrying
            if is_expired_token_error(e):
                logger.warning(
                    "Expired token detected (attempt %d/%d): %s. "
                    "Refreshing credentials...",
                    attempt + 1,
                    config.max_retries + 1,
                    str(e)[:100],
                )
                if not ensure_credentials():
                    logger.error("Failed to refresh credentials, giving up")
                    raise

                # Let caller recreate clients with fresh credentials
                if on_credential_refresh is not None:
                    logger.debug("Invoking on_credential_refresh callback")
                    on_credential_refresh()

            if attempt < config.max_retries:
                delay = calculate_delay_with_jitter(attempt, config)
                _log_retry(attempt, config, e, delay)
                await asyncio.sleep(delay)
            else:
                _log_exhausted(config, func.__name__, e)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception captured after retries")


def retry_sync_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig | None = None,
    on_credential_refresh: Callable[[], None] | None = None,
    **kwargs,
) -> T:
    """
    Execute a sync function with retry and exponential backoff.

    Synchronous version of retry_with_backoff for use in non-async contexts
    like agent invoke() methods.

    For expired token errors, triggers a credential refresh via ensure_credentials()
    before retrying. If on_credential_refresh is provided, it is called after
    successful credential refresh so callers can recreate clients/LLMs that hold
    stale session tokens.
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _should_retry(e, config, attempt):
                raise

            last_exception = e

            # If expired token, refresh credentials before retrying
            if is_expired_token_error(e):
                logger.warning(
                    "Expired token detected (attempt %d/%d): %s. "
                    "Refreshing credentials...",
                    attempt + 1,
                    config.max_retries + 1,
                    str(e)[:100],
                )
                if not ensure_credentials():
                    logger.error("Failed to refresh credentials, giving up")
                    raise

                # Let caller recreate clients with fresh credentials
                if on_credential_refresh is not None:
                    logger.debug("Invoking on_credential_refresh callback")
                    on_credential_refresh()

            if attempt < config.max_retries:
                delay = calculate_delay_with_jitter(attempt, config)
                _log_retry(attempt, config, e, delay)
                time.sleep(delay)
            else:
                _log_exhausted(config, func.__name__, e)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unexpected state: no exception captured after retries")


# Default retry config for LLM calls - more aggressive for throttling
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=8,
    base_delay=2.0,
    max_delay=120.0,
    jitter_factor=0.5,
)
