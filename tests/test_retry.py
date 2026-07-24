# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for retry handling with exponential backoff and jitter.

Includes property-based tests for:
- Property 5: Exponential Backoff with Cap
- Property 6: Jitter Bounds
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from patient_agent_bench.utils.retry import (
    RetryConfig,
    calculate_delay_with_jitter,
    is_expired_token_error,
    retry_with_backoff,
    retry_sync_with_backoff,
    ThrottlingException,
    ServiceUnavailableException,
    is_throttling_error,
)


# =============================================================================
# Unit Tests for RetryConfig
# =============================================================================


class TestRetryConfig:
    """Unit tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test that RetryConfig has correct default values."""
        config = RetryConfig()
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter_factor == 0.5
        assert ThrottlingException in config.retryable_exceptions
        assert ServiceUnavailableException in config.retryable_exceptions

    def test_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0,
            jitter_factor=0.25,
        )
        assert config.max_retries == 3
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.jitter_factor == 0.25


# =============================================================================
# Unit Tests for is_throttling_error
# =============================================================================


class TestIsThrottlingError:
    """Unit tests for is_throttling_error function."""

    def test_throttling_exception(self):
        """Test that ThrottlingException is detected."""
        error = ThrottlingException("Rate limit exceeded")
        assert is_throttling_error(error) is True

    def test_service_unavailable_exception(self):
        """Test that ServiceUnavailableException is detected."""
        error = ServiceUnavailableException("Service unavailable")
        assert is_throttling_error(error) is True

    def test_generic_exception_with_429(self):
        """Test that exceptions with 429 in message are detected."""
        error = Exception("HTTP 429 Too Many Requests")
        assert is_throttling_error(error) is True

    def test_generic_exception_with_throttle(self):
        """Test that exceptions with 'throttle' in message are detected."""
        error = Exception("Request was throttled")
        assert is_throttling_error(error) is True

    def test_generic_exception_with_rate_limit(self):
        """Test that exceptions with 'rate limit' in message are detected."""
        error = Exception("Rate limit exceeded")
        assert is_throttling_error(error) is True

    def test_non_throttling_exception(self):
        """Test that non-throttling exceptions are not detected."""
        error = ValueError("Invalid input")
        assert is_throttling_error(error) is False


# =============================================================================
# Unit Tests for calculate_delay_with_jitter
# =============================================================================


class TestCalculateDelayWithJitter:
    """Unit tests for calculate_delay_with_jitter function."""

    def test_attempt_zero(self):
        """Test delay calculation for first attempt."""
        config = RetryConfig(base_delay=1.0, max_delay=60.0, jitter_factor=0.0)
        delay = calculate_delay_with_jitter(0, config)
        # With jitter_factor=0, delay should be exactly base_delay * 2^0 = 1.0
        assert delay == 1.0

    def test_exponential_growth(self):
        """Test that delay grows exponentially."""
        config = RetryConfig(base_delay=1.0, max_delay=60.0, jitter_factor=0.0)
        delays = [calculate_delay_with_jitter(i, config) for i in range(5)]
        # Expected: 1, 2, 4, 8, 16
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, jitter_factor=0.0)
        # Attempt 5: 1 * 2^5 = 32, but capped at 10
        delay = calculate_delay_with_jitter(5, config)
        assert delay == 10.0

    def test_non_negative_result(self):
        """Test that result is never negative."""
        config = RetryConfig(base_delay=1.0, max_delay=60.0, jitter_factor=0.5)
        # Run multiple times to account for randomness
        for _ in range(100):
            delay = calculate_delay_with_jitter(0, config)
            assert delay >= 0.0


# =============================================================================
# Property-Based Tests for Retry Module
# =============================================================================


class TestRetryPropertyBased:
    """
    Property-based tests for retry module.

    **Feature: parallel-execution**
    **Property 5: Exponential Backoff with Cap**
    **Property 6: Jitter Bounds**
    **Validates: Requirements 4.2, 4.5, 5.1, 5.2, 5.3**
    """

    @given(
        attempt=st.integers(min_value=0, max_value=20),
        base_delay=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        max_delay=st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_5_exponential_backoff_with_cap(self, attempt, base_delay, max_delay):
        """
        Property 5: Exponential Backoff with Cap

        *For any* retry attempt number A with base_delay B and max_delay M,
        the base delay before jitter SHALL equal min(B * 2^A, M).

        **Validates: Requirements 4.2, 4.5**
        """
        # Use jitter_factor=0 to test the base exponential calculation
        config = RetryConfig(
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_factor=0.0,
        )

        delay = calculate_delay_with_jitter(attempt, config)

        # Expected: min(base_delay * 2^attempt, max_delay)
        expected_base = base_delay * (2 ** attempt)
        expected = min(expected_base, max_delay)

        assert delay == pytest.approx(expected, rel=1e-9)

    @given(
        attempt=st.integers(min_value=0, max_value=10),
        base_delay=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        max_delay=st.floats(min_value=10.0, max_value=60.0, allow_nan=False, allow_infinity=False),
        jitter_factor=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_6_jitter_bounds(self, attempt, base_delay, max_delay, jitter_factor):
        """
        Property 6: Jitter Bounds

        *For any* calculated delay D with jitter_factor J, the final delay with jitter
        SHALL be in the range [D * (1-J), D * (1+J)] and SHALL always be non-negative.

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
        """
        config = RetryConfig(
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_factor=jitter_factor,
        )

        # Calculate the base delay (without jitter)
        base_exp_delay = base_delay * (2 ** attempt)
        capped_delay = min(base_exp_delay, max_delay)

        # Calculate expected bounds
        jitter_min = 1 - jitter_factor
        jitter_max = 1 + jitter_factor
        expected_min = capped_delay * jitter_min
        expected_max = capped_delay * jitter_max

        # Run multiple times to test randomness bounds
        for _ in range(10):
            delay = calculate_delay_with_jitter(attempt, config)

            # Delay must be within jitter bounds
            assert delay >= expected_min - 1e-9, f"Delay {delay} < min {expected_min}"
            assert delay <= expected_max + 1e-9, f"Delay {delay} > max {expected_max}"

            # Delay must always be non-negative
            assert delay >= 0.0, f"Delay {delay} is negative"


# =============================================================================
# Async Tests for retry_with_backoff
# =============================================================================


class TestRetryWithBackoff:
    """Tests for retry_with_backoff async function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test that successful calls return immediately."""
        call_count = 0

        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(successful_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_throttling(self):
        """Test that throttling exceptions trigger retry."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ThrottlingException("Rate limited")
            return "success"

        config = RetryConfig(max_retries=5, base_delay=0.01, jitter_factor=0.0)
        result = await retry_with_backoff(flaky_func, config=config)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that exception is raised after max retries."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ThrottlingException("Always fails")

        config = RetryConfig(max_retries=2, base_delay=0.01, jitter_factor=0.0)

        with pytest.raises(ThrottlingException):
            await retry_with_backoff(always_fails, config=config)

        # Should have tried 3 times (initial + 2 retries)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are raised immediately."""
        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        config = RetryConfig(max_retries=5, base_delay=0.01)

        with pytest.raises(ValueError):
            await retry_with_backoff(raises_value_error, config=config)

        # Should have only tried once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        """Test that args and kwargs are passed to the function."""
        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await retry_with_backoff(func_with_args, "x", "y", c="z")
        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_service_unavailable_retried(self):
        """Test that ServiceUnavailableException triggers retry."""
        call_count = 0

        async def service_unavailable_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ServiceUnavailableException("Service down")
            return "recovered"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)
        result = await retry_with_backoff(service_unavailable_func, config=config)

        assert result == "recovered"
        assert call_count == 2


# =============================================================================
# Tests for is_expired_token_error
# =============================================================================


class TestIsExpiredTokenError:
    """Tests for is_expired_token_error function."""

    def test_expired_token_client_error(self):
        """Test detection of ExpiredTokenException from ClientError."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "Token expired"}},
            "InvokeModel",
        )
        assert is_expired_token_error(error) is True

    def test_expired_token_variant(self):
        """Test detection of ExpiredToken variant."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ExpiredToken", "Message": "Token expired"}},
            "InvokeModel",
        )
        assert is_expired_token_error(error) is True

    def test_request_expired(self):
        """Test detection of RequestExpired."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "RequestExpired", "Message": "Request expired"}},
            "InvokeModel",
        )
        assert is_expired_token_error(error) is True

    def test_unrecognized_client(self):
        """Test detection of UnrecognizedClientException."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "UnrecognizedClientException", "Message": "Unrecognized"}},
            "InvokeModel",
        )
        assert is_expired_token_error(error) is True

    def test_throttling_is_not_expired(self):
        """Test that throttling errors are not detected as expired token."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )
        assert is_expired_token_error(error) is False

    def test_generic_exception_with_expired_token_string(self):
        """Test detection from generic exception with ExpiredTokenException in message."""
        error = Exception("An error occurred (ExpiredTokenException): token has expired")
        assert is_expired_token_error(error) is True

    def test_generic_exception_without_expired_token(self):
        """Test that unrelated generic exceptions are not detected."""
        error = Exception("Something else went wrong")
        assert is_expired_token_error(error) is False

    def test_mantle_sigv4_expired_token_401(self):
        """Mantle (SigV4 via OpenAI client) surfaces expiry as a 401 whose body
        carries the AWS SigV4 phrasing; it must be treated as an expired token
        so the retry loop triggers a credential refresh."""
        error = Exception(
            "Error code: 401 - {'error': {'code': 'invalid_api_key', "
            "'message': 'The security token included in the request is expired', "
            "'type': 'permission_denied_error'}}"
        )
        assert is_expired_token_error(error) is True

    def test_openai_direct_bad_key_not_expired_token(self):
        """A plain OpenAI/LiteLLM bad-key 401 (static key) is NOT an expired
        token — a credential refresh cannot fix it, so it must fail fast rather than
        loop through credential refreshes."""
        error = Exception(
            "Error code: 401 - {'error': {'code': 'invalid_api_key', "
            "'message': 'Incorrect API key provided: sk-xxxx'}}"
        )
        assert is_expired_token_error(error) is False

    def test_expired_token_not_throttling(self):
        """Test that expired token errors are NOT classified as throttling."""
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ExpiredTokenException", "Message": "Token expired"}},
            "InvokeModel",
        )
        assert is_throttling_error(error) is False


# =============================================================================
# Tests for retry with expired token credential refresh
# =============================================================================


class TestRetryWithExpiredToken:
    """Tests for retry behavior when ExpiredTokenException occurs."""

    @pytest.mark.asyncio
    async def test_async_retry_refreshes_credentials_on_expired_token(self):
        """Test that async retry calls ensure_credentials on expired token."""
        from unittest.mock import patch, MagicMock
        from botocore.exceptions import ClientError

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClientError(
                    {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                    "InvokeModel",
                )
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        with patch("patient_agent_bench.utils.retry.ensure_credentials", return_value=True):
            result = await retry_with_backoff(flaky_func, config=config)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_fails_if_credential_refresh_fails(self):
        """Test that async retry raises if ensure_credentials returns False."""
        from unittest.mock import patch
        from botocore.exceptions import ClientError

        async def always_expired():
            raise ClientError(
                {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                "InvokeModel",
            )

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        with patch("patient_agent_bench.utils.retry.ensure_credentials", return_value=False):
            with pytest.raises(ClientError):
                await retry_with_backoff(always_expired, config=config)

    def test_sync_retry_refreshes_credentials_on_expired_token(self):
        """Test that sync retry calls ensure_credentials on expired token."""
        from unittest.mock import patch
        from botocore.exceptions import ClientError

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClientError(
                    {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                    "InvokeModel",
                )
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        with patch("patient_agent_bench.utils.retry.ensure_credentials", return_value=True):
            result = retry_sync_with_backoff(flaky_func, config=config)

        assert result == "success"
        assert call_count == 2

    def test_sync_retry_fails_if_credential_refresh_fails(self):
        """Test that sync retry raises if ensure_credentials returns False."""
        from unittest.mock import patch
        from botocore.exceptions import ClientError

        def always_expired():
            raise ClientError(
                {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                "InvokeModel",
            )

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        with patch("patient_agent_bench.utils.retry.ensure_credentials", return_value=False):
            with pytest.raises(ClientError):
                retry_sync_with_backoff(always_expired, config=config)

    @pytest.mark.asyncio
    async def test_ensure_credentials_called_once_per_expired_token(self):
        """Test that ensure_credentials is called each time an expired token occurs."""
        from unittest.mock import patch, call
        from botocore.exceptions import ClientError

        call_count = 0

        async def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ClientError(
                    {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                    "InvokeModel",
                )
            return "success"

        config = RetryConfig(max_retries=5, base_delay=0.01, jitter_factor=0.0)

        with patch(
            "patient_agent_bench.utils.retry.ensure_credentials", return_value=True
        ) as mock_ensure:
            result = await retry_with_backoff(fails_twice_then_succeeds, config=config)

        assert result == "success"
        assert mock_ensure.call_count == 2


    def test_no_credentials_error_detected_as_expired(self):
        """Test that NoCredentialsError is detected by is_expired_token_error."""
        from botocore.exceptions import NoCredentialsError

        error = NoCredentialsError()
        assert is_expired_token_error(error) is True

    def test_unable_to_locate_credentials_string(self):
        """Test detection from generic exception with 'Unable to locate credentials'."""
        error = Exception("Unable to locate credentials")
        assert is_expired_token_error(error) is True

    def test_sync_retry_refreshes_on_no_credentials(self):
        """Test that sync retry calls ensure_credentials on NoCredentialsError."""
        from unittest.mock import patch
        from botocore.exceptions import NoCredentialsError

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NoCredentialsError()
            return "success"

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter_factor=0.0)

        with patch("patient_agent_bench.utils.retry.ensure_credentials", return_value=True):
            result = retry_sync_with_backoff(flaky_func, config=config)

        assert result == "success"
        assert call_count == 2
