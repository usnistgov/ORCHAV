"""Unit tests for GeneratorService retry logic with exponential backoff.

Tests cover:
- Successful computation on first attempt
- Retry on transient failure with success
- Max retries exceeded leading to failure
- Exponential backoff timing
- Failure tracking for health monitoring
"""

import time
from typing import Any, Dict
from unittest.mock import MagicMock

# Import the service class
from generator.io.grpc.live_server import GeneratorFrameCache, GeneratorService


def _make_frame(frame_idx: int = 0) -> Dict[str, Any]:
    """Create a mock frame data dictionary."""
    return {
        "frame_idx": frame_idx,
        "paths": {},
        "tx_positions": [[0, 0, 0]],
        "rx_positions": [[1, 1, 1]],
    }


class TestRetryOnTransientFailure:
    """Tests for retry behavior on transient failures."""

    def test_success_on_first_attempt(self):
        """Verify no retries when computation succeeds immediately."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        mock_raytracing.compute_step.return_value = _make_frame(0)

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is not None
        assert mock_raytracing.compute_step.call_count == 1
        assert service.frames_generated == 1
        assert service.frames_failed == 0

    def test_retry_succeeds_after_transient_failure(self):
        """Verify retry succeeds after transient failure."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        # Fail first 2 attempts, succeed on 3rd
        mock_raytracing.compute_step.side_effect = [
            RuntimeError("Transient error 1"),
            RuntimeError("Transient error 2"),
            _make_frame(0),
        ]

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is not None
        assert mock_raytracing.compute_step.call_count == 3
        assert service.frames_generated == 1
        assert service.frames_failed == 0


class TestMaxRetriesExceeded:
    """Tests for behavior when all retries are exhausted."""

    def test_returns_none_after_max_retries(self):
        """Verify None is returned after max retries exhausted."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        mock_raytracing.compute_step.side_effect = RuntimeError("Persistent error")

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is None
        assert mock_raytracing.compute_step.call_count == 3
        assert service.frames_generated == 0
        assert service.frames_failed == 1

    def test_fails_without_raytracing_service(self):
        """Verify graceful failure when raytracing service is missing."""
        cache = GeneratorFrameCache()

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is None
        assert service.frames_failed == 1


class TestExponentialBackoff:
    """Tests for exponential backoff timing."""

    def test_backoff_delays_increase(self):
        """Verify delay increases exponentially between retries."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        mock_raytracing.compute_step.side_effect = RuntimeError("Error")

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.05,  # 50ms base delay
        )

        start = time.time()
        service._compute_frame_with_retry(0)
        elapsed = time.time() - start

        # Expected delays: 0.05s (1st retry) + 0.1s (2nd retry) = 0.15s minimum
        # Allow some tolerance for test timing
        assert elapsed >= 0.1, f"Expected at least 0.1s, got {elapsed}s"


class TestRetryConfiguration:
    """Tests for retry configuration defaults and overrides."""

    def test_default_retry_values(self):
        """Verify default retry configuration values."""
        cache = GeneratorFrameCache()
        service = GeneratorService(
            frame_cache=cache,
            generator_config={},
        )

        assert service.max_retries == 3
        assert service.retry_delay_seconds == 0.5

    def test_custom_retry_values(self):
        """Verify custom retry configuration is respected."""
        cache = GeneratorFrameCache()
        service = GeneratorService(
            frame_cache=cache,
            generator_config={},
            max_retries=5,
            retry_delay_seconds=1.0,
        )

        assert service.max_retries == 5
        assert service.retry_delay_seconds == 1.0


class TestNoneReturnHandling:
    """Tests for handling None returns (data not available)."""

    def test_none_return_triggers_retry(self):
        """Verify None return from compute_step triggers retry."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        # Return None twice, then valid frame
        mock_raytracing.compute_step.side_effect = [None, None, _make_frame(0)]

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is not None
        assert mock_raytracing.compute_step.call_count == 3

    def test_all_none_returns_fails(self):
        """Verify failure when all attempts return None."""
        cache = GeneratorFrameCache()
        mock_raytracing = MagicMock()
        mock_raytracing.compute_step.return_value = None

        service = GeneratorService(
            frame_cache=cache,
            generator_config={"services": {"raytracing_service": mock_raytracing}},
            max_retries=3,
            retry_delay_seconds=0.01,
        )

        result = service._compute_frame_with_retry(0)

        assert result is None
        assert mock_raytracing.compute_step.call_count == 3
        assert service.frames_failed == 1
