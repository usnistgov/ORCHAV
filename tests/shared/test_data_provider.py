"""
Tests for shared.frames.provider_base module.

These tests verify the DataProvider abstract base class and related types
that define the interface for all data providers.
"""

import numpy as np
import pytest

from shared.frames.contracts import FrameReadRequest
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.provider_base import (
    DataProvider,
    ProviderCapability,
    ProviderInfo,
)
from shared.frames.types import StandardMPCFrame


def _minimal_frame(frame_index: int) -> StandardMPCFrame:
    """Create a complete empty-path frame for provider contract tests."""

    return standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        rx_positions=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
        vertices_by_pair=[np.empty((0, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((0, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
    )


class TestProviderCapability:
    """Tests for ProviderCapability flags."""

    def test_capability_flags_are_distinct(self):
        """Each capability should have a unique flag value."""
        caps = [
            ProviderCapability.STREAMING,
            ProviderCapability.CACHING,
            ProviderCapability.RANDOM_ACCESS,
            ProviderCapability.WRITE,
            ProviderCapability.OVERRIDES,
        ]
        # Each capability should have a different value
        values = [c.value for c in caps]
        assert len(values) == len(set(values))

    def test_capability_flags_can_be_combined(self):
        """Capabilities can be combined with bitwise OR."""
        combined = ProviderCapability.STREAMING | ProviderCapability.CACHING
        assert combined & ProviderCapability.STREAMING
        assert combined & ProviderCapability.CACHING
        assert not (combined & ProviderCapability.WRITE)

    def test_capability_none_is_falsy(self):
        """NONE capability should be falsy."""
        assert not ProviderCapability.NONE


class TestProviderInfo:
    """Tests for ProviderInfo dataclass."""

    def test_create_minimal_info(self):
        """Can create ProviderInfo with minimal fields."""
        info = ProviderInfo(name="TestProvider", source="/path/to/data")
        assert info.name == "TestProvider"
        assert info.source == "/path/to/data"
        assert info.total_frames == -1  # Default
        assert info.frame_rate == 0.0  # Default

    def test_create_full_info(self):
        """Can create ProviderInfo with all fields."""
        info = ProviderInfo(
            name="TestProvider",
            source="localhost:50051",
            total_frames=100,
            frame_rate=30.0,
            capabilities=ProviderCapability.STREAMING | ProviderCapability.CACHING,
        )
        assert info.total_frames == 100
        assert info.frame_rate == 30.0
        assert info.capabilities & ProviderCapability.STREAMING


class TestDataProviderABC:
    """Tests for DataProvider abstract base class."""

    def test_cannot_instantiate_directly(self):
        """DataProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            DataProvider()


class ConcreteTestProvider(DataProvider):
    """Concrete implementation for testing."""

    def __init__(self, frames: list[int] = None):
        self._frames = frames or [0, 1, 2]
        self._opened = False
        self._closed = False

    def open(self):
        self._opened = True

    def close(self):
        self._closed = True

    def list_frames(self) -> list[int]:
        return self._frames

    def has_frame(self, step: int) -> bool:
        return step in self._frames

    def load_frame(self, step: int) -> StandardMPCFrame:
        if step not in self._frames:
            raise KeyError(f"Frame {step} not found")
        return _minimal_frame(step)


class TestConcreteProvider:
    """Tests for concrete provider implementation."""

    def test_list_frames(self):
        """list_frames should return available frames."""
        provider = ConcreteTestProvider(frames=[0, 5, 10])
        assert provider.list_frames() == [0, 5, 10]

    def test_has_frame_true(self):
        """has_frame should return True for existing frames."""
        provider = ConcreteTestProvider(frames=[0, 1, 2])
        assert provider.has_frame(0)
        assert provider.has_frame(1)

    def test_has_frame_false(self):
        """has_frame should return False for non-existing frames."""
        provider = ConcreteTestProvider(frames=[0, 1, 2])
        assert not provider.has_frame(99)

    def test_load_frame_success(self):
        """load_frame should return frame data for existing frames."""
        provider = ConcreteTestProvider(frames=[0])
        frame = provider.load_frame(0)
        assert isinstance(frame, StandardMPCFrame)
        np.testing.assert_allclose(frame.tx_positions, [[1.0, 2.0, 3.0]])

    def test_load_frame_missing(self):
        """load_frame should raise KeyError for missing frames."""
        provider = ConcreteTestProvider(frames=[0])
        with pytest.raises(KeyError):
            provider.load_frame(99)


class TestProviderLifecycle:
    """Tests for provider lifecycle methods."""

    def test_open_is_called(self):
        """open() should be callable."""
        provider = ConcreteTestProvider()
        provider.open()
        assert provider._opened

    def test_close_is_called(self):
        """close() should be callable."""
        provider = ConcreteTestProvider()
        provider.close()
        assert provider._closed

    def test_context_manager(self):
        """Provider can be used as context manager."""
        provider = ConcreteTestProvider()
        with provider:
            assert provider._opened
        assert provider._closed

    def test_info_property(self):
        """Default info property returns metadata."""
        provider = ConcreteTestProvider(frames=[0, 1, 2, 3, 4])
        info = provider.info
        assert info.name == "ConcreteTestProvider"
        assert info.total_frames == 5
        assert info.capabilities & ProviderCapability.RANDOM_ACCESS


class TestProviderOptionalMethods:
    """Tests for optional provider methods."""

    def test_subscribe_default_does_nothing(self):
        """Default subscribe() does nothing."""
        provider = ConcreteTestProvider()

        def _callback(idx, frame):
            return None

        # Should not raise
        provider.subscribe(_callback)

    def test_unsubscribe_default_does_nothing(self):
        """Default unsubscribe() does nothing."""
        provider = ConcreteTestProvider()

        def callback(idx, frame):
            return None

        provider.subscribe(callback)
        # Should not raise
        provider.unsubscribe(callback)

    def test_load_with_overrides_default_returns_none(self):
        """Default load_frame_with_overrides() returns None."""
        provider = ConcreteTestProvider(frames=[0])
        result = provider.load_frame_with_overrides(0, [])
        assert result is None

    def test_projection_reports_provider_that_returned_no_frame(self):
        class MissingFrameProvider(ConcreteTestProvider):
            def load_frame(self, step: int):
                return None

        provider = MissingFrameProvider(frames=[0])

        with pytest.raises(KeyError, match="Frame 0.*MissingFrameProvider"):
            provider.load_frame_projection(0, FrameReadRequest())


class _FailingListProvider(ConcreteTestProvider):
    def list_frames(self) -> list[int]:
        raise RuntimeError("not open")


def test_info_handles_list_frames_failure():
    provider = _FailingListProvider(frames=[0, 1])
    info = provider.info
    assert info.total_frames == 0
