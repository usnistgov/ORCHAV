from pathlib import Path

import pytest

from shared.scenarios.actors import TimelineSpec
from visualizer.src.io.frame_source_extensions import register_frame_source_extension
from visualizer.src.io.frame_sources import (
    FileSource,
    LiveGrpcSource,
    RemoteHdf5Source,
    make_frame_source,
)
from visualizer.src.io.scenario_config import Scenario


def _scenario(data_mode: str, data_spec: dict | None = None, **kwargs) -> Scenario:
    return Scenario(
        root=Path("/mock/root"),
        scene_spec={},
        data_mode=data_mode,
        data_spec=data_spec or {},
        view_defaults={},
        timeline=TimelineSpec(steps=1, duration_s=0.0),
        **kwargs,
    )


def test_make_frame_source_files():
    """Test creating FileSource from scenario."""
    scenario = _scenario("files", {"files": {"directory": "packed", "format": "h5"}})

    source = make_frame_source(scenario)

    assert isinstance(source, FileSource)
    assert source.fmt == "h5"
    assert source.directory == "packed"


def test_make_frame_source_live_grpc():
    """Test creating LiveGrpcSource from scenario."""
    scenario = _scenario("live_grpc", {"live_grpc": {"endpoint": "grpc://localhost:50051"}})

    source = make_frame_source(scenario)

    assert isinstance(source, LiveGrpcSource)
    assert source.endpoint == "grpc://localhost:50051"


def test_make_frame_source_live_grpc_uses_endpoint_fallback():
    """Live gRPC mode endpoint falls back to Scenario.live_grpc_endpoints['sionna']."""
    scenario = _scenario(
        "live_grpc",
        {"live_grpc": {}},
        live_grpc_endpoints={"sionna": "grpc://fallback:50051"},
    )

    source = make_frame_source(scenario)

    assert isinstance(source, LiveGrpcSource)
    assert source.endpoint == "grpc://fallback:50051"


def test_make_frame_source_remote_hdf5():
    """Test creating RemoteHdf5Source from scenario."""
    scenario = _scenario(
        "remote_hdf5",
        {
            "remote_hdf5": {
                "server": "localhost:50052",
                "cache_size": 7,
                "connect_timeout": 2.5,
                "frame_index_ttl_s": 3.0,
            }
        },
    )

    source = make_frame_source(scenario)

    assert isinstance(source, RemoteHdf5Source)
    assert source.server_address == "localhost:50052"
    assert source.cache_size == 7
    assert source.connect_timeout == 2.5
    assert source.frame_index_ttl_s == 3.0


@pytest.mark.parametrize(
    ("remote_spec", "message"),
    [
        ({"cache_size": 0}, "cache_size"),
        ({"connect_timeout": 0}, "connect_timeout"),
        ({"frame_index_ttl_s": -1}, "frame_index_ttl_s"),
    ],
)
def test_make_frame_source_remote_hdf5_rejects_invalid_settings(remote_spec, message):
    scenario = _scenario("remote_hdf5", {"remote_hdf5": remote_spec})

    with pytest.raises(ValueError, match=message):
        make_frame_source(scenario)


def test_make_frame_source_uses_registered_extension():
    class CustomSource:
        pass

    register_frame_source_extension(
        "unit_custom_source",
        lambda scenario: CustomSource(),
        label="Unit Custom",
    )
    scenario = _scenario("unit_custom_source")

    source = make_frame_source(scenario)

    assert isinstance(source, CustomSource)
    assert source.frame_source_label == "Unit Custom"


def test_make_frame_source_unknown_mode_reports_error():
    scenario = _scenario("missing_mode")

    with pytest.raises(ValueError, match="Unknown data mode: missing_mode"):
        make_frame_source(scenario)


def test_file_source_open(monkeypatch):
    """Test FileSource opening."""
    calls = []

    class MockHdf5Provider:
        def __init__(self, root, *, frames_subdir):
            calls.append((root, frames_subdir))

    monkeypatch.setattr("visualizer.src.io.frame_sources.Hdf5Provider", MockHdf5Provider)

    source = FileSource(Path("/mock/root"), "packed/frames", "h5")
    source.open()

    assert calls == [(str(Path("/mock/root")), "packed/frames")]
    assert source.provider is not None


def test_file_source_close_releases_provider_and_allows_reopen(monkeypatch):
    events = []

    class MockHdf5Provider:
        def __init__(self, _root, *, frames_subdir):
            events.append(("open", frames_subdir))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr("visualizer.src.io.frame_sources.Hdf5Provider", MockHdf5Provider)
    source = FileSource(Path("/mock/root"), "frames", "h5")

    source.open()
    source.close()
    source.open()

    assert events == [("open", "frames"), ("close",), ("open", "frames")]
    assert source.provider is not None


def test_file_source_unsupported_format():
    """Test FileSource with unsupported format."""
    source = FileSource(Path("/mock/root"), "frames", "csv")
    with pytest.raises(ValueError):
        source.open()
