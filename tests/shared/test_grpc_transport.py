"""Tests for shared gRPC endpoint parsing policy."""

from __future__ import annotations

import pytest

from shared.grpc_transport import parse_grpc_endpoint


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("localhost:50051", ("localhost", 50051)),
        ("grpc://server.example:60000", ("server.example", 60000)),
        ("grpc://[::1]:50051", ("::1", 50051)),
    ],
)
def test_parse_grpc_endpoint_accepts_host_and_port(endpoint, expected) -> None:
    assert parse_grpc_endpoint(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://localhost:50051",
        "grpc://localhost",
        "grpc://user@localhost:50051",
        "grpc://localhost:50051/path",
        "grpc://localhost:50051?option=true",
        "grpc://localhost:0",
    ],
)
def test_parse_grpc_endpoint_rejects_unusable_client_endpoints(endpoint) -> None:
    with pytest.raises(ValueError):
        parse_grpc_endpoint(endpoint)


def test_parse_grpc_endpoint_allows_zero_only_for_low_level_bind_callers() -> None:
    assert parse_grpc_endpoint("localhost:0", allow_port_zero=True) == ("localhost", 0)
