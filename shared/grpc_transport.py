"""Shared policy for ORCHAV's optional gRPC transports.

Live generation and remote frame playback use the same bounded message size
and endpoint formatting.  This module contains no gRPC import so importing the
base ``shared`` package does not require the optional transport dependency.

These transports use insecure gRPC. Servers bind to loopback by default;
selecting a non-loopback bind address is an explicit trusted-network choice,
not an authentication or encryption feature.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

GRPC_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
GRPC_MESSAGE_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
)

DEFAULT_GRPC_BIND_HOST = "127.0.0.1"
DEFAULT_GRPC_CONNECT_TIMEOUT_S = 10.0
DEFAULT_GRPC_UNARY_TIMEOUT_S = 30.0
DEFAULT_GRPC_SHUTDOWN_GRACE_S = 5.0


def parse_grpc_endpoint(
    endpoint: str,
    *,
    allow_port_zero: bool = False,
) -> tuple[str, int]:
    """Parse a host-and-port gRPC endpoint without importing the gRPC runtime."""
    raw_endpoint = str(endpoint).strip()
    if not raw_endpoint:
        raise ValueError("gRPC endpoint must not be empty")

    candidate = raw_endpoint if "://" in raw_endpoint else f"grpc://{raw_endpoint}"
    parsed = urlsplit(candidate)
    if parsed.scheme.casefold() != "grpc":
        raise ValueError(f"gRPC endpoint must use the grpc:// scheme, got {raw_endpoint!r}")
    if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"gRPC endpoint must contain only a host and port, got {raw_endpoint!r}")

    host = parsed.hostname
    if not host or any(character.isspace() for character in host):
        raise ValueError(f"gRPC endpoint must contain only a host and port, got {raw_endpoint!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"gRPC endpoint has an invalid port: {raw_endpoint!r}") from exc
    if port is None:
        raise ValueError(f"gRPC endpoint must include a port: {raw_endpoint!r}")
    if port == 0 and not allow_port_zero:
        raise ValueError("gRPC endpoint port must be between 1 and 65535")
    return host, port


def is_loopback_grpc_host(host: str) -> bool:
    """Return whether *host* identifies a loopback-only listener."""
    normalized_host = str(host).strip().strip("[]")
    if normalized_host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized_host).is_loopback
    except ValueError:
        return False


def format_grpc_endpoint(host: str, port: int) -> str:
    """Return a validated gRPC bind endpoint for an IPv4 or IPv6 host."""
    normalized_host = str(host).strip()
    if not normalized_host:
        raise ValueError("gRPC bind host must not be empty")

    normalized_port = int(port)
    if not 0 <= normalized_port <= 65535:
        raise ValueError(f"gRPC port must be between 0 and 65535, got {normalized_port}")

    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    return f"{normalized_host}:{normalized_port}"


def bind_grpc_server(server: object, host: str, port: int) -> tuple[str, int]:
    """Bind *server* or raise with the exact endpoint that could not bind."""
    endpoint = format_grpc_endpoint(host, port)
    add_insecure_port = getattr(server, "add_insecure_port", None)
    if not callable(add_insecure_port):
        raise TypeError("gRPC server does not provide add_insecure_port()")

    try:
        bound_port = int(add_insecure_port(endpoint))
    except RuntimeError as exc:
        raise RuntimeError(f"Could not bind gRPC server to {endpoint}: {exc}") from exc
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {endpoint}")
    return endpoint, bound_port
