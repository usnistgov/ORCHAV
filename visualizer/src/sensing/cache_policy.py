"""Explicit retention budgets for independent sensing display caches.

These limits bound retained derived data, not application RSS. Source indexes,
current results, temporary reads, plotting buffers and renderer allocations have
separate owners. Changing a retention budget never changes recorded science.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

MIB = 1024 * 1024


def _positive_limit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def add_sensing_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the same explicit limits in GUI and offline preparation commands."""
    for option, description in (
        (
            "overview-cache-mb",
            "MiB retained for compact recording-wide sensing spectra (default: 64).",
        ),
        (
            "history-cache-mb",
            "MiB retained for detailed detection/tracking history (default: 64).",
        ),
        (
            "history-results",
            "Maximum retained detailed history results (default: 4096). "
            "Independent of compact spectral coverage.",
        ),
        (
            "result-cache-mb",
            "MiB retained for individual sensing figure requests "
            "(default: 512; 2048 with prepare).",
        ),
        (
            "result-cache-entries",
            "Maximum resident sensing figure requests (default: 512; 2048 with prepare).",
        ),
    ):
        parser.add_argument(
            f"--sensing-{option}",
            type=_positive_limit,
            metavar="N",
            default=None,
            help=description + " Applies at recording opening; not a total application RAM cap.",
        )


@dataclass(frozen=True)
class SensingCachePolicy:
    """Source-scoped limits shared by ordinary and prepared figure readers."""

    overview_bytes: int = 64 * MIB
    history_bytes: int = 64 * MIB
    history_results: int = 4096
    request_bytes: int = 512 * MIB
    request_entries: int = 512

    def __post_init__(self) -> None:
        for name in (
            "overview_bytes",
            "history_bytes",
            "history_results",
            "request_bytes",
            "request_entries",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def for_preparation(cls) -> SensingCachePolicy:
        """Keep the existing larger resident-result allowance for preparation."""
        return cls(request_bytes=2048 * MIB, request_entries=2048)

    @property
    def history_key(self) -> tuple[int, int]:
        """Detailed-history admission depends on both limits."""
        return self.history_results, self.history_bytes


def policy_from_options(options, *, prepare: bool = False) -> SensingCachePolicy:
    """Resolve optional MiB/count overrides once, before starting workers."""
    default = SensingCachePolicy.for_preparation() if prepare else SensingCachePolicy()
    values = {}
    for option, field, factor in (
        ("sensing_overview_cache_mb", "overview_bytes", MIB),
        ("sensing_history_cache_mb", "history_bytes", MIB),
        ("sensing_history_results", "history_results", 1),
        ("sensing_result_cache_mb", "request_bytes", MIB),
        ("sensing_result_cache_entries", "request_entries", 1),
    ):
        value = getattr(options, option, None)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{option} must be a positive integer")
        values[field] = getattr(default, field) if value is None else value * factor
    return SensingCachePolicy(**values)
