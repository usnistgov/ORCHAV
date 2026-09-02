"""Structured failures raised while preparing actor poses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparationIssue:
    """One deterministic, machine-readable pose-preparation problem."""

    code: str
    path: str
    message: str
    actor_name: str | None = None


class PosePreparationError(ValueError):
    """Raised when a validated scenario cannot be converted into actor poses."""

    def __init__(
        self,
        code: str,
        path: str,
        message: str,
        *,
        actor_name: str | None = None,
    ) -> None:
        self.issue = PreparationIssue(
            code=code,
            path=path,
            message=message,
            actor_name=actor_name,
        )
        subject = f"actor {actor_name!r}: " if actor_name is not None else ""
        super().__init__(f"{path}: {subject}{message} [{code}]")
