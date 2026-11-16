"""On-demand geometry sequences; timeline length never owns mesh residency."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, overload

T = TypeVar("T")


class FrameGeometry(Sequence[T], Generic[T]):
    """Index a prepared timeline while retaining only the displayed frame.

    Mesh ownership belongs to the shared byte-bounded target asset cache. The
    sequence keeps no per-sample render records, even for long recordings.
    """

    def __init__(self, count: int, prepare: Callable[[int], T]) -> None:
        self._count = count
        self._prepare = prepare

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._prepare(index)
