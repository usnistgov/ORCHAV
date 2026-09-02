"""Track transient interactive edits independently from renderer objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class InteractivePose:
    """Position plus optional yaw/pitch/roll orientation for an editable entity."""

    position: tuple[float, float, float]
    orientation: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class InteractiveEdit:
    """Transient pose edit plus identity aliases captured when editing began."""

    kind: str
    index: int
    original: InteractivePose
    current: InteractivePose
    identity_aliases: tuple[str, ...] = ()


def _pose(position: object, orientation: object | None = None) -> InteractivePose:
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
        raise ValueError("Interactive edit position must be a finite XYZ vector")
    orient_tuple: tuple[float, float, float] | None = None
    if orientation is not None:
        orient = np.asarray(orientation, dtype=np.float64).reshape(-1)
        if orient.size < 3 or not np.all(np.isfinite(orient[:3])):
            raise ValueError("Interactive edit orientation must be a finite yaw/pitch/roll vector")
        orient_tuple = (float(orient[0]), float(orient[1]), float(orient[2]))
    return InteractivePose(
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        orientation=orient_tuple,
    )


class InteractiveEditSession:
    """Remember dirty transient edits and their reset baselines."""

    def __init__(self) -> None:
        self._edits: dict[tuple[str, int], InteractiveEdit] = {}

    def clear(self) -> None:
        """Forget all transient edit state."""
        self._edits.clear()

    def record(
        self,
        *,
        kind: str,
        index: int,
        position: object,
        orientation: object | None = None,
        baseline_position: object,
        baseline_orientation: object | None = None,
        identity_aliases: Iterable[str] = (),
    ) -> InteractiveEdit:
        """Record or update one edit while preserving its original baseline."""
        key = (str(kind).lower(), int(index))
        current = _pose(position, orientation)
        previous = self._edits.get(key)
        original = (
            previous.original
            if previous is not None
            else _pose(baseline_position, baseline_orientation)
        )
        aliases = (
            previous.identity_aliases
            if previous is not None and previous.identity_aliases
            else tuple(sorted({str(alias) for alias in identity_aliases if str(alias)}))
        )
        edit = InteractiveEdit(
            kind=key[0],
            index=key[1],
            original=original,
            current=current,
            identity_aliases=aliases,
        )
        self._edits[key] = edit
        return edit

    def discard(self, kind: str, index: int) -> InteractiveEdit | None:
        """Remove and return one edit, if present."""
        return self._edits.pop((str(kind).lower(), int(index)), None)

    def get(self, kind: str, index: int) -> InteractiveEdit | None:
        """Return one edit without mutating the session."""
        return self._edits.get((str(kind).lower(), int(index)))

    def edited_keys(self) -> tuple[tuple[str, int], ...]:
        """Return edited entity keys in stable display order."""
        return tuple(sorted(self._edits))

    def edits(self) -> Iterable[InteractiveEdit]:
        """Iterate current edits in stable display order."""
        for key in self.edited_keys():
            yield self._edits[key]

    def dirty_count(self) -> int:
        """Return the number of edited entities."""
        return len(self._edits)

    def is_dirty(self) -> bool:
        """Return whether any transient edits are active."""
        return bool(self._edits)
