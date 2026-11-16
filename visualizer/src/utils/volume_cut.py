"""Cell-aligned display cuts shared by volume fills and grid outlines."""

from dataclasses import dataclass, replace


def cut_volume(payload, state):
    """Hide cells using the template's transparent lowest palette entry.

    Encoded sensing/category templates use zero, while physical coverage scales
    can be negative. Zero would become a visible high-value cell on a dB scale.
    """
    if state is None:
        return payload
    if isinstance(state, BoxCut):
        slices = state.slices(payload.values.shape)
    else:
        axis, index, keep_lower = state
        slices = [slice(None)] * 3
        slices[{"x": 2, "y": 1, "z": 0}[axis]] = (
            slice(index + 1, None) if keep_lower else slice(None, index)
        )
    values = payload.values.copy()
    values[tuple(slices)] = payload.limits[0]
    return replace(payload, values=values)


@dataclass(frozen=True)
class BoxCut:
    """Remove the half-open cell-index bounds, ordered X, Y, Z."""

    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def slices(self, shape):
        """Return Z/Y/X slices, rejecting bounds outside the sampled grid."""
        if len(self.bounds) != 3 or len(shape) != 3:
            raise ValueError("A box cut requires three dimensions")
        for (start, stop), count in zip(self.bounds, shape[::-1], strict=True):
            if not 0 <= start < stop <= count:
                raise ValueError("Box cut outside the sampled grid")
        return tuple(slice(start, stop) for start, stop in self.bounds[::-1])
