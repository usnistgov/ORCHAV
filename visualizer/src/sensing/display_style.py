"""Identity colors shared by 2D/3D sensing estimates and evaluation overlays.

Colors identify saved track IDs, not inferred associations with truth targets.
Markers distinguish detections (cross), estimates (diamond), and truth (plus).
Confirmation changes emphasis, never a track's identity color.
"""

from __future__ import annotations

from zlib import crc32

TRACK_COLORS = ("#56a9df", "#ef8a47", "#70b85a", "#b883d5", "#df6384", "#42bcb1")
DETECTION_COLOR = "#ff5722"
# A dark edge separates identity colors from bright heatmap peaks in either theme.
HEATMAP_MARKER_OUTLINE = "#17212b"


def identity_color(identity: str | int, *, truth: bool = False) -> str:
    """Return a process-stable color independent of visible IDs or visit order."""
    key = str(identity)
    index = crc32(("truth:" + key).encode()) if truth else int(key)
    return TRACK_COLORS[index % len(TRACK_COLORS)]


def color_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a display hex color to renderer-neutral RGBA."""
    return (*tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)), alpha)


def detection_color(detection: dict) -> str:
    """Use a saved track identity when present, otherwise the detection color."""
    tid = detection.get("track_id")
    return identity_color(tid) if tid is not None else DETECTION_COLOR
