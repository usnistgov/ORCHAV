"""Display-only radio roles shared by figures and the desktop visualizer.

Grouping never changes device identities or the acquisition's pair inventory.
Only unambiguous one-to-one, colocated TX/RX pairs receive a radar label.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def monostatic_label_pairs(tx_positions: Any, rx_positions: Any) -> dict[int, int]:
    """Match unique colocated nodes (or complete sampled trajectories).

    Absolute tolerance is one micrometre, independent of world coordinates.
    Ambiguous multiple nodes at one position remain separately identified.
    Callers must first establish that sensing is enabled.
    """
    tx = [np.asarray(p, dtype=float) for p in tx_positions]
    rx = [np.asarray(p, dtype=float) for p in rx_positions]
    matches = np.zeros((len(tx), len(rx)), dtype=bool)
    for i, a in enumerate(tx):
        for j, b in enumerate(rx):
            matches[i, j] = (
                a.shape == b.shape
                and a.size > 0
                and a.shape[-1] == 3
                and np.all(np.isfinite(a))
                and np.all(np.isfinite(b))
                and np.allclose(a, b, atol=1e-6, rtol=0)
            )
    return {
        int(i): int(j)
        for i, j in zip(*np.nonzero(matches))
        if matches[i].sum() == 1 and matches[:, j].sum() == 1
    }


def sensing_role_label(
    kind: str,
    index: int,
    pairs: dict[int, int],
    *,
    device_names: tuple[str, str] | None = None,
) -> str:
    """Return a display role, optionally identifying a named colocated pair.

    ``device_names`` contains the matched transmitter and receiver names. A
    shared stem such as StationTX/StationRX identifies the radar; unrelated
    names are both retained. Conventional TX1/RX1 names keep the numbered role.
    """
    for number, (tx, rx) in enumerate(sorted(pairs.items()), 1):
        if (kind.lower() == "tx" and index == tx) or (kind.lower() == "rx" and index == rx):
            if device_names is not None:
                names = tuple(name.strip() for name in device_names)
                generic = all(
                    re.fullmatch(rf"(?:{role})[\s_-]*\d*", name, re.IGNORECASE)
                    for role, name in zip(("tx|transmitter", "rx|receiver"), names, strict=True)
                )
                if all(names) and not generic:
                    stems = tuple(
                        re.sub(rf"[\s_-]*(?:{role})$", "", name, flags=re.IGNORECASE)
                        for role, name in zip(("tx|transmitter", "rx|receiver"), names, strict=True)
                    )
                    name = stems[0] if stems[0] and stems[0] == stems[1] else " / ".join(names)
                    return f"{name} (monostatic radar)"
            return f"Monostatic radar {number}"
    role = "transmitter" if kind.lower() == "tx" else "receiver"
    return f"Sensing {role} {index + 1}"
