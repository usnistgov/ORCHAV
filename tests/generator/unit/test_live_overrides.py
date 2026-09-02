from __future__ import annotations

from generator.core.propagation import (
    LiveActorCategory,
    LiveActorOverride,
    normalize_live_overrides,
)


def test_normalized_live_override_uses_public_actor_types() -> None:
    category: LiveActorCategory = "tx"

    normalized = normalize_live_overrides(
        [
            {
                "category": category,
                "name": "BaseStation",
                "position": [1.0, 2.0, 3.0],
            }
        ]
    )

    assert normalized[category]["basestation"] == LiveActorOverride(
        name="BaseStation",
        category=category,
        position=(1.0, 2.0, 3.0),
    )
