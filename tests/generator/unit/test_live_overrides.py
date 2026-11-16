from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from generator.core.propagation import (
    LiveActorCategory,
    LiveActorOverride,
    apply_target_scale_overrides,
    normalize_live_overrides,
    resolve_live_actor_name,
)
from generator.core.propagation.live_overrides import apply_live_overrides


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

    assert normalized[category]["BaseStation"] == LiveActorOverride(
        name="BaseStation",
        category=category,
        position=(1.0, 2.0, 3.0),
    )


@pytest.mark.parametrize("category", ["tx", "rx", "target"])
@pytest.mark.parametrize("alias", ["{category}1", "{category}_1", "{category}1_upper", "main"])
def test_unambiguous_compatibility_names_resolve(category, alias) -> None:
    request = alias.format(category=category)
    if request.endswith("_upper"):
        request = request.removesuffix("_upper").upper()
    assert resolve_live_actor_name(category, request, ["Main", "Spare"]) == "Main"


@pytest.mark.parametrize("category", ["tx", "rx", "target"])
@pytest.mark.parametrize("collision", ["later_exact", "case_distinct", "earlier_exact"])
def test_exact_actor_name_moves_only_one_actor(category, collision) -> None:
    names, requested, selected = {
        "later_exact": (["alpha", f"{category}1"], f"{category}1", 1),
        "case_distinct": (["Station", "station"], "station", 1),
        "earlier_exact": ([f"{category}2", "alpha"], f"{category}2", 0),
    }[collision]
    configs = [SimpleNamespace(name=name) for name in names]
    positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
    orientations = [(0.0, 0.0, 0.0)] * 2
    kwargs = {
        f"{kind}_{field}": (
            configs if field == "configs" else positions if field == "positions" else orientations
        )
        for kind in (category,)
        for field in ("configs", "positions", "orientations")
    }
    for kind in {"tx", "rx", "target"} - {category}:
        kwargs.update({f"{kind}_{field}": [] for field in ("configs", "positions", "orientations")})
    applied = apply_live_overrides(
        [
            {
                "category": category,
                "name": requested,
                "position": (8, 9, 10),
                "orientation": (20, 30, 40),
            }
        ],
        **kwargs,
    )
    assert positions[selected] == (8.0, 9.0, 10.0)
    assert positions[1 - selected] == ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))[1 - selected]
    assert orientations[selected] == (20.0, 30.0, 40.0)
    assert orientations[1 - selected] == (0.0, 0.0, 0.0)
    assert list(applied[category]) == [requested]


@pytest.mark.parametrize(
    ("names", "requested", "error"),
    [
        (["Station", "station"], "STATION", "Ambiguous"),
        (["alpha", "TX1"], "tx1", "Ambiguous"),
        (["Main"], "typo", "Unknown"),
    ],
)
def test_unresolvable_edit_rejects_whole_batch_before_mutation(names, requested, error) -> None:
    tx_positions = [(0, 0, 0)]
    rx_positions = [(1, 1, 1)] * len(names)
    with pytest.raises(ValueError, match=error):
        apply_live_overrides(
            [
                {"category": "tx", "name": "main", "position": (9, 9, 9)},
                {"category": "rx", "name": requested.replace("tx", "rx"), "position": (8, 8, 8)},
            ],
            tx_configs=[SimpleNamespace(name="main")],
            rx_configs=[SimpleNamespace(name=name.replace("TX", "RX")) for name in names],
            target_configs=[],
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            target_positions=[],
            tx_orientations=[],
            rx_orientations=[],
            target_orientations=[],
        )
    assert tx_positions == [(0, 0, 0)]
    assert rx_positions == [(1, 1, 1)] * len(names)


def test_case_distinct_target_scale_does_not_cross_actor_boundary() -> None:
    managers = [
        SimpleNamespace(
            config=SimpleNamespace(name=name, scale=1.0), apply_scale_snapshot=MagicMock()
        )
        for name in ("Walker", "walker")
    ]
    normalized = normalize_live_overrides([{"category": "target", "name": "walker", "scale": 2.0}])
    apply_target_scale_overrides(managers, normalized["target"])
    managers[0].apply_scale_snapshot.assert_called_once_with(1.0)
    managers[1].apply_scale_snapshot.assert_called_once_with(2.0)
