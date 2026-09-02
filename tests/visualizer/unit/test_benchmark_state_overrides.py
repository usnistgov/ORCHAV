"""Tests for benchmark-only visualizer state override loading."""

from __future__ import annotations

import json

from visualizer.src.app.startup_workflow import load_benchmark_state_overrides


def test_load_benchmark_state_overrides_normalizes_app_state_fields(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "mpc_visibility": {
                    "enabled": True,
                    "paths": False,
                    "bounce_points": True,
                },
                "topk_render_enabled": True,
                "topk_render_max_paths": 50000,
                "color_mode": "delay",
                "mpc_allowed_orders": [0, 1, 2],
                "tx_labels": ["A", "B"],
                "unknown_key": "ignored",
            }
        ),
        encoding="utf-8",
    )

    overrides = load_benchmark_state_overrides(str(path))

    visibility = overrides["mpc_visibility"]
    assert visibility.enabled is True
    assert visibility.paths is False
    assert visibility.bounce_points is True
    assert overrides["topk_render_enabled"] is True
    assert overrides["topk_render_max_paths"] == 50000
    assert overrides["color_mode"] == "delay"
    assert overrides["mpc_allowed_orders"] == frozenset({0, 1, 2})
    assert overrides["tx_labels"] == ("A", "B")
    assert "unknown_key" not in overrides


def test_load_benchmark_state_overrides_requires_object(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[]", encoding="utf-8")

    try:
        load_benchmark_state_overrides(str(path))
    except ValueError as exc:
        assert "must contain an object" in str(exc)
    else:  # pragma: no cover - explicit failure path
        raise AssertionError("expected ValueError")
