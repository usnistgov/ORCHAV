"""Sensing retention controls distinguish coverage from per-result residency."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from visualizer.src.app.startup_workflow import build_argument_parser, parse_cli_args
from visualizer.src.sensing.cache_policy import MIB, SensingCachePolicy, policy_from_options
from visualizer.src.sensing.preparation_status import prepared_coverage_messages


def test_preparation_changes_residency_defaults_without_changing_history_coverage():
    ordinary = policy_from_options(SimpleNamespace())
    prepared = policy_from_options(SimpleNamespace(), prepare=True)
    assert ordinary.overview_bytes == prepared.overview_bytes == 64 * MIB
    assert ordinary.history_key == prepared.history_key == (4096, 64 * MIB)
    assert (ordinary.request_bytes, ordinary.request_entries) == (512 * MIB, 512)
    assert (prepared.request_bytes, prepared.request_entries) == (2048 * MIB, 2048)


def test_explicit_limits_survive_preparation_defaults_and_are_frozen():
    args = build_argument_parser().parse_args(
        [
            "--sensing-overview-cache-mb",
            "128",
            "--sensing-history-cache-mb",
            "96",
            "--sensing-history-results",
            "12000",
            "--sensing-result-cache-mb",
            "768",
            "--sensing-result-cache-entries",
            "1024",
        ]
    )
    policy = policy_from_options(args, prepare=True)
    assert policy == SensingCachePolicy(128 * MIB, 96 * MIB, 12000, 768 * MIB, 1024)
    with pytest.raises(FrozenInstanceError):
        policy.request_bytes = MIB


@pytest.mark.parametrize(
    "option",
    (
        "overview-cache-mb",
        "history-cache-mb",
        "history-results",
        "result-cache-mb",
        "result-cache-entries",
    ),
)
@pytest.mark.parametrize("value", ("0", "-1", "1.5"))
def test_invalid_limits_fail_before_starting_the_gui(option, value):
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([f"--sensing-{option}", value])


@pytest.mark.parametrize("value", (True, "128", 0, -1))
def test_unparsed_options_cannot_bypass_limit_validation(value):
    with pytest.raises(ValueError, match="sensing_overview_cache_mb"):
        policy_from_options(SimpleNamespace(sensing_overview_cache_mb=value))


def test_preparation_reports_limited_coverage_with_the_effective_control():
    messages = prepared_coverage_messages(
        {
            "motion": [
                {
                    "family": "saved-stft",
                    "variant": "primary",
                    "state": "limited",
                    "available_results": 125,
                    "loaded_results": 44,
                    "loaded_time_span": [1.327875, 2.015875],
                    "reason": "Retention budget reached",
                }
            ]
        }
    )
    assert len(messages) == 1
    assert "44/125 saved results" in messages[0]
    assert "1.32787–2.01587 s" in messages[0]
    assert "--sensing-overview-cache-mb" in messages[0]
    assert "result-cache" not in messages[0]


def test_missing_product_does_not_masquerade_as_a_cache_limit():
    assert (
        prepared_coverage_messages({"motion": [{"state": "unavailable", "available_results": 0}]})
        == ()
    )


@pytest.mark.parametrize("mode", ("all-ranges", "saved-stft"))
def test_motion_subview_is_explicit_and_requires_motion_opening(mode):
    args = parse_cli_args(["--sensing-figures", "motion", "--sensing-motion", mode])
    assert args.sensing_motion == mode
    with pytest.raises(SystemExit):
        parse_cli_args(["--sensing-figures", "summary", "--sensing-motion", mode])
