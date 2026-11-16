"""Compact current-acquisition facts from saved metadata, without product reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..app.theme import current_theme


@dataclass(frozen=True)
class ResultCard:
    """Display-only facts for one stage of the saved result."""

    lines: tuple[str, ...]


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int | None:
    if isinstance(value, Real) and not isinstance(value, bool) and isfinite(value):
        if value >= 0 and int(value) == value:
            return int(value)
    return None


def _count_text(value: Any) -> str:
    count = _count(value)
    return "—" if count is None else str(count)


def result_context_label(sensing: Mapping, label: str = "", *, compact: bool = False) -> str:
    """Add the authored acquisition span without substituting a scene timestamp."""
    temporal = _mapping(sensing.get("temporal"))
    acquisition = _mapping(temporal.get("acquisition"))
    result = _mapping(temporal.get("result"))
    start, end = acquisition.get("start_time_s"), acquisition.get("end_time_s")
    reference = result.get("reference_time_s")

    def valid(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)

    has_external_context = bool(label)
    if not label and valid(reference):
        label = f"Result · {reference:.6g} s"
    if not all(valid(value) for value in (start, end, reference)) or not start <= reference <= end:
        return "" if compact and has_external_context else label
    if start == end:
        return "" if compact and has_external_context else label
    config = _mapping(sensing.get("config_resolved"))
    strategy = _mapping(config.get("processing")).get("strategy", config.get("processing_strategy"))
    kind = "CPI" if strategy == "coherent_cpi_fft" else "Acquisition"
    span = f"{kind} {start:.6g}–{end:.6g} s"
    if compact:
        return span
    return " · ".join(part for part in (label, span) if part)


def result_cards(
    sensing: Mapping,
    evaluation: Mapping | None = None,
    tracking_snapshot: Any = None,
) -> dict[str, ResultCard]:
    """Describe recorded populations, keeping unavailable evidence distinct from zero.

    Detector grids may be projected across beams. Their cell counts are never
    substituted for report or track counts here. Evaluation, when supplied, is
    already computed for this result by the shared evaluator.
    """
    metadata = _mapping(sensing.get("detection_filter_metadata"))
    tracker = _mapping(metadata.get("tracker"))
    published = _count(metadata.get("final_kept"))
    if (
        published is None
        and sensing.get("_display_detection_collection") != "detections_before_tracking"
    ):
        rows = sensing.get("detections")
        if isinstance(rows, (list, tuple)):
            published = len(rows)
    detection = [f"{published} published" if published is not None else "Not recorded"]
    if tracker.get("enabled") is True:
        raw = _count(tracker.get("raw"))
        if raw is not None:
            detection.append(f"{raw} entering tracking")
    cards = {"Detection": ResultCard(tuple(detection))}

    filtering = []
    applied = None
    for descriptor in _mapping(sensing.get("products")).values():
        if not isinstance(descriptor, Mapping):
            continue
        if descriptor.get("kind") == "range_doppler" and descriptor.get("variant") == "primary":
            applied = _mapping(descriptor.get("processing")).get("slow_time_mean_removal_applied")
            break
    if type(applied) is bool:
        filtering.append(f"Mean removal · {'on' if applied else 'off'}")
    inactive_filters = 0
    for key, label in (("range_gate", "Range gate"), ("ghost_filter", "Multipath filter")):
        evidence = _mapping(metadata.get(key))
        if evidence.get("enabled") is True:
            filtering.append(f"{label} · {_count_text(evidence.get('rejected'))} rejected")
        elif evidence.get("enabled") is False:
            inactive_filters += 1
    if inactive_filters == 2:
        filtering.append("Report filters · off")
    cards["Filtering"] = ResultCard(tuple(filtering or ["Not recorded"]))

    config = _mapping(sensing.get("config_resolved"))
    enabled = tracker.get("enabled", config.get("track_confirmation_enabled"))
    if enabled is False:
        tracking = ("Disabled",)
    elif enabled is True:
        active, confirmed = _count(tracker.get("active_tracks")), _count(
            tracker.get("confirmed_tracks")
        )
        tracking = (f"{_count_text(active)} active · {_count_text(confirmed)} confirmed",)
        if active is not None and confirmed is not None and confirmed <= active:
            tracking += (f"{active - confirmed} tentative",)
    else:
        tracking = ("Not recorded",)
    if tracking_snapshot is not None and tracking_snapshot.source_key[:1] == ("saved_world_track",):
        states = tracking_snapshot.states
        if tracking_snapshot.complete_states:
            confirmed = (
                sum(state["confirmed"] for state in states)
                if all(type(state.get("confirmed")) is bool for state in states)
                else None
            )
            tracking = (
                f"{len(states)} active · {_count_text(confirmed)} confirmed",
                "Saved world tracks",
            )
        elif not states:
            tracking = ("Unavailable", "Saved world tracks")
        else:
            tracking = (f"{len(states)} saved states", "Saved world tracks")
    cards["Tracking"] = ResultCard(tracking)

    if evaluation is not None:
        summary = _mapping(evaluation.get("default_summary", evaluation))
        reason = evaluation.get("reason") or summary.get("reason")
        if "truth" in evaluation and evaluation["truth"] is None:
            reason = reason or "Ground truth unavailable"
        values = [_count(summary.get(key)) for key in ("total_tp", "total_fp", "total_fn")]
        if not reason and all(value is not None for value in values):
            lines = (f"{values[0]} matched · {values[1]} unmatched", f"{values[2]} missed targets")
        else:
            lines = ("Unavailable",)
        scope = evaluation.get("scope_label")
        if isinstance(scope, str) and scope:
            lines += (scope,)
        cards["Evaluation"] = ResultCard(lines)
    return cards


class _Card(QFrame):
    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("currentResultCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self.title = QLabel(title)
        self.title.setObjectName("currentResultHeading")
        layout.addWidget(self.title)
        self.value = QLabel("Not recorded")
        self.value.setWordWrap(True)
        layout.addWidget(self.value)
        layout.addStretch()


class CurrentResultSummary(QWidget):
    """Three compact metadata cards, with optional current-result evaluation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CurrentResultSummary")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.context = QLabel()
        self.context.setWordWrap(True)
        layout.addWidget(self.context)
        self.grid = QGridLayout()
        layout.addLayout(self.grid)
        self.cards = {
            title: _Card(title, self)
            for title in ("Detection", "Filtering", "Tracking", "Evaluation")
        }
        self._facts: dict[str, ResultCard] = {}
        self._columns = 0
        self._card_layout = None
        self.update_result({})
        self.apply_theme()

    def apply_theme(self) -> None:
        theme = current_theme()
        self.setStyleSheet(
            f"QFrame#currentResultCard {{ background: {theme.bg_secondary}; "
            f"border: 1px solid {theme.border_subtle}; border-radius: 6px; }}"
            f"QLabel#currentResultHeading {{ color: {theme.text_secondary}; font-weight: 600; }}"
            "QFrame#currentResultCard QLabel { border: none; background: transparent; }"
        )

    def update_result(
        self,
        sensing: Mapping,
        *,
        result_label: str = "",
        evaluation: Mapping | None = None,
        tracking_snapshot: Any = None,
    ) -> None:
        context_label = result_context_label(sensing, result_label, compact=True)
        self.context.setText(context_label)
        self.context.setVisible(bool(context_label))
        self._facts = result_cards(sensing, evaluation, tracking_snapshot)
        for title, card in self.cards.items():
            fact = self._facts.get(title)
            card.setVisible(fact is not None)
            if fact is not None:
                card.value.setText("\n".join(fact.lines))
        self._layout_cards()

    def _layout_cards(self) -> None:
        columns = 1 if self.width() < 300 else (2 if self.width() < 900 else len(self._facts))
        layout = (columns, tuple(self._facts))
        if layout == self._card_layout:
            return
        for index, title in enumerate(self._facts):
            self.grid.addWidget(self.cards[title], index // columns, index % columns)
        for column in range(max(columns, self._columns)):
            self.grid.setColumnStretch(column, int(column < columns))
        self._columns = columns
        self._card_layout = layout

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._layout_cards()
