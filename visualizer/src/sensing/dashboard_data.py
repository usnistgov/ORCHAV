"""Background, selective readers for the sensing dashboard.

The display clock never supplies signal samples. A recording overview collapses
each saved RD product in power at its recorded result time. Independent CPIs
remain independent; no FFT bridges them, skipped display frames, or data gaps.
Workers own their service caches, while provider projections retain the shared
storage layer's locking and lifetime rules. Workers never access Qt widgets.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from threading import Event, RLock
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from shared.cache_sizing import estimate_retained_bytes
from shared.sensing.evaluation_truth import SIGNAL_TRUTH_VERSION

from ..services.sensing_service import SensingService
from .cache_policy import SensingCachePolicy
from .dashboard_evaluation import (
    DEFAULT_RANGE_GATE_M,
    DEFAULT_VELOCITY_GATE_M_S,
    EVALUATION_CACHE_PROTOCOL,
    evaluation_event,
    summarize_events,
)
from .figure_requirements import figure_requirements, product_requirements
from .motion_overview import MotionOverview, load_motion_overview
from .recorded_products import saved_micro_doppler
from .result_navigation import ResultLocation, result_locations
from .tracking_presentation import tracking_snapshot
from .update_request import SensingUpdateRequest, _DisplayDataToken

MAX_HISTORY_RESULTS = 4096
MAX_HISTORY_BYTES = 64 * 1024 * 1024
MAX_DASHBOARD_REQUESTS = 512
# A 192-result front-Opéra RD recording retains about 400 MiB of validated
# requests. Bound common maps and their prepared inspection models together.
MAX_DASHBOARD_REQUEST_BYTES = 512 * 1024 * 1024
_SHARED_PLOT_VIEWS = frozenset(
    {"analysis", "angular", "diagnostics", "analysis-inspection", "processing-clutter"}
)


@dataclass(frozen=True)
class CompactMotionCursor:
    """Playback position and a metadata-selected result, without result details."""

    scene_id: int | str
    scene_time_s: float
    result: ResultLocation | None


def history_cells(
    times: np.ndarray, widths: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Place recorded result columns at their times, leaving missing hops empty.

    A column occupies one processing hop, not its overlapping acquisition
    window. This is a display footprint; it does not resample the saved signal.
    """
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(widths))):
        raise ValueError("Nonfinite recording times or hop durations")
    if np.any(widths <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Recording times must increase and hop durations must be positive")
    edges = [times[0] - widths[0] / 2]
    columns = []
    for time_s, width, column in zip(times, widths, values, strict=True):
        left, right = time_s - width / 2, time_s + width / 2
        if left > edges[-1] + 1e-10:
            columns.append(np.full_like(column, np.nan, dtype=float))
            edges.append(left)
        elif left < edges[-1] - 1e-10:
            raise ValueError("Overlapping result display intervals in this recording")
        columns.append(column)
        edges.append(right)
    return np.array(edges), np.stack(columns)


@dataclass(frozen=True)
class HistoryRecord:
    """One recorded result, reduced to plot-sized arrays and estimate metadata."""

    time_s: float
    identity: tuple[str, str, str, str]
    spectrum: np.ndarray | None
    doppler_hz: np.ndarray | None
    range_m: np.ndarray | None
    range_power: np.ndarray | None
    detections: tuple[dict[str, Any], ...]
    truth: dict[str, Any] | None
    truth_positions: tuple[tuple[str, tuple[float, ...]], ...]
    cadence_s: float | None = None
    storage_id: int | str | None = None
    scene_frame_id: int | str | None = None
    evaluation: dict | None = None
    tracking_states: tuple[dict, ...] = ()
    tracking_states_complete: bool = False
    sensing_metadata: dict | None = None
    micro_power: np.ndarray | None = None
    micro_times_s: np.ndarray | None = None
    micro_doppler_hz: np.ndarray | None = None
    micro_hop_s: float | None = None
    micro_recipe: dict | None = None


@dataclass(frozen=True)
class RecordingHistory:
    """A bounded recording overview, independent of playback visit order."""

    records: tuple[HistoryRecord, ...]
    truncated: bool = False
    error: str | None = None

    prepared_evaluation: dict | None = None
    available_results: int = 0
    available_time_span: tuple[float, float] | None = None
    byte_limit: int = MAX_HISTORY_BYTES
    result_limit: int = MAX_HISTORY_RESULTS

    def coverage_text(self) -> str:
        """Explain detailed-history coverage without implying whole-run totals."""
        text = (
            f"{len(self.records):,}/{self.available_results or len(self.records):,} detail results"
        )
        if self.records:
            text += f" · {self.records[0].time_s:.4g}–{self.records[-1].time_s:.4g} s"
        if self.truncated:
            text += " · detail limit reached"
        return text

    @cached_property
    def evaluation_summary(self):
        """Default full-recording metrics, shared across playback and reopen."""
        if self.prepared_evaluation is not None:
            return self.prepared_evaluation
        return summarize_events(r.evaluation for r in self.records)

    @cached_property
    def tracking_plot_data(self):
        """Reuse the same immutable plot arrays across seeks and dashboard windows."""
        from .tracking_plot_data import prepare_tracking_plot_data

        return prepare_tracking_plot_data(self)


def result_context(sensing: dict[str, Any]) -> tuple[str, str, str]:
    """Keep clocks, processing runs, and sensor pairs out of one another's plots."""
    temporal = sensing["temporal"]
    return (
        temporal["clock"]["clock_id"],
        temporal["result"]["processing_run_id"],
        temporal["acquisition"]["pair_id"],
    )


def _completed_summary_rows(rows: tuple) -> tuple:
    """Exclude retained unprocessed acquisitions from whole-run cohort selection."""
    return tuple(
        (storage, frame)
        for storage, frame in rows
        if SensingService._eligible_dashboard_result(
            frame,
            frame["sensing"]["temporal"]["display_binding"]["scene_frame_id"],
            0.0,
            hold=False,
        )
        and all(isinstance(value, str) and value for value in result_context(frame["sensing"]))
    )


def _prepared_result_identity(sensing: dict[str, Any]) -> tuple:
    temporal = sensing["temporal"]
    return (
        *result_context(sensing),
        temporal["acquisition"]["acquisition_id"],
        temporal["result"]["reference_time_s"],
        temporal["display_binding"]["scene_frame_id"],
    )


def _dashboard_product_names(view: str, extra_products: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Use the same product eligibility for fresh reads and held-result reuse."""
    # Tracking consumes saved estimates, including acquisitions with no detections.
    # Request a required JSON field so selection materializes the compact payload,
    # rather than returning its trimmed event index or reading dense spectra.
    return figure_requirements(view, visible_products=extra_products).product_names


def _selected_names(sensing, view, variant, micro="auto", extra=()):
    """Resolve exact authored keys while leaving angle cubes transient."""
    names = figure_requirements(
        view,
        variant=variant,
        micro_mode=micro,
        descriptors=sensing.get("products", {}),
        visible_products=extra,
    ).product_names
    return tuple(
        name
        for name in names
        if (sensing.get("products", {}).get(name) or {}).get("kind") != "range_doppler_angle"
    )


def _prepared_plot_supports(value, sensing, names, variant, *, require_complete=True):
    """Admit a complete owner for the authored products this view actually needs."""
    if (
        not isinstance(value, dict)
        or value.get("identity") != _prepared_result_identity(sensing)
        or not isinstance(value.get("arrays"), dict)
        or not {"rp", "rd"}.issubset(value)
        or (
            require_complete
            and any(name not in value["arrays"] for name in names if name != "config_resolved")
        )
    ):
        return False
    descriptors = sensing.get("products", {})
    for kind, field in (("range_profile", "rp"), ("range_doppler", "rd")):
        if not any(descriptors.get(name, {}).get("kind") == kind for name in names):
            continue
        selected = product_requirements((kind,), descriptors, variant)
        model = value[field]
        if model is None and not require_complete:
            continue
        metadata = getattr(model, "metadata", {})
        if not isinstance(metadata, dict):
            return False
        descriptor = descriptors.get(metadata.get("data_key"), {})
        if (
            len(selected) != 1
            or metadata.get("variant") != selected[0].variant
            or descriptor.get("kind") != kind
            or descriptor.get("variant") != selected[0].variant
            or metadata.get("product_id") != descriptor.get("product_id")
        ):
            return False
    return True


def _extend_sensing_products(service, storage, sensing, names):
    """Extend a validated owner only with missing selected product keys."""
    missing = tuple(name for name in names if name != "config_resolved" and name not in sensing)
    if not missing:
        return sensing
    frame = service.dashboard_load_row(storage, ("config_resolved", *missing))
    if frame is None or _prepared_result_identity(frame["sensing"]) != _prepared_result_identity(
        sensing
    ):
        raise ValueError("Selected sensing result changed while loading its products")
    return {
        **sensing,
        **{key: value for key, value in frame["sensing"].items() if isinstance(value, np.ndarray)},
    }


def _extend_request_products(service, request, key):
    """Share resident supersets while extending a partial owner on demand."""
    _, view, variant, _, micro, *extra = key
    if (
        view in {"processing-inputs", "processing-frequency"}
        and request.processing_inspection is not None
        and not extra
    ):
        return request
    frame = request.display_frame
    names = _selected_names(frame["sensing"], view, variant, micro, extra[0] if extra else ())
    if all(name in frame["sensing"] for name in names):
        return request
    storage = frame["_sensing_display_provenance"]["storage_frame_id"]
    sensing = _extend_sensing_products(service, storage, frame["sensing"], names)
    if sensing is frame["sensing"]:
        return request
    frame = {**frame, "sensing": sensing}
    rp, rd = service._dashboard_products_from_validated(sensing, variant)
    rp = request.rp_result if request.rp_result is not None else rp
    rd = request.rd_result if request.rd_result is not None else rd
    token = _DisplayDataToken(sensing)
    token.require_complete = False
    result = replace(
        request,
        display_frame=frame,
        rp_result=rp,
        rd_result=rd,
        display_data_token=token,
        prepared_evaluation=(frame, request.evaluation),
    )
    cache = service.recording_cache
    if cache is not None and (rp is not None or rd is not None):
        from .prepared_results import save_prepared

        scoped = cache.for_frame(storage)
        saved = save_prepared(
            scoped,
            "plots",
            (request.display_step, variant),
            {
                "identity": _prepared_result_identity(sensing),
                "arrays": {k: v for k, v in sensing.items() if isinstance(v, np.ndarray)},
                "rp": rp,
                "rd": rd,
            },
        )
        if scoped._building and not saved:
            raise OSError("Could not save the extended sensing figure model")
    return result


def _has_requested_products(request, key):
    """Pure admission check for a foreground hit while another reader is busy."""
    _, view, variant, _, micro, *extra = key
    if (
        view in {"processing-inputs", "processing-frequency"}
        and request.processing_inspection is not None
        and not extra
    ):
        return True
    sensing = request.display_frame["sensing"]
    return all(
        name in sensing
        for name in _selected_names(sensing, view, variant, micro, extra[0] if extra else ())
    )


def _request_product_key(key: tuple) -> tuple:
    """Retain common maps once, extending that result for compact inspections."""
    step, view, variant, pinned, _micro, *extra = key
    if _shares_plot_data(view, _micro, extra[0] if extra else ()):
        return step, "analysis", variant, pinned, "auto"
    return key


def _shares_plot_data(view: str, micro: str, extra: tuple[str, ...]) -> bool:
    return (
        view in _SHARED_PLOT_VIEWS or (view == "micro-doppler" and micro != "saved-stft")
    ) and set(extra).issubset({"range_profile", "range_doppler", "range_doppler_angle"})


def _request_owner_keys(key: tuple) -> tuple[tuple, ...]:
    """Look for a resident superset without broadening a selective cache miss.

    Presentation is not product identity. Live/Tracking requests can borrow
    Current's maps and compact inspections, but must stay lightweight when no
    such owner exists. Keep variant and replay policy in every candidate key.
    """
    normalized = _request_product_key(key)
    step, view, variant, pinned, _micro, *extra = normalized
    if view in {"live", "tracking", "summary"} and set(extra[0] if extra else ()).issubset(
        {"range_profile", "range_doppler", "range_doppler_angle"}
    ):
        return (step, "analysis", variant, pinned, "auto"), normalized
    return (normalized,)


def _compatible_request_owner(key: tuple, owner: tuple, request: SensingUpdateRequest) -> bool:
    """Do not transfer an automatic map fallback to a different presentation.

    Metadata-only auto requests select primary, whereas Current may fall back
    to raw maps. Explicit variants and a request's own owner stay unambiguous.
    """
    return (
        key[1] not in {"live", "tracking", "summary"}
        or owner[1] == key[1]
        or request.variant == ("primary" if key[2] in {"auto", "clean"} else key[2])
    )


def load_dashboard_request(
    service: SensingService,
    step: int,
    view: str,
    variant: str,
    pin: bool,
    micro: str,
    extra_products: tuple[str, ...] = (),
) -> SensingUpdateRequest | None:
    """Read the selected view's products using their exact scene binding."""
    from .prepared_results import load_prepared, save_prepared

    requested_view = view
    input_kind = {
        "processing-inputs": "input-delay",
        "processing-frequency": "input-frequency",
    }.get(view)
    shared = _shares_plot_data(view, micro, extra_products)
    scene = service.dashboard_scene_frame(step)
    if scene is None:
        return None
    names = figure_requirements(
        view, variant=variant, micro_mode=micro, visible_products=extra_products
    ).product_names
    needs_angle = requested_view == "angular" or "range_doppler_angle" in extra_products
    names = tuple(name for name in names if name != "range_doppler_angle")
    if shared:
        service.dashboard_result_rows()
    cache = service.recording_cache if shared else None
    cached = None
    if cache is not None:
        # Prepared maps belong to the saved occurrence and its original binding.
        # A causal hold may display that occurrence at a later scene; the normal
        # resolver and identity comparison below still decide whether it is valid.
        hint = service.dashboard_result_cache_hint(
            step,
            pin_freshest=pin,
            scene_time_s=service._scene_time_s(scene),
            product_names=("config_resolved",),
        )
        if hint is not None:
            storage, bound_scene = hint
            cached = load_prepared(cache.for_frame(storage), "plots", (bound_scene, variant))
    if not isinstance(cached, dict) or not {"identity", "arrays", "rp", "rd"}.issubset(cached):
        cached = None
    processing_view = view.startswith("processing-")
    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=step,
        storage_frame_id=step,
        product_names=("config_resolved",) if cached is not None or input_kind else names,
        pin_freshest=pin,
        load_all_sensing_products=False,
        product_selector=(
            None
            if cached is not None or input_kind
            else lambda sensing: _selected_names(
                sensing, requested_view, variant, micro, extra_products
            )
        ),
    )
    if selection is None:
        return None
    if cache is not None:
        cache = cache.for_frame(selection.storage_frame_id)
    frame = selection.display_frame
    input_inspection = None
    if input_kind:
        from .processing_inspection import ProcessingInspection

        cache = service.recording_cache
        cache = cache.for_frame(selection.storage_frame_id) if cache is not None else None
        input_inspection = load_prepared(cache, input_kind, (selection.storage_frame_id, variant))
        if not isinstance(input_inspection, ProcessingInspection):
            input_inspection = None
    if cached is not None:
        # Compare the selected occurrence: hold policies may
        # select a different run/pair at the same scene position.
        required = _selected_names(frame["sensing"], requested_view, variant, micro, extra_products)
        if not _prepared_plot_supports(
            cached, frame["sensing"], required, variant, require_complete=False
        ):
            cached = None
            selection = service.resolve_display_sensing_frame(
                scene,
                scene_frame_id=step,
                storage_frame_id=step,
                product_names=names,
                pin_freshest=pin,
                load_all_sensing_products=False,
                product_selector=lambda sensing: _selected_names(
                    sensing, requested_view, variant, micro, extra_products
                ),
            )
            if selection is None:
                return None
            cache = service.recording_cache.for_frame(selection.storage_frame_id)
            frame = selection.display_frame
        else:
            if not _prepared_plot_supports(cached, frame["sensing"], required, variant):
                # A narrow ordinary owner must not mask a complete published
                # owner. Check normal typed candidates before reading signals.
                complete = load_prepared(
                    cache,
                    "plots",
                    (selection.bound_scene_frame_id, variant),
                    accept=lambda value: _prepared_plot_supports(
                        value, frame["sensing"], required, variant
                    ),
                )
                if complete is not None:
                    cached = complete
            frame = {**frame, "sensing": {**frame["sensing"], **cached["arrays"]}}
    original_sensing = frame["sensing"]
    sensing = _extend_sensing_products(
        service,
        selection.storage_frame_id,
        original_sensing,
        _selected_names(
            frame["sensing"],
            "live" if input_inspection is not None else requested_view,
            variant,
            micro,
            extra_products,
        ),
    )
    frame = {**frame, "sensing": sensing}
    # The resolver just detached this payload from its private, validated
    # product snapshot. No caller can have modified it between these calls.
    # Keep ordinary caller-owned frames on the public validation path.
    replay_identity = getattr(selection, "_replay_identity", None)
    owned_products = (
        cached is not None or getattr(replay_identity, "product_token", None) is not None
    )
    cached_models_complete = cached is not None and _prepared_plot_supports(
        cached,
        sensing,
        _selected_names(sensing, requested_view, variant, micro, extra_products),
        variant,
    )
    rp, rd = (
        (cached["rp"], cached["rd"])
        if cached_models_complete and sensing is original_sensing
        else (
            (
                (None, None)
                if not {"range_profile", "range_doppler"}.intersection(names)
                else (
                    service._dashboard_products_from_validated(sensing, variant)
                    if owned_products
                    else service.dashboard_products(frame, variant)
                )
            )
        )
    )
    kinds = service.dashboard_product_kinds(selection.storage_frame_id)
    if (
        rp is None
        and rd is None
        and not processing_view
        and view not in {"tracking", "summary", "live"}
        and bool({"range_profile", "range_doppler"}.intersection(names))
        and {"range_profile", "range_doppler"}.intersection(kinds)
        # A gated-out CPI can have no STFT columns while its recording history
        # still does. Its STFT mean policy need not exist as an RD variant.
        and not (view == "micro-doppler" and micro == "saved-stft")
    ):
        raise ValueError(f"No {variant} range or RD product in the selected result")
    inspection = None
    if processing_view:
        from .processing_inspection import inspect_processing

        inspection = input_inspection or inspect_processing(
            sensing,
            (
                "frequency"
                if view == "processing-frequency"
                else "inputs" if view == "processing-inputs" else "clutter"
            ),
            reconstruct=view == "processing-clutter-rebuild",
        )
        if input_kind and input_inspection is None:
            saved = save_prepared(
                cache, input_kind, (selection.storage_frame_id, variant), inspection
            )
            if cache is not None and cache._building and not saved:
                raise OSError("Could not save the selected input display model")
    cfg = sensing["config_resolved"]
    display_token = None
    if owned_products:
        display_token = _DisplayDataToken(sensing)
        display_token.require_complete = False
    request = SensingUpdateRequest(
        display_frame=frame,
        state_frame=scene,
        state_step=step,
        display_step=selection.bound_scene_frame_id,
        display_from_cache=selection.stale_for_current_scene,
        rp_result=rp,
        rd_result=rd,
        variant=(
            (rd or rp).metadata["variant"]
            if rd is not None or rp is not None
            else "primary" if variant in {"auto", "clean"} else variant
        ),
        range_mode=cfg["range_mode"],
        carrier_freq_hz=cfg["carrier_freq_hz"],
        prf_hz=cfg["prf_hz"],
        sampling_frequency=cfg["sampling_frequency"],
        replay_event=selection.replay_event,
        recording_managed=True,
        available_product_kinds=kinds,
        selective_products=True,
        processing_inspection=inspection,
        display_data_token=display_token,
    )
    request = replace(request, prepared_evaluation=(frame, request.evaluation))
    if shared:
        if cache is not None and (
            (cached is None and step == selection.bound_scene_frame_id)
            or (
                cached is not None
                and (sensing is not original_sensing or not cached_models_complete)
            )
        ):
            # Persist a validated selective owner after adding another view's
            # products. Its key remains the saved occurrence's original binding,
            # even when that occurrence is held at a later scene.
            saved = save_prepared(
                cache,
                "plots",
                (selection.bound_scene_frame_id, variant),
                {
                    "identity": _prepared_result_identity(sensing),
                    "arrays": {k: v for k, v in sensing.items() if isinstance(v, np.ndarray)},
                    "rp": rp,
                    "rd": rd,
                },
            )
            if cache._building and not saved:
                raise OSError("Could not save the sensing figure model")
        request = prepare_request_view(service, request, requested_view, selection=selection)
    if needs_angle:
        request = prepare_request_view(service, request, "angular")
    return request


def prepare_request_view(
    service: SensingService, request: SensingUpdateRequest, view: str, *, selection=None
) -> SensingUpdateRequest:
    """Extend one retained result with small, persisted inspection models.

    The session remains the sole RAM owner. Common maps are shared by Current,
    Detector, Filtering and Angle; full angular cubes only live during preparation.
    """
    if view not in _SHARED_PLOT_VIEWS or view in request.prepared_views:
        return request
    from .prepared_results import load_prepared, save_prepared

    cache = service.recording_cache
    storage = request.display_frame["_sensing_display_provenance"]["storage_frame_id"]
    if cache is not None:
        cache = cache.for_frame(storage)
    key = (storage, request.variant)
    changes = {}
    if view == "angular" and "range_doppler_angle" in (request.available_product_kinds or ()):
        from shared.sensing.angle import has_range_doppler_angle_variant

        from .prepared_angle import PreparedAngle, prepare_angular

        angle = None
        if has_range_doppler_angle_variant(request.display_frame["sensing"], request.variant):
            angle = load_prepared(cache, "angle", key)
            if not isinstance(angle, PreparedAngle):
                frame = service.dashboard_load_row(
                    storage,
                    figure_requirements(
                        "live",
                        variant=request.variant,
                        visible_products=("range_doppler_angle",),
                        descriptors=request.display_frame["sensing"].get("products", {}),
                    ).product_names,
                )
                angle = prepare_angular(frame["sensing"], request.variant) if frame else None
                if angle is not None:
                    save_prepared(cache, "angle", key, angle)
        changes["prepared_angular"] = angle
    elif view in {"diagnostics", "analysis-inspection"}:
        from .stage_inspector import DetectionStageBundle

        bundle = load_prepared(cache, "detector", key)
        if not isinstance(bundle, DetectionStageBundle):
            if selection is None:
                selection = service.resolve_display_sensing_frame(
                    request.state_frame,
                    scene_frame_id=request.state_step,
                    storage_frame_id=request.state_step,
                    product_names=("config_resolved",),
                    pin_freshest=bool(
                        request.display_frame["_sensing_display_provenance"]["pinned"]
                    ),
                )
            bundle = service.load_detection_stage_bundle(selection) if selection else None
            if bundle is not None:
                save_prepared(cache, "detector", key, bundle)
        if bundle is not None and request.replay_event is not None:
            bundle = service._cross_check_stage_bundle(bundle, request.replay_event)
        changes["detection_stage_bundle"] = bundle
    elif view == "processing-clutter":
        from .processing_inspection import ProcessingInspection, inspect_processing

        inspection = load_prepared(cache, "filtering", key)
        if not isinstance(inspection, ProcessingInspection):
            inspection = inspect_processing(request.display_frame["sensing"], "clutter")
            save_prepared(cache, "filtering", key, inspection)
        changes["processing_inspection"] = inspection
    ready = request.prepared_views | {view}
    if view in {"diagnostics", "analysis-inspection"}:
        ready |= {"diagnostics", "analysis-inspection"}
    return replace(request, **changes, prepared_views=ready)


def recording_history_cache_key(context, variant, include_signal_products=True, cache_policy=None):
    """Identify one recording overview, including its retained product policy."""
    policy = cache_policy or SensingCachePolicy()
    key = (
        "tracking-states-display-summary-v5",
        *context,
        variant,
        policy.history_results,
        policy.history_bytes,
        SIGNAL_TRUTH_VERSION,
        EVALUATION_CACHE_PROTOCOL,
        DEFAULT_RANGE_GATE_M,
        DEFAULT_VELOCITY_GATE_M_S,
    )
    return key if include_signal_products else (*key, "estimates-only")


def _decode_history_record(row: dict) -> HistoryRecord:
    """Restore the same owned representation on cold admission and cache reads."""
    row = dict(row)
    row["identity"] = tuple(row["identity"])
    row["detections"] = tuple(row["detections"])
    row["tracking_states"] = tuple(
        {
            **state,
            "source_key": tuple(state.get("source_key", ())),
            "state_labels": tuple(state.get("state_labels", ())),
        }
        for state in row.get("tracking_states", ())
    )
    row["truth_positions"] = tuple((tid, tuple(point)) for tid, point in row["truth_positions"])
    for name in (
        "spectrum",
        "doppler_hz",
        "range_m",
        "range_power",
        "micro_power",
        "micro_times_s",
        "micro_doppler_hz",
    ):
        row[name] = None if row[name] is None else np.asarray(row[name], dtype=float)
    return HistoryRecord(**row)


def _persisted_history_record(record: HistoryRecord) -> HistoryRecord:
    """Account for JSON's loss of shared source objects before budget admission.

    A cold record may alias its truth and evaluation metadata. JSON reconstructs
    independent containers, which must fit the same limit when reopened.
    """
    import json

    from .recording_cache import _json_value

    return _decode_history_record(json.loads(json.dumps(asdict(record), default=_json_value)))


def load_recording_history(
    service: SensingService,
    context: tuple[str, str, str],
    variant: str,
    cancelled: Event,
    include_signal_products: bool = True,
    *,
    cache_policy: SensingCachePolicy | None = None,
) -> RecordingHistory:
    """Collapse saved RD power without changing taper, mean removal or sampling.

    Only compact spectral/profile columns and saved estimates survive each read.
    Never retain an inventory of CIR or angle cubes. Long recordings retain the
    newest bounded set and expose that limit in the interface.
    """
    policy = cache_policy or SensingCachePolicy()
    rows = sorted(
        (
            row
            for row in service.dashboard_result_rows()
            if result_context(row[1]["sensing"]) == context
            and isinstance(
                row[1]["sensing"]["temporal"]["result"]["reference_time_s"], (float, int)
            )
            and np.isfinite(row[1]["sensing"]["temporal"]["result"]["reference_time_s"])
        ),
        key=lambda row: row[1]["sensing"]["temporal"]["result"]["reference_time_s"],
    )
    # Resolve repeated stored occurrences before applying a retention bound.
    unique = {row[1]["sensing"]["temporal"]["acquisition"]["acquisition_id"]: row for row in rows}
    rows = sorted(
        unique.values(), key=lambda row: row[1]["sensing"]["temporal"]["result"]["reference_time_s"]
    )
    available_span = (
        tuple(
            float(row[1]["sensing"]["temporal"]["result"]["reference_time_s"])
            for row in (rows[0], rows[-1])
        )
        if rows
        else None
    )
    cache = service.recording_cache
    cache_key = recording_history_cache_key(context, variant, include_signal_products, policy)
    cached = cache.load("overview", cache_key) if cache is not None else None
    if cached is not None:
        try:
            records = [_decode_history_record(row) for row in cached["records"]]
            if (
                len(records) > policy.history_results
                or sum(estimate_retained_bytes(r) for r in records) > policy.history_bytes
                or any(r.identity[:3] != context for r in records)
            ):
                raise ValueError("Cached overview has a different result context")
            service.report_read_progress("Cached overview", len(records), len(records))
            return RecordingHistory(
                tuple(records),
                truncated=cached["truncated"],
                prepared_evaluation=cached["evaluation_summary"],
                available_results=len(rows),
                available_time_span=available_span,
                byte_limit=policy.history_bytes,
                result_limit=policy.history_results,
            )
        except (KeyError, TypeError, ValueError):
            pass  # Disposable derived data; the validated recording is authoritative.
    records: list[HistoryRecord] = []
    identities: set[tuple[str, str, str, str]] = set()
    size = 0
    truncated = len(rows) > policy.history_results
    selected_rows = rows[-policy.history_results :]
    for index, (storage_id, _metadata) in enumerate(selected_rows):
        service.report_read_progress("Loading overview", index, len(selected_rows))
        if cancelled.is_set():
            return RecordingHistory(())
        if include_signal_products:
            frame, rp, rd = service.dashboard_history_products(storage_id, variant)
        else:
            frame = service.dashboard_load_row(storage_id, ("config_resolved",))
            rp = rd = None
        if frame is None:
            continue
        sensing = frame["sensing"]
        if result_context(sensing) != context:
            continue
        temporal = sensing["temporal"]
        identity = (*context, temporal["acquisition"]["acquisition_id"])
        if identity in identities:
            continue
        identities.add(identity)
        time_s = float(temporal["result"]["reference_time_s"])
        scene_id = temporal["display_binding"]["scene_frame_id"]
        pose = service.dashboard_scene_frame(scene_id) or {}
        detections = service.dashboard_world_detections({**pose, "sensing": sensing})
        evidence = evaluation_event(sensing, pose)
        track_snapshot = tracking_snapshot(sensing)
        track_states = track_snapshot.states
        # Project saved Cartesian states with the same bound pose as detections.
        # These rows are a presentation adapter, never detector observations.
        local_states = [
            state for state in track_states if state.get("position_frame") == "receiver local"
        ]
        if local_states:
            state_rows = [
                {
                    "track_id": state["track_id"],
                    **{f"track_{axis}_m": value for axis, value in zip("xyz", state["position_m"])},
                }
                for state in local_states
            ]
            projected = service.dashboard_world_detections(
                {**pose, "sensing": {**sensing, "detections": state_rows}}
            )
            world = {
                row["track_id"]: row["world_track_position"]
                for row in projected
                if row.get("world_track_position") is not None
            }
            track_states = tuple(
                {**state, "world_position_m": world.get(state["track_id"])}
                for state in track_states
            )
        # Cursor navigation uses the record's identity/time and saved track
        # states. Retain only the remaining browser/variant-selector fields;
        # sample provenance and full recipes belong to the selected result,
        # not every column of the recording overview.
        compact_metadata = {
            "config_resolved": {
                "antenna_rows": sensing.get("config_resolved", {}).get("antenna_rows"),
            },
            "products": {
                key: {
                    "kind": descriptor.get("kind"),
                    "variant": descriptor.get("variant"),
                    "processing": {
                        "slow_time_mean_removal_applied": descriptor.get("processing", {}).get(
                            "slow_time_mean_removal_applied"
                        ),
                    },
                }
                for key, descriptor in sensing.get("products", {}).items()
            },
            "detections": sensing.get("detections", []),
            "_display_detection_collection": sensing.get(
                "_display_detection_collection", "detections"
            ),
        }
        truth = evidence["truth"]
        positions = tuple(
            (t["target_id"], tuple(t["position_m"]))
            for t in (sensing.get("truth") or {}).get("targets", [])
            if truth is not None and t.get("valid") and t.get("lifecycle") == "active"
        )
        micro = saved_micro_doppler(sensing, variant) if include_signal_products else None
        record = HistoryRecord(
            evaluation=evidence,
            storage_id=storage_id,
            scene_frame_id=scene_id,
            time_s=time_s,
            identity=identity,
            spectrum=np.sum(rd.power_linear, axis=-1) if rd is not None else None,
            doppler_hz=np.array(rd.doppler_axis_hz) if rd is not None else None,
            range_m=np.array(rp.range_axis_m) if rp is not None else None,
            range_power=np.array(rp.power_linear) if rp is not None else None,
            detections=tuple(detections),
            truth=truth,
            truth_positions=positions,
            tracking_states=track_states,
            tracking_states_complete=track_snapshot.complete_states,
            sensing_metadata=compact_metadata,
            micro_power=np.array(micro["power"]) if micro else None,
            micro_times_s=np.array(micro["times_s"]) if micro else None,
            micro_doppler_hz=np.array(micro["doppler_hz"]) if micro else None,
            micro_hop_s=micro["hop_s"] if micro else None,
            micro_recipe=micro["recipe"] if micro else None,
            cadence_s=(
                float(sensing["config_resolved"]["acquisition"]["sample_period_s"])
                * int(sensing["config_resolved"]["acquisition"]["cpi_hop_samples"])
                if sensing["config_resolved"]["acquisition"]["sample_period_s"] is not None
                else None
            ),
        )
        record = _persisted_history_record(record)
        size += estimate_retained_bytes(record)
        records.append(record)
        while size > policy.history_bytes and records:
            size -= estimate_retained_bytes(records.pop(0))
            truncated = True
    result = RecordingHistory(
        tuple(records),
        truncated=truncated,
        available_results=len(rows),
        available_time_span=available_span,
        byte_limit=policy.history_bytes,
        result_limit=policy.history_results,
    )
    _ = result.evaluation_summary
    if cache is not None and not cancelled.is_set():
        service.report_read_progress("Saving overview cache", len(records), len(records))
        saved = cache.save(
            "overview",
            {
                "records": [asdict(r) for r in records],
                "truncated": truncated,
                "evaluation_summary": result.evaluation_summary,
            },
            cache_key,
        )
        if cache._building and not saved:
            raise OSError("Could not save the sensing overview")
    service.report_read_progress("Overview ready", len(records), len(records))
    return result


class DashboardSession:
    """Two workers for selection, recording overview and bounded result prefetch.

    The panel polls futures on its Qt timer. Changing scenarios retires this
    session; a late completion cannot access or modify the replacement window.
    """

    def __init__(
        self,
        provider: Any,
        *,
        previous: DashboardSession | None = None,
        source_path=None,
        saved_result_path=None,
        prepared=None,
        cache_byte_limit: int | None = None,
        cache_entry_limit: int | None = None,
        cache_policy: SensingCachePolicy | None = None,
    ) -> None:
        from .run_summary_session import RunSummarySession
        from .saved_tracking import SavedTrackingLoader

        self.provider = provider
        self.cache_policy = cache_policy or (
            previous.cache_policy
            if previous is not None and previous.source_identity == self.provider_identity(provider)
            else SensingCachePolicy()
        )
        self._cache_byte_limit = cache_byte_limit
        self._cache_entry_limit = cache_entry_limit
        self.preparation_summary = None
        self.saved_tracking = SavedTrackingLoader(source_path, saved_result_path)
        self.source_identity = self.provider_identity(provider)
        self.run_summary = RunSummarySession(
            provider,
            previous=(
                previous.run_summary
                if previous is not None and previous.source_identity == self.source_identity
                else None
            ),
        )
        self._summary_context = None
        self._summary_source_token = None
        self.prepared = (
            prepared
            if (prepared is not None and prepared.scene.matches_provider(provider))
            else None
        )
        self._cancelled = Event()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sensing-dashboard")
        proxy = SimpleNamespace(frame_source=SimpleNamespace(provider=provider))
        self.service = SensingService(proxy)
        self.history_service = self.service
        self.pending: Future | None = None
        self.pending_key: tuple | None = None
        self._request_lock = RLock()
        self._diagnostics_lock = RLock()
        self._diagnostics_enabled = False
        self._diagnostics = None
        self._diagnostic_requested_at = None
        self._prefetch_pending: Future | None = None
        self._prefetch_options = None
        self._prefetch_seen: set[tuple] = set()
        self.history_pending: Future | None = None
        self.history_key: tuple | None = None
        self.history_inventory: tuple | None = None
        self.history: RecordingHistory | None = None
        self._history_signal_products = True
        self.overview_pending: Future | None = None
        self.overview_key: tuple | None = None
        self.overview_inventory: tuple | None = None
        self.overview: MotionOverview | None = None
        self.overview_progress = ("Loading spectral overview", 0, 0)
        self._capability_rows = None
        self._stft_variants_by_context: dict[tuple, frozenset[str]] = {}
        self._overview_job = None
        self._overview_cancelled = Event()
        self._cursor_history: RecordingHistory | None = None
        self._cursor_rows: dict[int | str, HistoryRecord] = {}
        self.navigation = ()
        self._navigation_context = None
        self._navigation_rows = None
        self._navigation_inventory = None
        self._compact_cursor_rows = None
        self._compact_cursor_names = None
        self._compact_cursor_sorted = ()
        self._compact_cursor_times = ()
        self._compact_cursor_exact = {}
        self._compact_cursor_contexts = frozenset()
        self.progress = ("Loading result", 0, 0)
        # Cache validated, immutable display requests, not Qt objects. Common
        # maps and prepared inspection models share this byte budget.
        self._request_cache: OrderedDict[tuple, tuple[SensingUpdateRequest, int]] = OrderedDict()
        self._request_cache_bytes = 0
        self._request_cache_token = None
        self._request_cache_inventory = None
        if (
            previous is not None
            and previous.source_identity == self.source_identity
            and previous.pending is None
            and previous.history_pending is None
            and previous.overview_pending is None
            and previous.cache_policy == self.cache_policy
            and (previous._prefetch_pending is None or previous._prefetch_pending.done())
        ):
            # Reopening the same immutable recording can reuse completed data,
            # never a previous window's pending future or Qt object.
            self.service = previous.service
            self.history_service = self.service
            self._request_cache = previous._request_cache
            self._request_cache_bytes = previous._request_cache_bytes
            self._request_cache_token = previous._request_cache_token
            self._request_cache_inventory = previous._request_cache_inventory
            if cache_byte_limit is None:
                self._cache_byte_limit = previous._cache_byte_limit
            if cache_entry_limit is None:
                self._cache_entry_limit = previous._cache_entry_limit
            self.preparation_summary = previous.preparation_summary
            if (
                self.prepared is None
                and previous.prepared is not None
                and previous.prepared.scene.matches_provider(provider)
            ):
                self.prepared = previous.prepared
            if (
                self.saved_tracking.source == previous.saved_tracking.source
                and self.saved_tracking.result_path == previous.saved_tracking.result_path
            ):
                self.saved_tracking = previous.saved_tracking
            if previous.history_pending is None:
                self.history = previous.history
                self.history_key = previous.history_key
                self.history_inventory = previous.history_inventory
                self._history_signal_products = previous._history_signal_products
            self.overview = previous.overview
            self.overview_key = previous.overview_key
            self.overview_inventory = previous.overview_inventory
        self.service.configure_background_reader(self._cancelled, self._set_progress)

    @property
    def cache_byte_limit(self) -> int:
        return self._cache_byte_limit or self.cache_policy.request_bytes

    @property
    def cache_entry_limit(self) -> int:
        return self._cache_entry_limit or self.cache_policy.request_entries

    def _set_progress(self, stage: str, completed: int, total: int) -> None:
        """Publish immutable worker progress for the Qt polling timer."""
        self.progress = (stage, completed, total)

    def set_diagnostics_enabled(self, enabled: bool) -> None:
        """Collect bounded counters and request timings only on explicit opt-in."""
        with self._diagnostics_lock:
            self._diagnostics_enabled = bool(enabled)
            if enabled and self._diagnostics is None:
                self._diagnostics = {
                    "ram_hits": 0,
                    "ram_misses": 0,
                    "ram_evictions": 0,
                    "result_reuses": 0,
                    "completed_requests": 0,
                    "last_queue_ms": None,
                    "last_result_ms": None,
                    "last_request_ms": None,
                }
            if not enabled:
                self._diagnostic_requested_at = None
        self.service.set_replay_telemetry_enabled(enabled)

    def _record_diagnostics(self, counter: str | None = None, **values) -> None:
        if not self._diagnostics_enabled:
            return
        with self._diagnostics_lock:
            if self._diagnostics_enabled and self._diagnostics is not None:
                if counter is not None:
                    self._diagnostics[counter] += 1
                self._diagnostics.update(values)

    def diagnostics_snapshot(self) -> dict:
        """Read existing counters without waiting on product I/O or scanning arrays.

        Occupancy is a near-in-time sample, not an atomic transaction with the
        worker. Counter collection pauses when the diagnostic window is inactive.
        """
        with self._diagnostics_lock:
            metrics = dict(self._diagnostics) if self._diagnostics is not None else None
        cache = self.service.recording_cache
        return {
            "enabled": self._diagnostics_enabled,
            "request_entries": len(self._request_cache),
            "request_bytes": self._request_cache_bytes,
            "request_entry_limit": self.cache_entry_limit,
            "request_byte_limit": self.cache_byte_limit,
            "requests": metrics,
            "reader": self.service.replay_telemetry_snapshot().to_manifest(),
            "disk_hits": cache.hits if cache is not None else None,
        }

    @staticmethod
    def provider_identity(provider: Any) -> tuple:
        info = getattr(provider, "info", None)
        return (
            id(provider),
            getattr(info, "generation_id", None),
            getattr(info, "frame_set_id", None),
            getattr(provider, "_generation_epoch", None),
        )

    def request(self, key: tuple) -> None:
        """Submit at most one selection; callers coalesce changes while it runs."""
        if self._cancelled.is_set() or self.pending is not None:
            return
        self.pending_key = key
        self._diagnostic_requested_at = perf_counter() if self._diagnostics_enabled else None
        self.pending = self._executor.submit(self._load_request, key)

    def _cached_request(self, key: tuple, *, prefetch: bool = False) -> SensingUpdateRequest | None:
        """Serialize product preparation; speculative reads never evict visited results."""
        if self._request_lock.acquire(blocking=False):
            try:
                return self._cached_request_locked(key, prefetch=prefetch)
            finally:
                self._request_lock.release()
        # A complete owned snapshot needs no reader/service work. Do not make
        # a foreground cache hit wait behind an in-flight speculative disk read.
        token = (
            self.provider_identity(self.provider),
            id(getattr(self.provider, "frame_ids_snapshot", None)),
        )
        if not prefetch and not self._cancelled.is_set() and token == self._request_cache_token:
            retained = next(
                (
                    entry
                    for k in _request_owner_keys(key)
                    if (entry := self._request_cache.get(k)) is not None
                    and _compatible_request_owner(key, k, entry[0])
                ),
                None,
            )
            if (
                retained is not None
                and _has_requested_products(retained[0], key)
                and (key[1] not in _SHARED_PLOT_VIEWS or key[1] in retained[0].prepared_views)
                and (
                    not key[5:]
                    or "range_doppler_angle" not in key[5]
                    or "angular" in retained[0].prepared_views
                )
            ):
                self._record_diagnostics("ram_hits")
                return retained[0]
        with self._request_lock:
            return self._cached_request_locked(key, prefetch=prefetch)

    def _cached_request_locked(self, key: tuple, *, prefetch: bool) -> SensingUpdateRequest | None:
        requested_key = key
        key = _request_product_key(key)
        snapshot = getattr(self.provider, "frame_ids_snapshot", None)
        token = (self.provider_identity(self.provider), id(snapshot))
        if token != self._request_cache_token:
            self._request_cache.clear()
            self._request_cache_bytes = 0
            self._request_cache_token = token
        if snapshot is not None:
            key = next(
                (
                    k
                    for k in _request_owner_keys(key)
                    if k in self._request_cache
                    and _compatible_request_owner(key, k, self._request_cache[k][0])
                ),
                key,
            )
        cached = self._request_cache.get(key) if snapshot is not None else None
        reused = (
            self._reuse_result_request(key) if cached is None and snapshot is not None else None
        )
        if cached is not None:
            self._record_diagnostics("ram_hits")
            request = cached[0]
            self._request_cache.move_to_end(key)
        elif reused is not None:
            self._record_diagnostics("result_reuses")
            owner_key, request = reused
            key = owner_key
        else:
            self._record_diagnostics("ram_misses")
            request = load_dashboard_request(self.service, *requested_key)
        before = request
        if request is not None:
            request = _extend_request_products(self.service, request, requested_key)
        if request is not None and key[1] == "analysis" and not key[5:]:
            request = prepare_request_view(self.service, request, requested_key[1])
            if requested_key[5:] and "range_doppler_angle" in requested_key[5]:
                request = prepare_request_view(self.service, request, "angular")
        if (
            request is not None
            and (cached is None and reused is None or request is not before)
            and snapshot is not None
            and not self._cancelled.is_set()
            and token
            == (
                self.provider_identity(self.provider),
                id(getattr(self.provider, "frame_ids_snapshot", None)),
            )
        ):
            cache_entry = replace(
                request,
                display_data_token=(
                    request.display_data_token
                    if isinstance(request.display_data_token, _DisplayDataToken)
                    and request.display_data_token.sensing is request.display_frame["sensing"]
                    else _DisplayDataToken(request.display_frame["sensing"])
                ),
            )
            size = estimate_retained_bytes(cache_entry)
            if reused is not None:
                # Store new view data on the existing result owner. The caller
                # still receives its newly rebound scene context.
                key = owner_key
                cached = self._request_cache[key]
                owner = cached[0]
                owner_frame = {**owner.display_frame, "sensing": request.display_frame["sensing"]}
                cache_entry = replace(
                    owner,
                    display_frame=owner_frame,
                    rp_result=request.rp_result,
                    rd_result=request.rd_result,
                    display_data_token=request.display_data_token,
                    prepared_evaluation=(owner_frame, owner.evaluation),
                    prepared_angular=request.prepared_angular,
                    detection_stage_bundle=request.detection_stage_bundle,
                    processing_inspection=request.processing_inspection,
                    prepared_views=request.prepared_views,
                )
                size = estimate_retained_bytes(cache_entry)
            old_size = cached[1] if cached is not None else 0
            fits = size <= self.cache_byte_limit
            if prefetch:
                fits = fits and (
                    self._request_cache_bytes + size - old_size <= self.cache_byte_limit
                    and (cached is not None or len(self._request_cache) < self.cache_entry_limit)
                )
            if fits:
                if reused is None:
                    request = cache_entry
                self._request_cache[key] = (cache_entry, size)
                self._request_cache_inventory = self.service.dashboard_inventory_token()
                self._request_cache_bytes += size - old_size
                while (
                    len(self._request_cache) > self.cache_entry_limit
                    or self._request_cache_bytes > self.cache_byte_limit
                ):
                    _, (_, evicted_size) = self._request_cache.popitem(last=False)
                    self._request_cache_bytes -= evicted_size
                    self._record_diagnostics("ram_evictions")
        return request

    def prefetch_next(self, key: tuple) -> None:
        """Prepare one result between foreground reads, within the existing budget.

        Warm the selected view without evicting visited data or waiting for
        a complete recording.
        """
        if self._prefetch_pending is not None:
            if not self._prefetch_pending.done():
                return
            try:
                self._prefetch_pending.result()
            except Exception:
                # A foreground read reports invalid/unavailable data through
                # the usual UI; speculative work must not interrupt playback.
                pass
            self._prefetch_pending = None
        if (
            self._cancelled.is_set()
            or self.pending is not None
            or key[1] not in _SHARED_PLOT_VIEWS
            or getattr(self.provider, "frame_ids_snapshot", None) is None
            or len(self._request_cache) >= self.cache_entry_limit
            or self._request_cache_bytes >= self.cache_byte_limit
        ):
            return
        options = (self._request_cache_token, key[1:])
        if options != self._prefetch_options:
            self._prefetch_seen.clear()
            self._prefetch_options = options
        views = [key[1]]
        locations = list(self.navigation[: self.cache_entry_limit])
        current = next((i for i, row in enumerate(locations) if row.scene_id == key[0]), 0)
        locations = locations[current:] + locations[:current]
        for location in locations:
            for view in views:
                candidate = (location.scene_id, view, *key[2:])
                retained = self._request_cache.get(_request_product_key(candidate))
                if candidate in self._prefetch_seen or (
                    retained is not None
                    and view in retained[0].prepared_views
                    and (
                        not candidate[5:]
                        or "range_doppler_angle" not in candidate[5]
                        or "angular" in retained[0].prepared_views
                    )
                ):
                    continue
                self._prefetch_seen.add(candidate)
                self._prefetch_pending = self._executor.submit(
                    self._cached_request, candidate, prefetch=True
                )
                return

    def _load_request(self, key: tuple) -> SensingUpdateRequest | None:
        """Prepare products and navigation on the worker, including empty scene bindings."""
        started = perf_counter() if self._diagnostics_enabled else None
        if started is not None and self._diagnostic_requested_at is not None:
            self._record_diagnostics(last_queue_ms=(started - self._diagnostic_requested_at) * 1000)
        request = self._cached_request(key)
        if request is None and (
            key[1] == "summary" or (key[1] == "micro-doppler" and key[4] in {"auto", "saved-stft"})
        ):
            request = self._recording_overview_request(key)
        if started is not None and self._diagnostics_enabled:
            self._record_diagnostics(last_result_ms=(perf_counter() - started) * 1000)
        if (
            request is not None
            and not request.recording_overview_only
            and key[1] not in {"summary", "live"}
        ):
            if (
                self.prepared is not None
                and self.prepared.scene.matches_provider(self.provider)
                and self.saved_tracking.result_path is None
            ):
                saved, error = self.prepared.tracking, self.prepared.tracking_error
            else:
                saved, error = self.saved_tracking.load(self._cancelled)
            request = replace(
                request,
                tracking_history=saved.history if saved else None,
                saved_tracking=saved,
                saved_tracking_error=error,
                source_identity=self.source_identity,
            )
            if saved is not None:
                from .fusion_inspection import native_observation_request

                acquisition = request.display_frame["sensing"]["temporal"]["acquisition"]
                lineage = saved.reader["sources"][saved.sensor_id].get("lineage", [])
                binding = next(
                    (
                        row
                        for row in lineage
                        if row.get("acquisition_id") == acquisition["acquisition_id"]
                    ),
                    None,
                )
                if binding is not None:
                    request = native_observation_request(request, binding)
        if request is not None:
            if key[1] in {"summary", "live"} or request.recording_overview_only:
                request = replace(request, source_identity=self.source_identity)
            self._navigation_context = result_context(request.display_frame["sensing"])
        rows = self.service.dashboard_result_rows()
        self._refresh_navigation(rows)
        if key[1] == "summary":
            # A single-cohort overview has no playback-dependent data. Multiple
            # runs/pairs retain ordinary selection so their summaries cannot mix.
            contexts = {
                result_context(frame["sensing"]) for _, frame in _completed_summary_rows(rows)
            }
            self._summary_context = next(iter(contexts)) if len(contexts) == 1 else None
            self._summary_source_token = (
                self._request_cache_token,
                self.service.dashboard_inventory_token(),
            )
        if started is not None and self._diagnostics_enabled:
            self._record_diagnostics(
                "completed_requests", last_request_ms=(perf_counter() - started) * 1000
            )
        return request

    def _recording_overview_request(self, key: tuple) -> SensingUpdateRequest | None:
        """Load one recording's contract without inventing a current result.

        Whole-recording Motion and Summary remain available during warm-up or gaps.
        A representative result identifies saved products or summary metadata; its
        exact acquisition and scene binding remain intact. Multiple cohorts
        require an ordinary bound selection rather than an arbitrary choice.
        """
        rows = self.service.dashboard_result_rows()
        if key[1] == "summary":
            rows = _completed_summary_rows(rows)
        else:
            rows = tuple(
                (storage, frame)
                for storage, frame in rows
                if any(
                    item.get("kind") in {"range_doppler", "micro_doppler"}
                    for item in frame["sensing"].get("products", {}).values()
                )
            )
        contexts = {result_context(frame["sensing"]) for _, frame in rows}
        if len(contexts) != 1 or self._cancelled.is_set():
            return None
        scene = self.service.dashboard_scene_frame(key[0])
        if scene is None:
            return None
        for _, frame in rows:
            sensing = frame["sensing"]
            bound_scene = sensing["temporal"]["display_binding"]["scene_frame_id"]
            request = load_dashboard_request(
                self.service, bound_scene, key[1], key[2], False, key[4]
            )
            if request is not None:
                return replace(
                    request,
                    state_frame=scene,
                    state_step=key[0],
                    recording_overview_only=True,
                )
        return None

    def _reuse_result_request(self, key: tuple) -> tuple[tuple, SensingUpdateRequest] | None:
        """Rebind a cached result without rereading or validating its products.

        The metadata reader still selects the winner by physical time across all
        runs/pairs. Scene pose/time and separately saved tracking remain current.
        Exact and held policies may share products only after selecting that same
        saved result; their replay binding and provenance are rebuilt separately.
        """
        from shared.sensing.replay import ReplaySceneContext, rebind_replay_event

        step, view, _variant, pinned, _micro, *extra = key
        if view == "summary" or not self._request_cache:
            return None
        owners = _request_owner_keys(key)
        candidates = [
            (k, r)
            for k, (r, _) in reversed(self._request_cache.items())
            if any(k[1:3] == owner[1:3] and k[4:] == owner[4:] for owner in owners)
            and _compatible_request_owner(key, k, r)
        ]
        if not candidates:
            return None
        scene = self.service.dashboard_scene_frame(step)
        if scene is None:
            return None
        timestamp = self.service._scene_time_s(scene)
        if timestamp is None:
            return None
        names = _dashboard_product_names(view, extra[0] if extra else ())
        rows = self.service.dashboard_result_rows()
        if self._request_cache_inventory != self.service.dashboard_inventory_token():
            self._request_cache.clear()
            self._request_cache_bytes = 0
            return None
        if not pinned:
            rows = self.service._exact_scene_result_rows(rows, step)
        eligible = [
            (storage, frame)
            for storage, frame in rows
            if self.service._eligible_dashboard_result(frame, step, timestamp, hold=pinned)
            and self.service._declares_any_sensing_product(frame, names)
        ]
        if not eligible:
            return None
        storage, _ = max(eligible, key=lambda row: self.service._selection_key(row[1], row[0]))
        for cache_key, previous in candidates:
            event = previous.replay_event
            if event is None or event.scene_binding.storage_frame_id != storage:
                continue
            binding = event.scene_binding
            try:
                rebound = rebind_replay_event(
                    event,
                    ReplaySceneContext(
                        storage_frame_id=storage,
                        bound_scene_frame_id=binding.bound_scene_frame_id,
                        current_scene_frame_id=step,
                        bound_scene_time_s=binding.bound_scene_time_s,
                        current_scene_time_s=timestamp,
                        pinned=pinned,
                    ),
                )
            except ValueError:
                # An edited bound-scene clock requires the normal resolver.
                continue
            frame = self.service._compose_bound_display_frame(
                scene,
                previous.display_frame,
                bound_scene_frame=previous.display_frame,
                storage_frame_id=storage,
                bound_scene_frame_id=binding.bound_scene_frame_id,
                current_scene_frame_id=step,
                pinned=pinned,
            )
            # Only scene context changed. Keep the exact owned sensing mapping
            # so its scoped validation token cannot become a mismatched wrapper.
            frame["sensing"] = previous.display_frame["sensing"]
            provenance = {
                **previous.display_frame["_sensing_display_provenance"],
                **frame["_sensing_display_provenance"],
            }
            provenance["replay_disposition"] = rebound.preflight.disposition
            provenance["replay_event_identity"] = {
                **provenance["replay_event_identity"],
                "scene_binding": rebound.scene_binding.to_manifest(),
            }
            frame["_sensing_display_provenance"] = provenance
            self._request_cache.move_to_end(cache_key)
            return cache_key, replace(
                previous,
                display_frame=frame,
                state_frame=scene,
                state_step=step,
                display_from_cache=binding.bound_scene_frame_id != step,
                replay_event=rebound,
                prepared_evaluation=(frame, previous.evaluation),
            )
        return None

    def can_reuse_summary(self, request) -> bool:
        """Skip per-frame preparation for an unchanged, single-cohort overview."""
        snapshot = getattr(self.provider, "frame_ids_snapshot", None)
        return (
            snapshot is not None
            and self.pending is None
            and request is not None
            and self._summary_context == result_context(request.display_frame["sensing"])
            and self._summary_source_token
            == (
                (self.provider_identity(self.provider), id(snapshot)),
                self.service.dashboard_inventory_token(),
            )
        )

    def _refresh_navigation(self, rows) -> None:
        """Retain navigation while its authoritative inventory and cohort agree."""
        inventory = (self.service.dashboard_inventory_token(), self._navigation_context)
        if rows is not self._navigation_rows or inventory != self._navigation_inventory:
            self.navigation = result_locations(rows, self._navigation_context)
            self._navigation_rows = rows
            self._navigation_inventory = inventory

    def compact_motion_cursor(
        self,
        scene_id: int | str,
        variant: str,
        families: frozenset[str],
        *,
        pin_freshest: bool = False,
        scene_time_s: float | None = None,
    ) -> CompactMotionCursor | None:
        """Select a ready overview cursor using the ordinary result ordering.

        The complete metadata index supplies occurrence identity and physical
        time, never detections or evaluation. A missing causal result is an
        explicit cursor state; a stale source or different cohort needs the
        ordinary reader instead. No source projections or replay are performed.
        """
        overview = self.overview
        variant = "primary" if variant in {"auto", "clean"} else variant
        if (
            self._cancelled.is_set()
            or self.pending is not None
            or self.overview_pending is not None
            or getattr(self.provider, "frame_ids_snapshot", None) is None
            or self.provider_identity(self.provider) != self.source_identity
            or overview is None
            or not families
            or overview.variant != variant
            or self.overview_key != (*overview.context, variant, tuple(sorted(families)))
            or self.overview_inventory != self.service.dashboard_inventory_token()
            or not families.issubset(overview.families)
            or any(
                overview.families[f].coverage.state not in {"complete", "unavailable"}
                for f in families
            )
            or type(scene_time_s) not in (int, float)
            or not math.isfinite(scene_time_s)
        ):
            return None
        rows = self.service.cached_dashboard_result_rows()
        if rows is None:
            return None
        # Motion's normal demand includes this required JSON field; sparse STFT
        # columns must not change which current acquisition wins selection.
        names = ("config_resolved",)
        if rows is not self._compact_cursor_rows or names != self._compact_cursor_names:
            selected = []
            exact = {}
            for storage, frame in rows:
                sensing = frame["sensing"]
                temporal = sensing["temporal"]
                bound = temporal["display_binding"]["scene_frame_id"]
                if not self.service._eligible_dashboard_result(
                    frame, bound, 0.0, hold=False
                ) or not self.service._declares_any_sensing_product(frame, names):
                    continue
                order = self.service._selection_key(frame, storage)
                location = ResultLocation(
                    order[0],
                    bound,
                    storage,
                    result_context(sensing),
                    temporal["acquisition"]["acquisition_id"],
                )
                selected.append((order, location))
                if bound not in exact or order > exact[bound][0]:
                    exact[bound] = (order, location)
            selected.sort(key=lambda item: item[0])
            self._compact_cursor_sorted = tuple(item[1] for item in selected)
            self._compact_cursor_times = tuple(item[0][0] for item in selected)
            self._compact_cursor_exact = {bound: item[1] for bound, item in exact.items()}
            self._compact_cursor_contexts = frozenset(r.context for _, r in selected)
            self._compact_cursor_rows, self._compact_cursor_names = rows, names
        if pin_freshest:
            index = bisect_right(self._compact_cursor_times, scene_time_s) - 1
            result = self._compact_cursor_sorted[index] if index >= 0 else None
        else:
            result = self._compact_cursor_exact.get(scene_id)
        if (
            result is not None
            and result.context != overview.context
            or result is None
            and self._compact_cursor_contexts != {overview.context}
            or self.service.cached_dashboard_result_rows() is not rows
        ):
            return None
        return CompactMotionCursor(scene_id, float(scene_time_s), result)

    def motion_cursor(
        self,
        scene_id: int | str,
        *,
        pin_freshest: bool = False,
        scene_time_s: float | None = None,
    ) -> HistoryRecord | None:
        """Reuse a validated overview column; never guess missing scene bindings."""
        history = self.history
        if (
            self.pending is not None
            or self.history_pending is not None
            or history is None
            or history.error
        ):
            return None
        storage_id = self.service.dashboard_exact_result_id(
            scene_id, pin_freshest=pin_freshest, scene_time_s=scene_time_s
        )
        if self.history_inventory != self.service.dashboard_inventory_token():
            return None
        if history is not self._cursor_history:
            self._cursor_history = history
            self._cursor_rows = {
                r.storage_id: r for r in history.records if r.storage_id is not None
            }
        record = self._cursor_rows.get(storage_id)
        if record is None or record.spectrum is None or record.scene_frame_id != scene_id:
            return None
        return record

    def request_history(
        self, sensing: dict, variant: str, *, include_signal_products: bool = True
    ) -> None:
        """Prepare a recording once for its run, pair and explicit variant."""
        key = (*result_context(sensing), variant)
        inventory = self.service.dashboard_inventory_token()
        upgrade = include_signal_products and not getattr(self, "_history_signal_products", True)
        if self.history_pending is not None or (
            key == self.history_key and inventory == self.history_inventory and not upgrade
        ):
            return
        if key != self.history_key or upgrade:
            self.history = None
        self.history_key = key
        self.history_inventory = inventory
        self._history_signal_products = include_signal_products
        self.history_pending = self._executor.submit(
            load_recording_history,
            self.history_service,
            key[:3],
            variant,
            self._cancelled,
            include_signal_products,
            cache_policy=self.cache_policy,
        )

    def request_overview(self, sensing, variant, families: frozenset[str]) -> None:
        """Coalesce family changes without letting late data replace new demand."""
        key = (*result_context(sensing), variant, tuple(sorted(families)))
        inventory = self.service.dashboard_inventory_token()
        wanted = (key, inventory, self.provider_identity(self.provider))
        if self._cancelled.is_set():
            return
        if self._overview_job is not None and self._overview_job != wanted:
            self._overview_cancelled.set()
        self.overview_key = key
        if self.overview_pending is not None:
            return
        if not families:
            return
        if (
            self.overview is not None
            and self.overview_inventory == inventory
            and self.overview.context == key[:3]
            and self.overview.variant == variant
            and families.issubset(self.overview.families)
            and (
                frozenset(self.overview.families) == families
                or all(
                    self.overview.families[name].coverage.state in {"complete", "unavailable"}
                    for name in families
                )
            )
        ):
            return
        self.overview = None
        self._overview_cancelled = Event()
        self._overview_job = wanted
        self.overview_progress = ("Loading spectral overview", 0, 0)
        self.overview_pending = self._executor.submit(
            load_motion_overview,
            self.service,
            key[:3],
            variant,
            families,
            self._overview_cancelled,
            byte_limit=self.cache_policy.overview_bytes,
            progress=lambda *value: setattr(self, "overview_progress", value),
        )

    def poll_overview(self) -> bool:
        """Accept only a finished overview for the still-current source and demand."""
        future = self.overview_pending
        if future is None or not future.done():
            return False
        self.overview_pending = None
        job, self._overview_job = self._overview_job, None
        error = None
        try:
            result = future.result()
        except CancelledError:
            return False
        except (OSError, RuntimeError, MemoryError) as exc:
            # Storage/source retirement and allocation failures belong to the
            # requested GUI scope. Full preparation still propagates failures.
            error = exc
        if (
            self._cancelled.is_set()
            or job is None
            or self._overview_cancelled.is_set()
            or job
            != (
                self.overview_key,
                self.service.dashboard_inventory_token(),
                self.provider_identity(self.provider),
            )
        ):
            return False
        if error is not None:
            from .motion_overview import failed_motion_overview

            key = job[0]
            result = failed_motion_overview(
                key[:3], key[3], key[4], self.cache_policy.overview_bytes, error
            )
        self.overview, self.overview_inventory = result, job[1]
        return True

    def with_recording_data(self, request: SensingUpdateRequest) -> SensingUpdateRequest:
        """Attach matching data owners without changing the selected result."""
        key = (*result_context(request.display_frame["sensing"]), request.variant)
        history = self.history if key == self.history_key else None
        overview = self.overview
        if overview is not None and (*overview.context, overview.variant) != key:
            overview = None
        return replace(
            request,
            recording_history=history,
            motion_overview=overview,
            recording_stft_variants=self._recording_stft_variants(request),
        )

    def _recording_stft_variants(self, request) -> frozenset[str] | None:
        """Index sparse authored STFT availability once, without starting I/O."""
        rows = (
            self.service.cached_dashboard_result_rows()
            if self.source_identity == self.provider_identity(self.provider)
            and (not request.source_identity or request.source_identity == self.source_identity)
            else None
        )
        if rows is None:
            self._capability_rows = None
            self._stft_variants_by_context = {}
            return None
        if rows is not self._capability_rows:
            variants: dict[tuple, set[str]] = {}
            for _, frame in _completed_summary_rows(rows):
                sensing = frame["sensing"]
                for descriptor in sensing.get("products", {}).values():
                    if (
                        descriptor.get("kind") == "micro_doppler"
                        and descriptor.get("shape")
                        and all(size > 0 for size in descriptor["shape"])
                    ):
                        variants.setdefault(result_context(sensing), set()).add(
                            descriptor["variant"]
                        )
            self._stft_variants_by_context = {
                context: frozenset(values) for context, values in variants.items()
            }
            self._capability_rows = rows
        return self._stft_variants_by_context.get(
            result_context(request.display_frame["sensing"]), frozenset()
        )

    def close(self) -> None:
        """Cancel queued work; in-flight storage reads release their own leases."""
        self._cancelled.set()
        self._overview_cancelled.set()
        self.run_summary.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
