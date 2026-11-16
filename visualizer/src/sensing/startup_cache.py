"""Prepare a complete, disposable sensing playback cache before GUI startup."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

from .cache_policy import SensingCachePolicy
from .dashboard_data import (
    DashboardSession,
    load_recording_history,
    recording_history_cache_key,
    result_context,
)
from .prepared_results import load_prepared
from .recording_cache import RecordingCache, _change_time, _temporary_file, prepared_cache_root
from .run_summary_data import load_run_summary
from .scene_recording import SCENE_CACHE_KEY, prepare_scene_recording

_VERSION = 2
_PLOT_COVERAGE = "authored-current-products-v1"


def discard_prepared_dashboard(provider) -> None:
    """Release an unclaimed warm session when its scenario is abandoned."""
    session = getattr(provider, "_prepared_sensing_dashboard", None)
    if session is not None:
        del provider._prepared_sensing_dashboard
        session.close()


def _new_session(provider, cancelled, progress, retain_session, cache_policy=None):
    policy = cache_policy or SensingCachePolicy.for_preparation()
    session = DashboardSession(
        provider,
        source_path=provider.source,
        cache_byte_limit=policy.request_bytes if retain_session else None,
        cache_entry_limit=policy.request_entries if retain_session else None,
        cache_policy=policy,
    )
    session.service.configure_background_reader(cancelled, progress)
    return session


def _selection_plan(session):
    rows = session.service.dashboard_result_rows()
    contexts = {}
    scenes = {}
    for storage, frame in rows:
        sensing = frame["sensing"]
        context = result_context(sensing)
        if not all(isinstance(value, str) and value for value in context) or not (
            session.service._eligible_dashboard_result(frame, None, float("inf"), hold=True)
        ):
            continue
        variants = {
            product.get("variant")
            for product in (sensing.get("products") or {}).values()
            if product.get("kind") in {"range_profile", "range_doppler"}
            and product.get("variant") in {"primary", "raw"}
        }
        spectral_variants = variants | {
            product.get("variant")
            for product in (sensing.get("products") or {}).values()
            if product.get("kind") == "micro_doppler"
            and product.get("variant") in {"primary", "raw"}
        }
        # STFT may have a mean-removal variant with no corresponding RD/profile.
        # Prepare its recording overview without inventing per-result map views.
        contexts.setdefault(context, set()).update(spectral_variants or {"primary"})
        scene = sensing["temporal"]["display_binding"]["scene_frame_id"]
        if isinstance(scene, int) and session.provider.has_frame(scene):
            order = session.service._selection_key(frame, storage)
            if scene not in scenes or order > scenes[scene][0]:
                scenes[scene] = (order, variants or {"primary"})
    return rows, contexts, scenes


def _retain_initial_history(session, context, variant, history):
    if session.history is None:
        session.history = history
        session.history_key = (*context, variant)
        session.history_inventory = session.service.dashboard_inventory_token()
        session._history_signal_products = False
        _ = history.tracking_plot_data


def _prepare_overview(session, context, variant, cancelled, progress, retain):
    """Use the same exact family loader as an ordinary Motion request."""
    from .motion_overview import SPECTRAL_FAMILIES, load_motion_overview

    families = frozenset(SPECTRAL_FAMILIES)
    overview = load_motion_overview(
        session.service,
        context,
        variant,
        families,
        cancelled,
        byte_limit=session.cache_policy.overview_bytes,
        progress=progress,
    )
    if retain and session.overview is None:
        session.overview = overview
        session.overview_key = (*context, variant, tuple(sorted(families)))
        session.overview_inventory = session.service.dashboard_inventory_token()
    return _overview_capabilities(overview, context, variant)


def _overview_capabilities(overview, context, variant):
    return [
        {
            "context": list(context),
            "variant": variant,
            "family": family,
            "state": item.coverage.state,
            "available_results": item.coverage.available_results,
            "loaded_results": item.coverage.loaded_results,
            "loaded_time_span": item.loaded_time_span,
            "byte_limit": item.coverage.byte_limit,
            "reason": item.coverage.reason,
        }
        for family, item in overview.families.items()
    ]


class _CachedComponentReader:
    """Verify durable components through normal readers without repairing them."""

    _building = False

    def __init__(self, cache):
        self._cache = cache

    def load(self, *args, **kwargs):
        return self._cache.load(*args, **kwargs)

    def load_arrays(self, *args, **kwargs):
        return self._cache.load_arrays(*args, **kwargs)

    def unchanged(self):
        return self._cache.unchanged()

    def save(self, *_args, **_kwargs):
        raise OSError("The requested Motion component was not durably cached")

    save_arrays = save


def prepare_motion_cache(
    provider,
    cancelled: Event,
    progress=lambda *_: None,
    *,
    motion_mode="auto",
    cache_policy=None,
    include_scene=False,
):
    """Persist only requested Motion families, optionally the explicit 3D scene.

    Ordinary independent component files are reusable by both the GUI and full
    preparation. This never publishes or replaces a full-bundle ready pointer.
    Returned coverage distinguishes a durable limited artifact from full span.
    """
    from ..services.sensing_service import SensingService
    from .motion_overview import load_motion_overview

    if motion_mode not in {"auto", "all-ranges", "saved-stft"}:
        raise ValueError("Motion preparation requires all-ranges or saved-stft")
    if cancelled.is_set():
        raise CancelledError()
    policy = cache_policy or SensingCachePolicy()
    source_cache = RecordingCache.for_provider(provider)
    if source_cache is None:
        raise OSError("This recording cannot use a persistent sensing cache")
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    service.configure_background_reader(cancelled, progress)
    rows, contexts, _scenes = _selection_plan(SimpleNamespace(service=service, provider=provider))
    if not contexts:
        raise ValueError("No completed sensing cohort is available for Motion preparation")
    families = frozenset(
        {"saved-stft" if motion_mode == "saved-stft" else "summed-rd", "range-profile"}
    )
    requests = []
    for context, variants in contexts.items():
        # The GUI resolves Auto to primary/raw before requesting its overview.
        # Primary also retains the authored fixed-STFT fallback semantics.
        for variant in sorted(variants | {"primary"}):
            if cancelled.is_set():
                raise CancelledError()
            overview = load_motion_overview(
                service,
                context,
                variant,
                families,
                cancelled,
                byte_limit=policy.overview_bytes,
                progress=progress,
            )
            if any(item.coverage.state == "error" for item in overview.families.values()):
                raise ValueError("A requested Motion family could not be prepared")
            requests.append((context, variant))
    if include_scene:
        scene = prepare_scene_recording(provider, cancelled, progress)
        if not scene.disk_hit:
            if source_cache.load("scene", SCENE_CACHE_KEY) is None:
                raise OSError("Could not persist the requested 3D sensing scene")
            if not prepare_scene_recording(provider, cancelled, progress).disk_hit:
                raise OSError("Could not verify the requested 3D sensing scene")

    def missing_component(*_args, **_kwargs):
        raise OSError("The requested Motion component was not durably cached")

    def authored_metadata(storage, products):
        if tuple(products) != ("config_resolved",):
            raise OSError("Motion cache verification may read authored metadata only")
        return service.dashboard_load_row(storage, products)

    verifier = SimpleNamespace(
        recording_cache=_CachedComponentReader(service.recording_cache),
        dashboard_result_rows=lambda: rows,
        dashboard_load_row=authored_metadata,
        dashboard_spectral_products=missing_component,
    )
    capabilities = []
    # Verify after all writes, including optional scene writes: ordinary quota
    # pruning may otherwise remove an earlier family before preparation ends.
    for context, variant in requests:
        checked = load_motion_overview(
            verifier,
            context,
            variant,
            families,
            cancelled,
            byte_limit=policy.overview_bytes,
            progress=progress,
        )
        capabilities.extend(_overview_capabilities(checked, context, variant))
    if cancelled.is_set():
        raise CancelledError()
    if not source_cache.unchanged():
        raise RuntimeError("Recording changed while preparing Motion")
    complete = all(row["state"] in {"complete", "unavailable"} for row in capabilities)
    progress(
        "Motion cache ready" if complete else "Motion cache prepared with limits",
        len(requests),
        len(requests),
    )
    return {
        "disk_ready": True,
        "coverage_complete": complete,
        "motion_mode": "saved-stft" if motion_mode == "saved-stft" else "all-ranges",
        "scene": bool(include_scene),
        "motion": capabilities,
    }


def _prepare_summary(session, context, cancelled, retain):
    from .dashboard_evaluation import DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S

    result = load_run_summary(session.service, context, cancelled)
    if result.cancelled:
        raise CancelledError()
    if retain and session.run_summary.data is None:
        session.run_summary.data = result
        session.run_summary.key = (
            tuple(context),
            session.service.dashboard_inventory_token(),
            (DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S, None, False),
        )
    return {
        "context": list(context),
        "acquisitions": result.acquisition_count,
        "evaluated": result.evaluated_count,
        "settings": "default gates; all saved outputs",
    }


def _retain_playback_session(provider, session, scene, scenes, cancelled, progress):
    """Keep bounded canonical result owners; playback resolves its own provenance."""
    from .preparation import PreparedSensing

    progress("Preparing saved tracking", 0, 1)
    tracking, error = session.saved_tracking.load(cancelled)
    if tracking is not None:
        _ = tracking.history.tracking_plot_data
    if scene is not None:
        session.prepared = PreparedSensing(scene, tracking, error)
    requested = {
        (scene_id, view, variant)
        for scene_id, (_, variants) in scenes.items()
        for variant in variants
        for view in ("analysis", "tracking")
    }
    shared_views = {"analysis", "diagnostics", "processing-clutter", "angular"}
    resident = set()
    for key, (request, _size) in session._request_cache.items():
        if cancelled.is_set():
            raise CancelledError()
        event = request.replay_event
        if event is None or key[1] not in {"analysis", "tracking"}:
            continue
        binding = event.scene_binding
        bound = binding.bound_scene_frame_id
        planned = scenes.get(bound)
        if planned is None or key[2] not in planned[1]:
            continue
        if (
            session.service._selection_key(request.display_frame, binding.storage_frame_id)
            != planned[0]
        ):
            continue
        # A half-sample CPI center can lie after its bound scene's clock.
        # Exact owners still serve later causal requests through normal replay
        # rebinding; neither counting nor handoff needs to load those requests.
        resident.add((bound, "tracking", key[2]))
        # A common map may fit while an inspection model exceeds the budget.
        # Unavailable products also mark their view prepared, avoiding false misses.
        if key[1] == "analysis" and shared_views.issubset(request.prepared_views):
            resident.add((bound, "analysis", key[2]))
    retained = len(requested & resident)
    cache = session.service.recording_cache
    bundle = cache._prepared_root if cache is not None else None
    try:
        bundle_bytes = (
            sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())
            if bundle is not None
            else None
        )
    except OSError:
        bundle_bytes = None
    session.preparation_summary = {
        "disk_ready": True,
        "disk_scope": "Saved maps, Motion families, Summary and inspection; complex Inputs on demand",
        "requested_requests": len(requested),
        "retained_requests": retained,
        "residency_scope": "Canonical saved-result views; causal playback selects and rebinds owners",
        "request_bytes": session._request_cache_bytes,
        "request_byte_limit": session.cache_byte_limit,
        "request_entry_limit": session.cache_entry_limit,
        "requests_complete": retained == len(requested),
        "history_context": session.history_key,
        "history_scope": "Bounded details for initial completed cohort; exact Motion stored separately",
        "bundle_bytes": bundle_bytes,
    }
    if bundle is not None:
        try:
            capabilities = json.loads((bundle / "complete.json").read_text())["capabilities"]
            session.preparation_summary["disk_components"] = capabilities
            session.preparation_summary["disk_coverage_complete"] = all(
                row["state"] in {"complete", "unavailable"} for row in capabilities["motion"]
            )
        except (OSError, ValueError, TypeError, KeyError):
            session.preparation_summary["disk_ready"] = False
    if cancelled.is_set():
        raise CancelledError()
    if cache is not None and not cache.unchanged():
        raise RuntimeError("Recording changed while preparing resident sensing figures")
    discard_prepared_dashboard(provider)
    session.close()
    provider._prepared_sensing_dashboard = session
    label = (
        "Sensing playback ready"
        if retained == len(requested)
        else "Sensing playback partly resident"
    )
    progress(label, retained, len(requested))


def _warm_published_requests(session, scenes, cancelled, progress):
    """Warm an initial prefix without evicting figures already ready for playback."""
    for position, (scene_id, (_, variants)) in enumerate(sorted(scenes.items())):
        progress("Loading sensing figures into RAM", position, len(scenes))
        for variant in sorted(variants):
            if cancelled.is_set():
                raise CancelledError()
            if (
                len(session._request_cache) >= session.cache_entry_limit
                or session._request_cache_bytes >= session.cache_byte_limit
            ):
                return
            for view in (
                "analysis",
                "diagnostics",
                "processing-clutter",
                "angular",
                "tracking",
            ):
                if cancelled.is_set():
                    raise CancelledError()
                request = session._cached_request(
                    (scene_id, view, variant, True, "auto"), prefetch=True
                )
                if request is not None and not any(
                    owner.display_data_token is request.display_data_token
                    and request.prepared_views.issubset(owner.prepared_views)
                    for owner, _size in session._request_cache.values()
                ):
                    # A map or added inspection model did not fit. Held scenes
                    # may share an earlier owner, so check its immutable token
                    # rather than requiring an entry under this scene's key.
                    return


def _warm_published_cache(provider, cancelled, progress, *, include_scene=True, cache_policy=None):
    scene = prepare_scene_recording(provider, cancelled, progress) if include_scene else None
    session = _new_session(provider, cancelled, progress, True, cache_policy)
    try:
        _rows, contexts, scenes = _selection_plan(session)
        if contexts:
            context, variants = next(iter(contexts.items()))
            variant = "primary" if "primary" in variants else min(variants)
            history = load_recording_history(
                session.service,
                context,
                variant,
                cancelled,
                include_signal_products=False,
                cache_policy=session.cache_policy,
            )
            _retain_initial_history(session, context, variant, history)
            _prepare_overview(session, context, variant, cancelled, progress, True)
            _prepare_summary(session, context, cancelled, True)
        _warm_published_requests(session, scenes, cancelled, progress)
        _retain_playback_session(provider, session, scene, scenes, cancelled, progress)
    finally:
        session.close()


def _component_protocols():
    """Declared dependencies make old component keys invalidate preparation."""
    from . import component_cache, motion_overview, prepared_results, recording_cache
    from .dashboard_evaluation import EVALUATION_CACHE_PROTOCOL
    from .run_summary_data import SUMMARY_PROTOCOL

    return {
        "cache": recording_cache.CACHE_VERSION,
        "numeric": recording_cache.ARRAY_CACHE_VERSION,
        "component": component_cache.COMPONENT_VERSION,
        "models": prepared_results.PREPARATION_VERSION,
        "spectral": motion_overview.SPECTRAL_PROTOCOL,
        "summary": SUMMARY_PROTOCOL,
        "evaluation": EVALUATION_CACHE_PROTOCOL,
        "scene": list(SCENE_CACHE_KEY),
        "saved_views": "common-saved-figures-v1",
        "history": list(recording_history_cache_key(("", "", ""), "primary", False)),
    }


def _artifact_stamp(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, _change_time(path, stat)]


def _coverage_policy(policy):
    return {
        "overview_bytes": policy.overview_bytes,
        "history_bytes": policy.history_bytes,
        "history_results": policy.history_results,
    }


def _reusable_model_manifest(root, signature):
    """Admit only a wholly verified current bundle, ignoring new history limits."""
    if root is None:
        return None
    try:
        marker = json.loads((root / "complete.json").read_text())
        previous_policy = SensingCachePolicy(**marker["coverage_policy"])
        if not _complete(
            root,
            signature,
            require_scene=False,
            cache_policy=previous_policy,
            require_plot_coverage=False,
        ):
            return None
        models = marker["models"]
        return models if isinstance(models, list) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _indexed_plot_complete(value, sensing, variant):
    """Check coverage of a source/stamp-verified model against the compact index.

    The index omits product IDs in some versions. It still supplies occurrence,
    authored keys, variants and shapes. Supplied IDs remain authoritative; normal
    dashboard reads retain their stricter full-descriptor product-ID check.
    """
    import numpy as np

    from .dashboard_data import _prepared_result_identity, _selected_names

    if not isinstance(value, dict) or value.get("identity") != _prepared_result_identity(sensing):
        return False
    arrays = value.get("arrays")
    if not isinstance(arrays, dict) or not {"rp", "rd"}.issubset(value):
        return False
    descriptors = sensing.get("products", {})
    names = _selected_names(sensing, "analysis", variant)
    for name in names:
        if name == "config_resolved":
            continue
        array = arrays.get(name)
        descriptor = descriptors.get(name, {})
        if not isinstance(array, np.ndarray) or tuple(descriptor.get("shape", ())) != array.shape:
            return False
    for kind, field in (("range_profile", "rp"), ("range_doppler", "rd")):
        required = [name for name in names if descriptors.get(name, {}).get("kind") == kind]
        if not required:
            continue
        if len(required) != 1:
            return False
        name = required[0]
        descriptor = descriptors[name]
        metadata = getattr(value[field], "metadata", {})
        if (
            not isinstance(metadata, dict)
            or metadata.get("data_key") != name
            or metadata.get("variant") != descriptor.get("variant")
            or (
                "product_id" in descriptor
                and metadata.get("product_id") != descriptor["product_id"]
            )
        ):
            return False
    return True


def _promote_models(session, models, scenes, cancelled, progress):
    """Promote complete typed owners; leave only insufficient models for repair."""
    from .prepared_results import save_prepared

    if models is None:
        return None
    verified_root = session.service.recording_cache._prepared_root
    if verified_root is None:
        return None
    expected = {(scene, variant) for scene, (_, variants) in scenes.items() for variant in variants}
    indexed = dict(session.service.dashboard_result_rows())
    seen = set()
    reusable = {}
    readers = {}
    allowed = {"plots", "detector", "filtering", "angle", "input-delay", "input-frequency"}
    try:
        for position, record in enumerate(models):
            if cancelled.is_set():
                raise CancelledError()
            scene, variant, storage = record["scene"], record["variant"], record["storage"]
            item = (scene, variant)
            if (
                item not in expected
                or item in seen
                or not isinstance(record["unavailable"], list)
                or not isinstance(record["deferred"], list)
            ):
                return None
            seen.add(item)
            selected = session.service.dashboard_exact_result_id(
                scene, product_names=("config_resolved",)
            )
            if selected != storage or storage not in indexed:
                return None
            artifacts = record["artifacts"]
            if (
                not isinstance(artifacts, list)
                or not any(artifact.get("kind") == "plots" for artifact in artifacts)
                or len({artifact["kind"] for artifact in artifacts}) != len(artifacts)
            ):
                return None
            complete = True
            sensing = indexed[storage]["sensing"]
            scoped = session.service.recording_cache.for_frame(storage)
            if scoped.paths not in readers:
                reader = RecordingCache(scoped.paths, root=verified_root)
                if reader.signature != scoped.signature:
                    return None
                readers[scoped.paths] = reader
            reader = readers[scoped.paths]
            for artifact in artifacts:
                kind = artifact["kind"]
                key = tuple(artifact["key"])
                if kind not in allowed or key != (
                    (scene, variant) if kind == "plots" else (storage, variant)
                ):
                    return None
                value = load_prepared(
                    reader,
                    kind,
                    key,
                    accept=(
                        (lambda value: _indexed_plot_complete(value, sensing, variant))
                        if kind == "plots"
                        else None
                    ),
                )
                if value is None:
                    complete = False
                elif not save_prepared(scoped, kind, key, value):
                    raise OSError("Could not promote a verified sensing figure")
            if complete:
                reusable[item] = record
            progress("Reusing verified sensing figures", position + 1, len(models))
        return reusable if seen == expected else None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _complete(
    root: Path,
    signature: str,
    *,
    require_scene=True,
    cache_policy=None,
    require_plot_coverage=True,
) -> bool:
    try:
        metadata = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        files = metadata["files"]
        stamps = metadata["artifact_stamps"]
        capabilities = metadata["capabilities"]
        return (
            metadata["version"] == _VERSION
            and metadata["signature"] == signature
            and metadata["protocols"] == _component_protocols()
            and (not require_plot_coverage or metadata.get("plot_coverage") == _PLOT_COVERAGE)
            and isinstance(capabilities, dict)
            and capabilities.get("version") == 1
            and isinstance(capabilities.get("motion"), list)
            and isinstance(capabilities.get("summary"), list)
            and isinstance(metadata.get("models"), list)
            and metadata["coverage_policy"]
            == _coverage_policy(cache_policy or SensingCachePolicy.for_preparation())
            and type(metadata.get("scene_prepared", True)) is bool
            and (not require_scene or metadata.get("scene_prepared", True))
            and isinstance(files, list)
            and isinstance(stamps, dict)
            and set(stamps) == set(files)
            and len(files) == len(set(files))
            and all(
                isinstance(name, str)
                and Path(name).name == name
                and (root / name).is_file()
                and not (root / name).is_symlink()
                and stamps[name] == _artifact_stamp(root / name)
                for name in files
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_startup_cache(
    provider,
    cancelled: Event,
    progress=lambda *_: None,
    *,
    retain_session=False,
    include_scene=True,
    cache_policy=None,
) -> bool:
    """Build primary and alternate saved playback figures with one source reader.

    Return True for a newly built bundle and False for an already complete one.
    A per-recording SQLite write lock serializes builds across processes. The
    manifest appears only after all required files have been written and checked.
    Figure-only bundles omit 3D overlays and cannot satisfy a later full request.
    """
    policy = cache_policy or SensingCachePolicy.for_preparation()
    cache = RecordingCache.for_provider(provider)
    if cache is None:
        raise OSError("This recording cannot use a persistent sensing cache")
    recordings = cache.root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    target = prepared_cache_root(cache.root, cache.signature)
    if target is not None and _complete(
        target, cache.signature, require_scene=include_scene, cache_policy=policy
    ):
        if retain_session:
            _warm_published_cache(
                provider, cancelled, progress, include_scene=include_scene, cache_policy=policy
            )
        return False

    connection = sqlite3.connect(recordings / f"{cache.signature[:20]}.lock", timeout=0.25)
    try:
        while True:
            if cancelled.is_set():
                raise CancelledError()
            try:
                connection.execute("BEGIN IMMEDIATE")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                progress("Waiting for another sensing cache preparation", 0, 0)
        target = prepared_cache_root(cache.root, cache.signature)
        if target is not None and _complete(
            target, cache.signature, require_scene=include_scene, cache_policy=policy
        ):
            if retain_session:
                _warm_published_cache(
                    provider, cancelled, progress, include_scene=include_scene, cache_policy=policy
                )
            return False

        stage = recordings / f"bundle_{uuid4().hex[:16]}"
        reusable_models = _reusable_model_manifest(target, cache.signature)
        stage.mkdir()
        provider._sensing_cache_build_root = stage
        published = False
        session = None
        try:
            scene_recording = (
                prepare_scene_recording(provider, cancelled, progress) if include_scene else None
            )
            if cancelled.is_set():
                raise CancelledError()
            session = _new_session(provider, cancelled, progress, retain_session, policy)
            rows, contexts, scenes = _selection_plan(session)
            capabilities = {
                "version": 1,
                "motion": [],
                "summary": [],
                "saved_views": [
                    "analysis",
                    "diagnostics",
                    "processing-clutter",
                    "angular",
                    "tracking",
                ],
                "deferred_views": ["complex-inputs", "complex-frequency", "explicit-reprocessing"],
                "scene": include_scene,
            }
            build_cache = session.service.recording_cache
            if build_cache is None:
                raise OSError("Sensing cache is unavailable")

            for context, variants in contexts.items():
                progress("Preparing run summary", 0, len(rows))
                capabilities["summary"].append(
                    _prepare_summary(session, context, cancelled, retain_session)
                )
                for variant in sorted(variants):
                    progress("Preparing exact Motion overview", 0, len(rows))
                    capabilities["motion"].extend(
                        _prepare_overview(
                            session, context, variant, cancelled, progress, retain_session
                        )
                    )
                    progress("Preparing sensing history", 0, len(rows))
                    history = load_recording_history(
                        session.service,
                        context,
                        variant,
                        cancelled,
                        include_signal_products=False,
                        cache_policy=policy,
                    )
                    if retain_session:
                        _retain_initial_history(session, context, variant, history)
                    key = recording_history_cache_key(context, variant, False, policy)
                    if not build_cache._path("overview", key).is_file():
                        raise OSError("Could not save the sensing history cache")
            reused_models = (
                _promote_models(session, reusable_models, scenes, cancelled, progress) or {}
            )
            models = list(reused_models.values())
            if reused_models:
                for model in models:
                    for artifact in model["artifacts"]:
                        view = {
                            "input-delay": "processing-inputs",
                            "input-frequency": "processing-frequency",
                        }.get(artifact["kind"])
                        if view and view not in capabilities["saved_views"]:
                            capabilities["saved_views"].append(view)
            pending_scenes = [
                (scene, (order, pending))
                for scene, (order, variants) in sorted(scenes.items())
                if (
                    pending := {
                        variant for variant in variants if (scene, variant) not in reused_models
                    }
                )
            ]
            for position, (scene, (_, variants)) in enumerate(pending_scenes):
                if cancelled.is_set():
                    raise CancelledError()
                progress("Preparing sensing figures", position, len(pending_scenes))
                for variant in sorted(variants):
                    request = session._cached_request((scene, "analysis", variant, False, "auto"))
                    if request is None:
                        raise OSError(f"Could not bind saved sensing result for frame {scene}")
                    storage = request.display_frame["_sensing_display_provenance"][
                        "storage_frame_id"
                    ]
                    scoped = build_cache.for_frame(storage)
                    from .dashboard_data import _prepared_plot_supports, _selected_names

                    sensing = request.display_frame["sensing"]
                    names = _selected_names(sensing, "analysis", variant)
                    if (
                        load_prepared(
                            scoped,
                            "plots",
                            (scene, variant),
                            accept=lambda value: _prepared_plot_supports(
                                value, sensing, names, variant
                            ),
                        )
                        is None
                    ):
                        raise OSError(f"Could not save sensing figures for frame {scene}")
                    model = {
                        "scene": scene,
                        "storage": storage,
                        "variant": variant,
                        "artifacts": [{"kind": "plots", "key": [scene, variant]}],
                        "unavailable": [],
                        "deferred": [],
                    }
                    models.append(model)
                    for view, kind, attribute in (
                        ("diagnostics", "detector", "detection_stage_bundle"),
                        ("processing-clutter", "filtering", "processing_inspection"),
                        ("angular", "angle", "prepared_angular"),
                    ):
                        detailed = session._cached_request((scene, view, variant, False, "auto"))
                        if detailed is not None and getattr(detailed, attribute) is not None:
                            key = (storage, detailed.variant)
                            if load_prepared(scoped, kind, key) is None:
                                raise OSError(f"Could not save {view} for frame {scene}")
                            model["artifacts"].append({"kind": kind, "key": list(key)})
                        else:
                            model["unavailable"].append(view)
                    from .figure_requirements import preparable_saved_views, supported_saved_views

                    descriptors = request.display_frame["sensing"].get("products", {})
                    safe_views = preparable_saved_views(descriptors)
                    supported_views = supported_saved_views(descriptors)
                    for view, kind in (
                        ("processing-inputs", "input-delay"),
                        ("processing-frequency", "input-frequency"),
                    ):
                        if view not in safe_views:
                            model["deferred" if view in supported_views else "unavailable"].append(
                                view
                            )
                            continue
                        detail = session._cached_request((scene, view, variant, False, "auto"))
                        if detail is None or detail.processing_inspection is None:
                            model["unavailable"].append(view)
                            continue
                        key = (storage, variant)
                        if load_prepared(scoped, kind, key) is None:
                            raise OSError(f"Could not save {view} for frame {scene}")
                        model["artifacts"].append({"kind": kind, "key": list(key)})
                        if view not in capabilities["saved_views"]:
                            capabilities["saved_views"].append(view)
                    if retain_session:
                        session._cached_request((scene, "tracking", variant, False, "auto"))

            if include_scene and not build_cache._path("scene", SCENE_CACHE_KEY).is_file():
                raise OSError("Could not save the prepared sensing scene")
            if cancelled.is_set():
                raise CancelledError()
            if not cache.unchanged():
                raise RuntimeError("Recording changed while preparing its sensing cache")
            files = sorted(p.name for p in stage.iterdir() if p.suffix in {".gz", ".npz"})
            (stage / "complete.json").write_text(
                json.dumps(
                    {
                        "version": _VERSION,
                        "plot_coverage": _PLOT_COVERAGE,
                        "signature": cache.signature,
                        "protocols": _component_protocols(),
                        "coverage_policy": _coverage_policy(policy),
                        "capabilities": capabilities,
                        "models": models,
                        "files": files,
                        "artifact_stamps": {name: _artifact_stamp(stage / name) for name in files},
                        "scene_prepared": include_scene,
                    }
                ),
                encoding="utf-8",
            )
            pointer, stream = _temporary_file(recordings)
            try:
                with stream:
                    stream.write(stage.name.encode("utf-8"))
                os.replace(pointer, recordings / f"{cache.signature[:20]}.ready")
                published = True
            finally:
                pointer.unlink(missing_ok=True)
            if retain_session:
                del provider._sensing_cache_build_root
                # The retained reader now uses the published bundle for fallback
                # and ordinary repair writes, never its staging directory.
                session.service.recording_cache = RecordingCache.for_provider(provider)
                if reused_models:
                    _warm_published_requests(session, scenes, cancelled, progress)
                _retain_playback_session(
                    provider, session, scene_recording, scenes, cancelled, progress
                )
            else:
                limited = any(
                    row["state"] not in {"complete", "unavailable"}
                    for row in capabilities["motion"]
                )
                progress(
                    "Sensing cache prepared with limits" if limited else "Sensing cache ready",
                    len(scenes),
                    len(scenes),
                )
            return True
        finally:
            if session is not None:
                session.close()
            if hasattr(provider, "_sensing_cache_build_root"):
                del provider._sensing_cache_build_root
            if not published:
                shutil.rmtree(stage, ignore_errors=True)
    finally:
        connection.close()
