"""Transient MPC selection and selected-path flow animation.

The service deliberately owns no application state.  It is created only for an
active MPC Explorer session, borrows that session's canonical catalog, and
sends a small selected-path snapshot directly to a capable renderer.  Frame
pipeline updates and bulk MPC geometry are never involved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal

from ..renderers.mpc_path_inspection import MpcPathInspectionSnapshot
from ..renderers.protocol import renderer_capabilities
from .mpc_interaction_style_service import (
    MPC_UNKNOWN_COLOR,
    build_mpc_type_palette,
    colorize_mpc_interaction_types,
    mpc_interaction_label,
)

MPC_FLOW_INTERVAL_MS = 33
MPC_FLOW_LOOP_SECONDS = 1.5
_MIN_FLOW_LENGTH = 1.0e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MpcSelectionIdentity:
    """Stable identity for one frame-local canonical path selection."""

    frame_token: Any
    canonical_path_id: int


@dataclass(frozen=True, slots=True)
class MpcSelectedPathDetails:
    """Small renderer-neutral detail snapshot for the selected path."""

    identity: MpcSelectionIdentity
    origin: str
    tx_id: Optional[int]
    rx_id: Optional[int]
    path_loss_db: Optional[float]
    delay_ns: Optional[float]
    interaction_count: int
    interaction_types: tuple[int, ...]
    interaction_labels: tuple[str, ...]
    material_ids: Optional[tuple[int, ...]]
    material_names: tuple[str, ...]
    geometric_length_m: float
    is_filtered: bool
    is_rendered: bool
    render_status: str
    delay_is_estimated: Optional[bool]
    path_loss_is_estimated: Optional[bool]

    @property
    def canonical_path_id(self) -> int:
        """Return the frame-local canonical path ID."""
        return self.identity.canonical_path_id

    @property
    def frame_token(self) -> Any:
        """Return the opaque accepted-frame token."""
        return self.identity.frame_token


class MpcSelectionService(QObject):
    """Own one transient MPC selection without touching the frame pipeline."""

    selectionChanged = Signal(object)
    detailsChanged = Signal(object)

    def __init__(
        self,
        visualizer: Any = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        parent: Optional[QObject] = None,
    ) -> None:
        """Create an idle service; no renderer work starts before presentation."""
        if parent is None and isinstance(visualizer, QObject):
            parent = visualizer
        super().__init__(parent)
        self.visualizer = visualizer
        self._clock = clock
        self._active = False
        self._shutdown = False
        self._frame_token: Any = None
        self._presented_packet_identity: Optional[int] = None
        self._catalog: Any = None
        self._selection: Optional[MpcSelectionIdentity] = None
        self._details: Optional[MpcSelectedPathDetails] = None
        self._snapshot: Optional[MpcPathInspectionSnapshot] = None
        self._flow_started_at = 0.0
        self._callback_renderer: Any = None
        self._overlay_renderer: Any = None

        self._flow_timer = QTimer(self)
        self._flow_timer.setInterval(MPC_FLOW_INTERVAL_MS)
        self._flow_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._flow_timer.timeout.connect(self._on_flow_timeout)

    @property
    def is_active(self) -> bool:
        """Whether the service currently borrows an accepted frame catalog."""
        return self._active and not self._shutdown

    @property
    def frame_token(self) -> Any:
        """Return the currently accepted opaque frame token."""
        return self._frame_token

    @property
    def catalog(self) -> Any:
        """Return the borrowed active catalog, or ``None`` while inactive."""
        return self._catalog

    @property
    def presented_packet_identity(self) -> Optional[int]:
        """Identity of the packet whose renderer submission was accepted."""
        return self._presented_packet_identity

    @property
    def selection(self) -> Optional[MpcSelectionIdentity]:
        """Return the current selection identity."""
        return self._selection

    @property
    def selected_path_id(self) -> Optional[int]:
        """Return the selected frame-local path ID, if any."""
        if self._selection is None:
            return None
        return self._selection.canonical_path_id

    @property
    def details(self) -> Optional[MpcSelectedPathDetails]:
        """Return the current selected-path detail snapshot."""
        return self._details

    @property
    def flow_timer_active(self) -> bool:
        """Expose timer state for lifecycle diagnostics and focused tests."""
        return self._flow_timer.isActive()

    def set_presented_frame(
        self,
        token: Any,
        catalog: Any,
        render_packet: Any = None,
        *,
        frame_changed: bool = False,
    ) -> None:
        """Adopt one successfully presented frame.

        Token equality, rather than the caller's advisory ``frame_changed``
        value, controls selection lifetime.  Reinstalling a catalog for the
        same token preserves and refreshes the selected canonical path.
        ``render_packet`` is accepted for the Explorer port contract but is
        intentionally not retained.
        """
        del frame_changed
        if self._shutdown:
            return
        if token is None or catalog is None:
            self.clear_presented_frame(reason="presented frame unavailable")
            return

        previous_token = self._frame_token
        token_changed = previous_token is not None and token != previous_token
        if token_changed:
            self._clear_selection(emit=True)

        self._frame_token = token
        self._presented_packet_identity = id(render_packet) if render_packet is not None else None
        self._catalog = catalog
        self._active = True
        self._install_renderer_callback()

        selection = self._selection
        if selection is None:
            return
        path_id = selection.canonical_path_id
        if not self._valid_path_id(path_id):
            self._clear_selection(emit=True)
            return
        if previous_token == token and self._snapshot is not None:
            self._ensure_existing_overlay()
            return
        self._present_selection(path_id, origin=self._details_origin(), emit_selection=False)

    def prepare_viewport_mapping(
        self,
        token: Any,
        catalog: Any,
        render_packet: Any,
    ) -> bool:
        """Attach the worker-built filtered-segment mapping to the active renderer."""
        if (
            not self.is_active
            or token != self._frame_token
            or catalog is not self._catalog
            or render_packet is None
            or id(render_packet) != self._presented_packet_identity
        ):
            return False
        renderer = self._current_renderer()
        setter = getattr(renderer, "set_mpc_pick_segment_mapping", None)
        if not self._renderer_inspection_available(renderer) or not callable(setter):
            return False
        try:
            mapping = catalog.rendered_segment_indices
            return bool(setter(id(render_packet), mapping))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def refresh_selected_details(self) -> None:
        """Refresh status classifications without rebuilding selected geometry."""
        selection = self._selection
        snapshot = self._snapshot
        if selection is None or snapshot is None or not self.is_active:
            return
        if selection.frame_token != self._frame_token:
            return
        interaction_values = snapshot.bounce_interaction_types
        if interaction_values is None:
            interaction_values = np.empty((0,), dtype=np.int32)
        try:
            details = self._build_details(
                selection.canonical_path_id,
                origin=self._details_origin(),
                snapshot=snapshot,
                interaction_values=np.asarray(interaction_values, dtype=np.int32),
            )
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return
        self._details = details
        self.detailsChanged.emit(details)

    def on_presented_frame(
        self,
        token: Any,
        catalog: Any,
        render_packet: Any = None,
        *,
        frame_changed: bool = False,
    ) -> None:
        """Compatibility spelling for accepted-frame lifecycle callers."""
        self.set_presented_frame(
            token,
            catalog,
            render_packet,
            frame_changed=frame_changed,
        )

    def select_path(
        self,
        token: Any,
        canonical_path_id: int,
        *,
        origin: str,
        packet_identity: Optional[int] = None,
    ) -> None:
        """Select one canonical path from a table, viewport, or future plot."""
        normalized_origin = str(origin)
        if (
            not self.is_active
            or token != self._frame_token
            or not self._valid_path_id(canonical_path_id)
            or (
                normalized_origin == "viewport"
                and (
                    self._presented_packet_identity is None
                    or packet_identity != self._presented_packet_identity
                )
            )
        ):
            return
        self._present_selection(
            int(canonical_path_id),
            origin=normalized_origin,
            emit_selection=True,
        )

    def clear_selection(self, *, reason: str = "selection cleared") -> None:
        """Clear only the current selection while retaining the active catalog."""
        del reason
        self._clear_selection(emit=True)

    def clear_presented_frame(self, *, reason: str) -> None:
        """Release all frame data, renderer hooks, overlays, and animation."""
        del reason
        self._active = False
        self._remove_renderer_callback()
        self._clear_selection(emit=True)
        self._frame_token = None
        self._presented_packet_identity = None
        self._catalog = None

    def shutdown(self) -> None:
        """Synchronously release every transient resource and become inert."""
        if self._shutdown:
            return
        self.clear_presented_frame(reason="selection service shutdown")
        if self._overlay_renderer is not None:
            self._clear_renderer_overlay()
        self._shutdown = True
        self.visualizer = None

    def _present_selection(
        self,
        path_id: int,
        *,
        origin: str,
        emit_selection: bool,
    ) -> None:
        """Build and publish one small path snapshot from the full catalog slice."""
        try:
            snapshot, details = self._build_snapshots(path_id, origin=origin)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return

        identity_changed = details.identity != self._selection
        self._stop_flow()
        self._clear_renderer_overlay()
        self._selection = details.identity
        self._details = details
        self._snapshot = snapshot

        overlay_accepted = self._set_renderer_overlay(snapshot)
        if overlay_accepted and snapshot.total_length > _MIN_FLOW_LENGTH:
            self._flow_started_at = self._clock()
            self._flow_timer.start()

        if emit_selection and identity_changed:
            self.selectionChanged.emit(self._selection)
        self.detailsChanged.emit(self._details)

    def _build_snapshots(
        self,
        path_id: int,
        *,
        origin: str,
    ) -> tuple[MpcPathInspectionSnapshot, MpcSelectedPathDetails]:
        catalog = self._catalog
        points = np.asarray(catalog.path_points(path_id))
        interaction_values = np.asarray(
            catalog.interaction_sequence(path_id),
            dtype=np.int32,
        ).reshape(-1)
        bounce_count = max(0, int(points.shape[0]) - 2)
        if interaction_values.size != bounce_count:
            raise ValueError("interaction sequence does not align with path bounces")

        bounce_colors = colorize_mpc_interaction_types(
            interaction_values,
            self._active_type_palette(),
        )
        snapshot = MpcPathInspectionSnapshot(
            frame_token=self._frame_token,
            canonical_path_id=path_id,
            points=points,
            bounce_interaction_types=interaction_values,
            bounce_colors=bounce_colors,
        )
        details = self._build_details(
            path_id,
            origin=origin,
            snapshot=snapshot,
            interaction_values=interaction_values,
        )
        return snapshot, details

    def _build_details(
        self,
        path_id: int,
        *,
        origin: str,
        snapshot: MpcPathInspectionSnapshot,
        interaction_values: np.ndarray,
    ) -> MpcSelectedPathDetails:
        """Build path metadata separately from immutable overlay geometry."""
        catalog = self._catalog
        identity = MpcSelectionIdentity(self._frame_token, path_id)

        material_values = catalog.material_sequence(path_id)
        material_ids: Optional[tuple[int, ...]]
        material_names: tuple[str, ...]
        if material_values is None:
            material_ids = None
            material_names = ()
        else:
            material_ids = tuple(int(value) for value in np.asarray(material_values).reshape(-1))
            material_names = tuple(
                self._material_label(catalog, material_id) for material_id in material_ids
            )

        is_filtered = self._scope_membership("is_filtered", path_id, default=True)
        is_rendered = self._scope_membership(
            "is_rendered",
            path_id,
            default=is_filtered,
        )
        return MpcSelectedPathDetails(
            identity=identity,
            origin=origin,
            tx_id=self._catalog_optional_int("tx", path_id),
            rx_id=self._catalog_optional_int("rx", path_id),
            path_loss_db=self._catalog_optional_float("path_loss_db", path_id),
            delay_ns=self._catalog_optional_float("delay_ns", path_id),
            interaction_count=self._catalog_interaction_count(
                path_id,
                interaction_values,
            ),
            interaction_types=tuple(int(value) for value in interaction_values),
            interaction_labels=tuple(
                mpc_interaction_label(int(value), explicit_unknown=True)
                for value in interaction_values
            ),
            material_ids=material_ids,
            material_names=material_names,
            geometric_length_m=snapshot.total_length,
            is_filtered=is_filtered,
            is_rendered=is_rendered,
            render_status=("rendered" if is_rendered else "outside current rendered set"),
            delay_is_estimated=self._canonical_optional_bool(
                "path_delay_is_estimated",
                path_id,
            ),
            path_loss_is_estimated=self._canonical_optional_bool(
                "path_loss_is_estimated",
                path_id,
            ),
        )

    @staticmethod
    def _material_label(catalog: Any, material_id: int) -> str:
        """Describe canonical material sentinels without calling them missing."""
        if material_id < 0:
            return "Unavailable"
        if material_id == 0:
            return "None"
        try:
            name = str(catalog.material_name(material_id) or "").strip()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            name = ""
        return name or f"Material {material_id}"

    def _active_type_palette(self) -> np.ndarray:
        """Resolve the same interaction palette used by the bulk MPC layer."""
        visualizer = self.visualizer
        mpc_core = None
        if visualizer is not None:
            mpc_core = getattr(visualizer, "mpc_core", None) or getattr(
                visualizer,
                "_mpc_core",
                None,
            )
        palette = getattr(mpc_core, "_type_palette", None)
        if self._valid_palette(palette):
            return np.asarray(palette, dtype=np.float32)

        try:
            from ..utils.colors import get_categorical_type_palette

            palette = get_categorical_type_palette(n_colors=9)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            palette = None
        if self._valid_palette(palette):
            return np.asarray(palette, dtype=np.float32)

        try:
            from shared.statistics.themes import theme_manager

            palette = build_mpc_type_palette(theme_manager.current.interaction_type)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            palette = np.tile(
                np.asarray(MPC_UNKNOWN_COLOR, dtype=np.float32),
                (9, 1),
            )
        return np.asarray(palette, dtype=np.float32)

    @staticmethod
    def _valid_palette(palette: Any) -> bool:
        if palette is None:
            return False
        values = np.asarray(palette)
        return values.ndim == 2 and values.shape[0] >= 6 and values.shape[1] >= 3

    def _catalog_optional_int(self, column_name: str, path_id: int) -> Optional[int]:
        value = self._catalog_scalar(column_name, path_id)
        if value is None:
            return None
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result >= 0 else None

    def _catalog_optional_float(
        self,
        column_name: str,
        path_id: int,
    ) -> Optional[float]:
        value = self._catalog_scalar(column_name, path_id)
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    def _catalog_scalar(self, column_name: str, path_id: int) -> Any:
        column = getattr(self._catalog, "column", None)
        if not callable(column):
            return None
        try:
            values = column(column_name)
            return values[path_id]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _catalog_interaction_count(
        self,
        path_id: int,
        interaction_values: np.ndarray,
    ) -> int:
        value = self._catalog_scalar("interactions", path_id)
        if value is None:
            return int(np.count_nonzero(interaction_values))
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return int(np.count_nonzero(interaction_values))

    def _canonical_optional_bool(self, field_name: str, path_id: int) -> Optional[bool]:
        canonical = getattr(self._catalog, "canonical_data", None)
        values = getattr(canonical, field_name, None)
        if values is None:
            return None
        try:
            return bool(np.asarray(values).reshape(-1)[path_id])
        except (IndexError, TypeError, ValueError):
            return None

    def _scope_membership(self, method_name: str, path_id: int, *, default: bool) -> bool:
        method = getattr(self._catalog, method_name, None)
        if not callable(method):
            return bool(default)
        try:
            return bool(method(path_id))
        except (IndexError, RuntimeError, TypeError, ValueError):
            return bool(default)

    def _valid_path_id(self, path_id: Any) -> bool:
        try:
            value = int(path_id)
            count = int(getattr(self._catalog, "path_count", 0))
        except (TypeError, ValueError):
            return False
        return value == path_id and 0 <= value < count

    def _install_renderer_callback(self) -> None:
        """Enable viewport selection only while this service is active."""
        self._remove_renderer_callback()
        if self._presented_packet_identity is None:
            return
        renderer = self._current_renderer()
        if not self._renderer_inspection_available(renderer):
            return
        setter = getattr(renderer, "set_mpc_path_selection_callback", None)
        if not callable(setter):
            return
        try:
            setter(self._on_viewport_path_selected)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        self._callback_renderer = renderer

    def _remove_renderer_callback(self) -> None:
        renderer = self._callback_renderer
        self._callback_renderer = None
        setter = getattr(renderer, "set_mpc_path_selection_callback", None)
        if callable(setter):
            try:
                setter(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    def _on_viewport_path_selected(
        self,
        path_id: int,
        packet_identity: Optional[int] = None,
    ) -> None:
        """Route viewport and table selections through the same method."""
        token = self._frame_token
        if token is not None:
            self.select_path(
                token,
                path_id,
                origin="viewport",
                packet_identity=packet_identity,
            )

    def _set_renderer_overlay(self, snapshot: MpcPathInspectionSnapshot) -> bool:
        renderer = self._current_renderer()
        if not self._renderer_inspection_available(renderer):
            return False
        if self._overlay_renderer is not None and self._overlay_renderer is not renderer:
            logger.warning(
                "MPC selection overlay replacement deferred until prior renderer cleanup succeeds"
            )
            return False
        setter = getattr(renderer, "set_mpc_path_inspection", None)
        if not callable(setter):
            return False
        try:
            accepted = bool(setter(snapshot))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        if accepted:
            self._overlay_renderer = renderer
        return accepted

    def _ensure_existing_overlay(self) -> None:
        """Keep a same-frame overlay and pulse intact across status refreshes."""
        snapshot = self._snapshot
        if snapshot is None:
            return
        renderer = self._current_renderer()
        if (
            renderer is self._overlay_renderer
            and self._renderer_inspection_available(renderer)
            and self._renderer_has_overlay(renderer, snapshot)
        ):
            return
        self._stop_flow()
        if not self._clear_renderer_overlay():
            return
        overlay_accepted = self._set_renderer_overlay(snapshot)
        if overlay_accepted and snapshot.total_length > _MIN_FLOW_LENGTH:
            self._flow_started_at = self._clock()
            self._flow_timer.start()

    def _clear_renderer_overlay(self) -> bool:
        """Clear the native overlay while retaining ownership on failure."""
        renderer = self._overlay_renderer
        if renderer is None:
            return True
        clear = getattr(renderer, "clear_mpc_path_inspection", None)
        if not callable(clear):
            logger.warning("MPC selection renderer cannot clear its owned overlay")
            return False
        try:
            cleared = bool(clear())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            cleared = False
            logger.warning("MPC selection overlay cleanup failed", exc_info=True)
        if cleared:
            self._overlay_renderer = None
            return True
        logger.warning("MPC selection overlay cleanup was rejected; ownership retained for retry")
        return False

    @staticmethod
    def _renderer_supports_inspection(renderer: Any) -> bool:
        try:
            return bool(renderer_capabilities(renderer).mpc_path_inspection)
        except TypeError:
            return False

    @classmethod
    def _renderer_inspection_available(cls, renderer: Any) -> bool:
        """Return whether a capable renderer currently owns a usable canvas."""
        if renderer is None or not cls._renderer_supports_inspection(renderer):
            return False
        probe = getattr(renderer, "mpc_path_inspection_available", None)
        if not callable(probe):
            return True
        try:
            return bool(probe())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def _renderer_has_overlay(
        renderer: Any,
        snapshot: MpcPathInspectionSnapshot,
    ) -> bool:
        """Probe renderer-local ownership when the backend exposes it."""
        probe = getattr(renderer, "has_mpc_path_inspection", None)
        if not callable(probe):
            return True
        try:
            return bool(probe(snapshot))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def _current_renderer(self) -> Any:
        visualizer = self.visualizer
        if visualizer is None:
            return None
        return getattr(visualizer, "renderer", None)

    def _on_flow_timeout(self) -> None:
        """Advance only the renderer's tiny pulse buffer at roughly 30 Hz."""
        snapshot = self._snapshot
        renderer = self._overlay_renderer
        if (
            not self.is_active
            or self._selection is None
            or snapshot is None
            or snapshot.total_length <= _MIN_FLOW_LENGTH
            or renderer is None
            or renderer is not self._current_renderer()
            or not self._renderer_inspection_available(renderer)
        ):
            self._stop_flow()
            return
        update = getattr(renderer, "update_mpc_path_flow", None)
        if not callable(update):
            self._stop_flow()
            return
        phase = ((self._clock() - self._flow_started_at) / MPC_FLOW_LOOP_SECONDS) % 1.0
        try:
            accepted = bool(update(phase))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            accepted = False
        if not accepted:
            self._stop_flow()

    def _stop_flow(self) -> None:
        self._flow_timer.stop()
        self._flow_started_at = 0.0

    def _clear_selection(self, *, emit: bool) -> None:
        previous_selection = self._selection
        previous_details = self._details
        self._stop_flow()
        self._clear_renderer_overlay()
        self._selection = None
        self._details = None
        self._snapshot = None
        if emit and previous_selection is not None:
            self.selectionChanged.emit(None)
        if emit and previous_details is not None:
            self.detailsChanged.emit(None)

    def _details_origin(self) -> str:
        if self._details is None:
            return "table"
        return self._details.origin


__all__ = [
    "MPC_FLOW_INTERVAL_MS",
    "MPC_FLOW_LOOP_SECONDS",
    "MpcSelectedPathDetails",
    "MpcSelectionIdentity",
    "MpcSelectionService",
]
