"""pygfx transform-gizmo support for interactive preview editing.

This mixin attaches pygfx's native ``TransformGizmo`` to renderer objects with
stable TX/RX/target names. It reports selected/changed/committed phases back to
controller code so UI preview edits can stay separate from renderer object
mutation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NODE_MARKER_RE = re.compile(r"^node:(tx|rx)_(\d+)::marker$")
_TARGET_MESH_RE = re.compile(r"^target:[^:]+::mesh$")


def reset_transform_gizmo_interaction(gizmo: Any) -> None:
    """Cancel native drag bookkeeping before a persistent gizmo is detached."""

    pointer_id = getattr(gizmo, "_orchav_pointer_id", None)
    release_pointer = getattr(gizmo, "release_pointer_capture", None)
    if pointer_id is not None and callable(release_pointer):
        try:
            release_pointer(pointer_id)
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
    try:
        gizmo._orchav_pointer_id = None
        gizmo._ref = None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    highlight = getattr(gizmo, "_highlight", None)
    if callable(highlight):
        try:
            highlight()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


class PygfxTransformGizmoMixin:
    """Attach a pygfx ``TransformGizmo`` to picked editable entities."""

    def clear_transform_gizmo(self) -> None:
        """Detach and hide the persistent transform gizmo, if present."""
        self._transform_session_callback = None
        self._transform_gizmo_target_name = None
        self._transform_gizmo_target_kind = None
        self._transform_gizmo_target_index = None
        self._transform_gizmo_last_position = None
        self._transform_gizmo_last_transform = None
        self._transform_gizmo_control_object = None
        self._remove_transform_gizmo_proxy()

        gizmo = getattr(self, "_transform_gizmo", None)
        if gizmo is None:
            return
        reset_transform_gizmo_interaction(gizmo)
        try:
            gizmo.set_object(None)
        except Exception:
            pass
        try:
            gizmo.visible = False
        except Exception:
            pass

    def _dispose_transform_gizmo(self) -> None:
        """Destroy the persistent gizmo during renderer teardown only."""

        self.clear_transform_gizmo()
        gizmo = getattr(self, "_transform_gizmo", None)
        if gizmo is None:
            return
        if self._scene is not None:
            try:
                self._scene.remove(gizmo)
            except (ValueError, RuntimeError):
                pass
        self._transform_gizmo = None

    def get_active_transform_target(self) -> dict[str, Any] | None:
        """Return the currently selected editable entity, if any."""
        name = getattr(self, "_transform_gizmo_target_name", None)
        kind = getattr(self, "_transform_gizmo_target_kind", None)
        index = getattr(self, "_transform_gizmo_target_index", None)
        if name is None or kind is None or index is None:
            return None
        return {"object_id": name, "kind": kind, "index": int(index)}

    def _select_transform_gizmo_target(
        self,
        name: str,
        kind: str,
        index: int,
        *,
        semantic_transform: Any = None,
    ) -> bool:
        """Attach the gizmo to one named node marker and emit selection."""
        obj = self._objects.get(name)
        if obj is None or self._scene is None:
            return False
        gizmo = self._ensure_transform_gizmo()
        if gizmo is None:
            return False
        control_obj = obj
        if kind == "target":
            control_obj = self._target_transform_proxy(name, int(index), obj)
            if control_obj is None:
                return False
        elif kind == "authoring":
            control_obj = self._authoring_transform_proxy(name, semantic_transform)
            if control_obj is None:
                return False

        self._transform_gizmo_target_name = name
        self._transform_gizmo_target_kind = kind
        self._transform_gizmo_target_index = int(index)
        self._transform_gizmo_control_object = control_obj
        self._transform_gizmo_last_position = self._object_world_position(control_obj)
        self._transform_gizmo_last_transform = self._object_transform_matrix(control_obj)
        try:
            gizmo.set_object(control_obj)
            gizmo.visible = True
        except Exception as exc:
            logger.warning("Could not attach transform gizmo to %s: %s", name, exc)
            self._transform_gizmo_target_name = None
            self._transform_gizmo_target_kind = None
            self._transform_gizmo_target_index = None
            self._transform_gizmo_control_object = None
            self._transform_gizmo_last_position = None
            self._transform_gizmo_last_transform = None
            self._remove_transform_gizmo_proxy()
            return False

        self._emit_transform_gizmo_event("selected", force=True)
        self.request_redraw()
        return True

    def _ensure_transform_gizmo(self, *, register_before_render: bool = False) -> Any | None:
        """Create the pygfx TransformGizmo and wire its object callbacks once.

        The renderer-lifetime interaction router owns the only renderer-level
        handler set and calls ``update_gizmo`` itself. ``register_before_render``
        remains an assertion-friendly parameter for the runtime boundary.
        """
        gizmo = getattr(self, "_transform_gizmo", None)
        if gizmo is not None:
            return gizmo
        if self._scene is None or self._renderer is None or self._camera is None:
            return None
        base_cls = getattr(self._gfx, "TransformGizmo", None)
        if base_cls is None:
            logger.warning("pygfx TransformGizmo is unavailable in this environment")
            return None

        owner = self

        class _OrchavTransformGizmo(base_cls):  # type: ignore[misc, valid-type]
            """Forward native gizmo pointer events into ORCHAV transform phases."""

            def process_event(self, event: Any) -> None:
                """Emit changed/committed callbacks after native gizmo handling."""
                event_type = str(getattr(event, "type", ""))
                super().process_event(event)
                if event_type == "pointer_down" and getattr(self, "_ref", None):
                    self._orchav_pointer_id = getattr(event, "pointer_id", None)
                elif event_type == "pointer_move":
                    owner._emit_transform_gizmo_event("changed")
                elif event_type == "pointer_up":
                    self._orchav_pointer_id = None
                    owner._emit_transform_gizmo_event("committed", force=True)

        try:
            gizmo = _OrchavTransformGizmo(screen_size=120)
            gizmo.toggle_mode("world")
            if register_before_render:
                raise ValueError(
                    "renderer-level gizmo handlers are owned by the interaction router"
                )
            viewport = self._gfx.Viewport(self._renderer)
            gizmo._viewport = viewport
            gizmo._camera = self._camera
            gizmo.add_event_handler(
                gizmo.process_event,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "wheel",
            )
            gizmo.visible = False
            self._scene.add(gizmo)
        except Exception as exc:
            logger.warning("Could not create pygfx transform gizmo: %s", exc)
            return None

        self._transform_gizmo = gizmo
        return gizmo

    def _emit_transform_gizmo_event(self, phase: str, *, force: bool = False) -> None:
        """Emit a controller-facing transform event when the target moved."""
        callback = getattr(self, "_transform_session_callback", None)
        name = getattr(self, "_transform_gizmo_target_name", None)
        kind = getattr(self, "_transform_gizmo_target_kind", None)
        index = getattr(self, "_transform_gizmo_target_index", None)
        if callback is None or name is None or kind is None or index is None:
            return

        target_obj = self._objects.get(name)
        if target_obj is None:
            return
        obj = (
            getattr(self, "_transform_gizmo_control_object", None)
            if kind in {"target", "authoring"}
            else target_obj
        )
        if obj is None:
            obj = target_obj
        position = self._object_world_position(obj)
        if position is None:
            return
        transform = self._object_transform_matrix(obj)
        last = getattr(self, "_transform_gizmo_last_position", None)
        last_transform = getattr(self, "_transform_gizmo_last_transform", None)
        if not force:
            position_unchanged = last is not None and np.allclose(last, position, atol=1e-6)
            if transform is not None and last_transform is not None:
                transform_unchanged = np.allclose(last_transform, transform, atol=1e-6)
                if position_unchanged and transform_unchanged:
                    return
            elif position_unchanged:
                return

        self._transform_gizmo_last_position = position
        self._transform_gizmo_last_transform = transform
        if kind not in {"target", "authoring"}:
            self._sync_tracked_transform_from_object(name, obj, position, transform)
        event = {
            "phase": phase,
            "object_id": name,
            "kind": kind,
            "index": int(index),
            "position": tuple(float(x) for x in position),
        }
        if transform is not None:
            event["transform"] = transform.astype(float).tolist()
        callback(event)

    def _parse_transform_target_name(self, name: Any) -> tuple[str, int] | None:
        """Parse stable editable-object names into kind and index."""
        if not isinstance(name, str):
            return None
        match = _NODE_MARKER_RE.match(name)
        if match is not None:
            try:
                return match.group(1), int(match.group(2))
            except (TypeError, ValueError):
                return None

        if _TARGET_MESH_RE.match(name) is not None:
            index = self._target_index_for_geometry_name(name)
            if index is not None:
                return "target", int(index)
        return None

    def _target_index_for_geometry_name(self, name: str) -> int | None:
        """Resolve a stable target mesh geometry name to the current target index."""
        try:
            from ...services.object_identity import make_target_entry_geometry_name
        except ImportError:
            return None
        entries = getattr(getattr(self, "visualizer", None), "target_entries", []) or []
        for index, entry in enumerate(entries):
            try:
                if make_target_entry_geometry_name(entry, "mesh") == name:
                    return int(index)
            except (AttributeError, TypeError, ValueError):
                continue
        return None

    def _target_semantic_position(self, index: int) -> np.ndarray | None:
        """Return the target AABB-center position used by Sionna and the UI."""
        entries = getattr(getattr(self, "visualizer", None), "target_entries", []) or []
        if 0 <= int(index) < len(entries):
            entry = entries[int(index)]
            for key in ("_target_position", "position"):
                value = entry.get(key)
                if value is None:
                    continue
                try:
                    arr = np.asarray(value, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue
                if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                    return arr[:3].copy()

        viz = getattr(self, "visualizer", None)
        for source_name in ("current_target_positions",):
            values = getattr(viz, source_name, None)
            try:
                arr = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 2 and 0 <= int(index) < len(arr) and arr.shape[1] >= 3:
                row = arr[int(index), :3]
                if np.all(np.isfinite(row)):
                    return row.copy()

        return None

    def _target_transform_proxy(self, name: str, index: int, obj: Any) -> Any | None:
        """Create a semantic-center proxy for pygfx target gizmo interaction."""
        world_object_cls = getattr(self._gfx, "WorldObject", None)
        if world_object_cls is None:
            world_object_cls = getattr(self._gfx, "Group", None)
        if world_object_cls is None:
            logger.warning("pygfx target editing requires WorldObject or Group support")
            return None

        self._remove_transform_gizmo_proxy()
        try:
            proxy = world_object_cls()
        except Exception as exc:
            logger.warning("Could not create target transform proxy for %s: %s", name, exc)
            return None
        try:
            proxy.visible = False
        except Exception:
            pass

        transform = self._object_transform_matrix(obj)
        proxy_transform = np.eye(4, dtype=np.float32)
        if transform is not None:
            proxy_transform[:3, :3] = transform[:3, :3]
        position = self._target_semantic_position(index)
        if position is None:
            position = self._object_world_position(obj)
        if position is None:
            return None
        proxy_transform[:3, 3] = np.asarray(position, dtype=np.float32)[:3]
        if not self._set_proxy_transform(proxy, proxy_transform):
            return None

        if self._scene is not None:
            try:
                self._scene.add(proxy)
            except Exception:
                pass
        self._transform_gizmo_target_proxy = proxy
        return proxy

    def _authoring_transform_proxy(self, name: str, transform: Any) -> Any | None:
        """Create a semantic entity-pose proxy for Scenario Builder targets."""

        world_object_cls = getattr(self._gfx, "WorldObject", None)
        if world_object_cls is None:
            world_object_cls = getattr(self._gfx, "Group", None)
        if world_object_cls is None:
            logger.warning("pygfx authoring target editing requires WorldObject or Group")
            return None
        try:
            matrix = np.asarray(transform, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None

        self._remove_transform_gizmo_proxy()
        try:
            proxy = world_object_cls()
            proxy.visible = False
        except Exception as exc:
            logger.warning("Could not create authoring transform proxy for %s: %s", name, exc)
            return None
        if not self._set_proxy_transform(proxy, matrix):
            return None
        if self._scene is not None:
            try:
                self._scene.add(proxy)
            except Exception:
                pass
        self._transform_gizmo_target_proxy = proxy
        return proxy

    def _remove_transform_gizmo_proxy(self) -> None:
        """Remove the invisible target-edit proxy from the scene, if present."""
        proxy = getattr(self, "_transform_gizmo_target_proxy", None)
        self._transform_gizmo_target_proxy = None
        if proxy is None or self._scene is None:
            return
        try:
            self._scene.remove(proxy)
        except (ValueError, RuntimeError):
            pass

    def sync_active_transform_target_pose(self, object_id: str, transform: Any) -> bool:
        """Sync the active target-edit proxy after ORCHAV applies a pose."""
        if object_id != getattr(self, "_transform_gizmo_target_name", None):
            return False
        if getattr(self, "_transform_gizmo_target_kind", None) not in {
            "target",
            "authoring",
        }:
            return False
        proxy = getattr(self, "_transform_gizmo_target_proxy", None)
        if proxy is None:
            return False
        try:
            matrix = np.asarray(transform, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return False
        if not self._set_proxy_transform(proxy, matrix):
            return False
        self._transform_gizmo_last_position = self._object_world_position(proxy)
        self._transform_gizmo_last_transform = self._object_transform_matrix(proxy)
        self.request_redraw()
        return True

    @staticmethod
    def _set_proxy_transform(proxy: Any, transform: np.ndarray) -> bool:
        """Apply a 4x4 transform to a pygfx proxy object."""
        local = getattr(proxy, "local", None)
        if local is None:
            return False
        try:
            if hasattr(local, "matrix"):
                local.matrix = np.asarray(transform, dtype=np.float32)
                return True
        except Exception:
            pass
        try:
            if hasattr(local, "position"):
                local.position = tuple(float(value) for value in transform[:3, 3])
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _object_world_position(obj: Any) -> np.ndarray | None:
        """Read a finite world/local position from a pygfx object."""
        for attr_name in ("world", "local"):
            transform = getattr(obj, attr_name, None)
            position = getattr(transform, "position", None)
            if position is None:
                continue
            try:
                arr = np.asarray(position, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                return arr[:3].copy()
        return None

    @staticmethod
    def _object_transform_matrix(obj: Any) -> np.ndarray | None:
        """Read a finite local/world transform matrix from a pygfx object."""
        for attr_name in ("local", "world"):
            transform = getattr(obj, attr_name, None)
            matrix = getattr(transform, "matrix", None)
            if matrix is None:
                continue
            try:
                arr = np.asarray(matrix, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
                return arr.copy()
        return None

    def _sync_tracked_transform_from_object(
        self,
        name: str,
        obj: Any,
        position: np.ndarray,
        transform: np.ndarray | None = None,
    ) -> None:
        """Refresh renderer transform caches from the gizmo-mutated object."""
        self._positions[name] = tuple(float(x) for x in position[:3])
        if transform is not None:
            mat = np.asarray(transform, dtype=np.float32)
        else:
            local = getattr(obj, "local", None)
            matrix = getattr(local, "matrix", None)
            try:
                mat = np.asarray(matrix, dtype=np.float32)
            except (TypeError, ValueError):
                mat = np.eye(4, dtype=np.float32)
                mat[:3, 3] = np.asarray(position[:3], dtype=np.float32)
        if mat.shape != (4, 4):
            mat = np.eye(4, dtype=np.float32)
            mat[:3, 3] = np.asarray(position[:3], dtype=np.float32)
        self._transforms[name] = mat
