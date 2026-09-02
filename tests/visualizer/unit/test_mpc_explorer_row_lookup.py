"""Scalability tests for viewport-selection row synchronization."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

import visualizer.src.services.mpc_explorer_service as explorer_service_module
from visualizer.src.services.mpc_explorer_service import MpcExplorerSession


class _LookupModel:
    """Small Qt-model surface backed by a realistic int32 permutation."""

    def __init__(self, path_ids: np.ndarray, *, fetch_batch_size: int = 50_000) -> None:
        self.path_ids = path_ids
        self.permutation_revision = 7
        self.fetch_batch_size = int(fetch_batch_size)
        self.loaded_rows = min(self.fetch_batch_size, int(path_ids.size))
        self.cache: dict[int, Optional[int]] = {}
        self.fetch_calls = 0

    def path_lookup_snapshot(self) -> tuple[int, np.ndarray]:
        return self.permutation_revision, self.path_ids

    def cached_path_row(self, path_id: int) -> tuple[bool, Optional[int]]:
        normalized = int(path_id)
        return normalized in self.cache, self.cache.get(normalized)

    def cache_path_row(
        self,
        path_id: int,
        row: Optional[int],
        *,
        permutation_revision: int,
    ) -> bool:
        if int(permutation_revision) != self.permutation_revision:
            return False
        normalized_row = None if row is None else int(row)
        if normalized_row is not None:
            if int(self.path_ids[normalized_row]) != int(path_id):
                return False
        self.cache[int(path_id)] = normalized_row
        return True

    def row_for_path_id(self, path_id: int) -> Optional[int]:
        return self.cache.get(int(path_id))

    def rowCount(self) -> int:  # noqa: N802 - Qt-compatible test surface
        return self.loaded_rows

    def ensure_row_loaded(self, row: int) -> bool:
        target = int(row)
        if target < self.loaded_rows:
            return True
        self.fetch_calls += 1
        self.loaded_rows = min(
            self.loaded_rows + self.fetch_batch_size,
            int(self.path_ids.size),
        )
        return target < self.loaded_rows


class _LookupWindow:
    def __init__(self, model: _LookupModel) -> None:
        self.model = model
        self.selected: list[int] = []

    def select_path(self, path_id: int) -> bool:
        row = self.model.row_for_path_id(path_id)
        if row is None or row >= self.model.rowCount():
            return False
        self.selected.append(int(path_id))
        return True


def _wait_for(qapp, predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.001)
    qapp.processEvents()
    assert predicate()


def _active_session(model: _LookupModel) -> tuple[MpcExplorerSession, _LookupWindow]:
    session = MpcExplorerSession(object())
    window = _LookupWindow(model)
    session.window = window
    session._model = model
    session._active = True
    return session, window


def test_closed_session_does_not_create_a_row_lookup_worker() -> None:
    model = _LookupModel(np.arange(2_000_000, dtype=np.int32))
    session, _window = _active_session(model)
    session._active = False
    session._selected_path_id = 1_999_999

    session._request_path_row_reveal(1_999_999)

    assert session._row_lookup_executor is None
    assert session._active_row_lookup_future is None
    session.deleteLater()


def test_two_million_row_viewport_lookup_is_worker_side_and_zero_copy(
    qapp,
    monkeypatch,
) -> None:
    path_ids = np.arange(2_000_000, dtype=np.int32)[::-1].copy()
    model = _LookupModel(path_ids)
    session, window = _active_session(model)
    target = 17
    observed: dict[str, object] = {}
    gui_thread = threading.get_ident()
    execute = explorer_service_module._execute_row_lookup_request

    def observe(request):
        observed["same_array"] = request.path_ids is path_ids
        observed["thread"] = threading.get_ident()
        return execute(request)

    monkeypatch.setattr(
        explorer_service_module,
        "_execute_row_lookup_request",
        observe,
    )
    session._selected_path_id = target
    try:
        session._request_path_row_reveal(target)
        _wait_for(qapp, lambda: window.selected == [target])

        assert observed["same_array"] is True
        assert observed["thread"] != gui_thread
        assert model.cache[target] == 2_000_000 - target - 1
        assert model.fetch_calls > 1
    finally:
        session._active = False
        session._shutdown_row_lookup_executor()
        session.deleteLater()


def test_obsolete_lookup_is_discarded_and_latest_selection_wins(
    qapp,
    monkeypatch,
) -> None:
    path_ids = np.arange(100_000, dtype=np.int32)
    model = _LookupModel(path_ids, fetch_batch_size=25_000)
    session, window = _active_session(model)
    started = threading.Event()
    release = threading.Event()
    execute = explorer_service_module._execute_row_lookup_request
    call_count = 0

    def block_first(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            started.set()
            release.wait(timeout=2.0)
        return execute(request)

    monkeypatch.setattr(
        explorer_service_module,
        "_execute_row_lookup_request",
        block_first,
    )
    try:
        session._selected_path_id = 90_000
        session._request_path_row_reveal(90_000)
        assert started.wait(timeout=1.0)

        session._selected_path_id = 80_000
        session._request_path_row_reveal(80_000)
        release.set()
        _wait_for(qapp, lambda: window.selected == [80_000])

        assert 90_000 not in model.cache
        assert model.cache[80_000] == 80_000
        assert call_count == 2
    finally:
        release.set()
        session._active = False
        session._shutdown_row_lookup_executor()
        session.deleteLater()
