"""Single-worker, latest-request-wins preparation for retained-run summaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from ..services.sensing_service import SensingService
from .run_summary_data import load_run_summary


class RunSummarySession:
    """Keep scoring off Qt and reject stale completions after source/gate changes.

    Only the newest completed summary is retained. The worker owns its service;
    it does not share mutable selection caches with playback's reader.
    """

    def __init__(self, provider, *, previous=None):
        self.provider = provider
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensing-summary")
        self._cancel = Event()
        self._future = None
        self._pending_key = None
        self.key = None
        self.data = None
        self.error = None
        self.progress = ("Preparing summary", 0, 0)
        self.closed = False
        if (
            previous is not None
            and previous.provider is provider
            and not previous.pending
            and previous.data is not None
            and not previous.error
            and not previous.data.cancelled
        ):
            # Only completed immutable data transfers; futures/readers remain owned
            # by their original session. request() still checks inventory and gates.
            self.key = previous.key
            self.data = previous.data

    @property
    def pending(self):
        return self._future is not None

    def request(self, context, inventory, settings):
        """Reuse one exact inventory/cohort/gate result, otherwise retire old work."""
        key = (tuple(context), inventory, tuple(settings))
        if self.closed or key == self.key:
            return
        self._cancel.set()
        if self._future is not None:
            self._future.cancel()
        self._cancel = Event()
        self.key = self._pending_key = key
        self.data = None
        self.error = None
        self.progress = ("Preparing summary", 0, 0)
        self._future = self._executor.submit(self._load, key, self._cancel)

    def _load(self, key, cancelled):
        proxy = SimpleNamespace(frame_source=SimpleNamespace(provider=self.provider))
        service = SensingService(proxy)

        def progress(stage, completed, total):
            if not cancelled.is_set():
                self.progress = (stage, completed, total)

        service.configure_background_reader(cancelled, progress)
        context, _inventory, settings = key
        range_gate, velocity_gate, interval = settings[:3]
        confirmed_only = settings[3] if len(settings) > 3 else False
        return load_run_summary(
            service,
            context,
            cancelled,
            range_gate=range_gate,
            velocity_gate=velocity_gate,
            interval=interval,
            confirmed_only=confirmed_only,
        )

    def poll(self):
        """Publish on the caller's GUI thread; cancellation never publishes partial totals."""
        future = self._future
        if self.closed or future is None or not future.done():
            return False
        self._future = None
        if self._cancel.is_set() or self._pending_key != self.key:
            return False
        try:
            data = future.result()
            if data.cancelled:
                return False
            self.data = data
            self.error = data.error
        except Exception as exc:  # Reader boundary: surface errors without killing Qt polling.
            self.error = str(exc)
        return True

    def close(self):
        self.closed = True
        self._cancel.set()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
