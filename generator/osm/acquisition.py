"""Bounded OSM transport and feature batching, retaining OSMnx's parsers.

OSMnx 2.x's private request adapter is isolated here and covered by a real
parser/graph fixture test. Its recursive HTTP retry and status waits are not
used. No background downloader can publish a cache after this scope exits.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

logger = logging.getLogger(__name__)
_active = contextvars.ContextVar("osm_acquisition", default=None)
_lock = threading.RLock()  # OSMnx settings and the builder's indices are process globals.
_STATUS_TIMEOUT_S = 8.0  # Slot discovery must leave time for the actual query.
_RATE_LIMIT_WAIT_S = 30.0  # Public Overpass operator's minimum after HTTP 429.
_ERROR_BODY_PREFIX_BYTES = 4096  # Inspect a bounded prefix; never log provider bodies.


class AcquisitionError(OSError):
    """A failed acquisition, never a confirmed empty geographic response."""


class AcquisitionCancelled(BaseException):
    """Cancellation must escape optional-provider fallback handlers."""


def active_acquisition():
    return _active.get()


def validate_overpass(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise AcquisitionError("Invalid Overpass response: expected an elements array")
    if payload.get("remark") or payload.get("error"):
        raise AcquisitionError(
            f"Incomplete Overpass response: {payload.get('remark') or payload['error']}"
        )
    nodes = {
        e.get("id") for e in payload["elements"] if isinstance(e, dict) and e.get("type") == "node"
    }
    for element in payload["elements"]:
        if (
            not isinstance(element, dict)
            or element.get("type") not in {"node", "way", "relation"}
            or "id" not in element
        ):
            raise AcquisitionError("Malformed Overpass element")
        if element["type"] == "node":
            try:
                if not all(math.isfinite(float(element[k])) for k in ("lat", "lon")):
                    raise ValueError("nonfinite coordinates")
            except (KeyError, TypeError, ValueError) as exc:
                raise AcquisitionError("Malformed Overpass node coordinates") from exc
        if element["type"] == "way" and (
            not isinstance(element.get("nodes"), list) or not set(element["nodes"]).issubset(nodes)
        ):
            raise AcquisitionError("Incomplete Overpass way node references")
        if element["type"] == "relation" and not isinstance(element.get("members"), list):
            raise AcquisitionError("Malformed Overpass relation members")
    return payload


def _retry_after(value):
    if not value:
        return 5.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(
                0.0, (parsedate_to_datetime(value) - dt.datetime.now(dt.UTC)).total_seconds()
            )
        except (ValueError, TypeError):
            return 5.0


def _http_error(status, url, body):
    """Explain known HTTP failures without exposing arbitrary provider content."""
    message = f"Overpass HTTP {status} from {url}"
    if status == 429:
        return message + ": server rate limit reached. Try again later."
    if status in {502, 503, 504}:
        text = " ".join(body[:_ERROR_BODY_PREFIX_BYTES].decode("utf-8", errors="replace").split())
        if (
            "This Overpass API server is currently at capacity. "
            "All query workers are busy and the request queue is full."
        ) in text:
            return (
                message
                + ": server is at capacity (workers busy; request queue full). Try again later."
            )
        return message + ": server temporarily unavailable; it may be busy. Try again later."
    return message


class Acquisition:
    """One build's cumulative network budget, endpoint policy and progress."""

    def __init__(
        self, *, endpoint=None, timeout=180.0, request_timeout=45.0, cancel=None, progress=None
    ):
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("Acquisition timeouts must be finite and positive")
        self.endpoint = (endpoint or "https://overpass-api.de/api").rstrip("/")
        if self.endpoint.endswith("/interpreter"):
            self.endpoint = self.endpoint.removesuffix("/interpreter")
        self.remaining = float(timeout)
        self.request_timeout = float(request_timeout)
        self.cancel = cancel
        self.progress = progress
        self.layers = {}
        self.requests = []
        self.started = time.monotonic()
        self._worker = None
        self._reader = None
        self._messages = queue.Queue()

    def check(self):
        if self.cancel is not None and self.cancel():
            raise AcquisitionCancelled("OSM acquisition cancelled")
        if self.remaining <= 0:
            raise AcquisitionError("OSM acquisition network budget exhausted")

    def record(self, label, outcome, **details):
        record = {"outcome": outcome, **details}
        self.layers[label] = record
        logger.info("OSM stage %s: %s %s", label, outcome, details)
        if self.progress:
            self.progress({"stage": label, **record})

    def notify(self, message, **details):
        """Report transport activity without treating it as a completed layer."""
        if self.progress:
            self.progress({"stage": "network", "message": message, **details})

    def wait(self, seconds):
        self.check()
        if seconds >= self.remaining:
            remaining = self.remaining
            self.remaining = 0.0
            raise AcquisitionError(
                f"Provider wait ({seconds:.1f}s) exceeds remaining network budget ({remaining:.1f}s)"
            )
        logger.info(
            "Waiting %.1f s before the next provider request (%.1f s network budget remaining)",
            seconds,
            self.remaining,
        )
        start = time.monotonic()
        self.notify(f"Waiting {seconds:.0f} s; {self.remaining:.0f} s network budget left.")
        try:
            while time.monotonic() - start < seconds:
                self.check()
                time.sleep(max(0.0, min(0.1, seconds - (time.monotonic() - start))))
        finally:
            self.remaining -= time.monotonic() - start

    def download(self, url, path, *, data=None, max_bytes=256 * 1024 * 1024, timeout=None):
        """Download only into a caller-owned temporary path, with a hard deadline."""
        self.check()
        budget = min(self.remaining, self.request_timeout if timeout is None else timeout)
        start = time.monotonic()
        next_progress = start + 5.0
        self.notify(
            f"Contacting {url} · request limit {budget:.0f} s · network budget {self.remaining:.0f} s"
        )
        logger.info(
            "%s %s (request limit %.1f s; network budget remaining %.1f s)",
            "POST" if data is not None else "GET",
            url,
            budget,
            self.remaining,
        )
        job = {
            "url": url,
            "data": data,
            "output": str(path),
            "timeout": budget,
            "max_bytes": max_bytes,
        }
        try:
            if self._worker is None:
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                self._worker = subprocess.Popen(
                    [sys.executable, "-S", str(Path(__file__).with_name("_http_worker.py"))],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    creationflags=flags,
                )

                def read_protocol():
                    for line in self._worker.stdout:
                        self._messages.put(line)
                    self._messages.put(None)

                self._reader = threading.Thread(target=read_protocol, daemon=True)
                self._reader.start()
            self._worker.stdin.write(json.dumps(job) + "\n")
            self._worker.stdin.flush()
            while True:
                self.check()
                left = budget - (time.monotonic() - start)
                if left <= 0:
                    raise AcquisitionError(f"Download deadline exceeded after {budget:.1f}s: {url}")
                try:
                    message = self._messages.get(timeout=min(0.1, left))
                    break
                except queue.Empty:
                    now = time.monotonic()
                    if now >= next_progress:
                        logger.info(
                            "Still waiting for %s (%.1f s elapsed; %.1f s until request deadline)",
                            url,
                            now - start,
                            max(0.0, budget - (now - start)),
                        )
                        self.notify(
                            f"Waiting for {url} · {now - start:.0f} s elapsed · {max(0.0, budget - (now - start)):.0f} s until request timeout"
                        )
                        next_progress = now + 5.0
            if message is None:
                raise AcquisitionError("Download worker exited unexpectedly")
            result = json.loads(message)
            if result.get("error"):
                raise AcquisitionError(result["error"])
            result.update(url=url, elapsed_s=time.monotonic() - start)
            self.requests.append(result)
            logger.info("HTTP %s from %s after %.1f s", result["status"], url, result["elapsed_s"])
            return result
        except BaseException as exc:
            self.requests.append(
                {
                    "url": url,
                    "elapsed_s": time.monotonic() - start,
                    "outcome": "failed",
                    "error": str(exc),
                }
            )
            self.close(kill=True)
            raise
        finally:
            self.remaining -= time.monotonic() - start

    def close(self, *, kill=False):
        if self._worker is not None:
            if kill and self._worker.poll() is None:
                self._worker.kill()
            self._worker.stdin.close()
            try:
                self._worker.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._worker.kill()
                self._worker.wait()
            self._reader.join()
            self._worker.stdout.close()
            self._worker = None
            self._messages = queue.Queue()

    def _request(self, url, *, data=None, max_bytes=256 * 1024 * 1024, timeout=None):
        with tempfile.TemporaryDirectory(prefix="orchav-http-") as tmp:
            path = Path(tmp) / "body"
            response = self.download(url, path, data=data, max_bytes=max_bytes, timeout=timeout)
            return response, path.read_bytes()

    def _wait_after_http_error(self, response, message):
        delay = _retry_after(response.get("retry_after"))
        if response["status"] == 429:
            delay = max(_RATE_LIMIT_WAIT_S, delay)
        try:
            self.wait(delay)
        except AcquisitionError as exc:
            raise AcquisitionError(f"{message} {exc}") from exc

    def _slot(self):
        """Honor advertised waits without requiring the advisory status endpoint.

        Some providers omit /status, and it can fail while /interpreter works.
        The query itself still enforces rate limits and its Retry-After header.
        """
        try:
            response, body = self._request(
                self.endpoint + "/status",
                max_bytes=1024 * 1024,
                timeout=min(_STATUS_TIMEOUT_S, self.request_timeout),
            )
        except AcquisitionError as exc:
            self.check()
            self.notify(f"Server status unavailable ({exc}); trying the data request.")
            return
        if response["status"] != 200:
            message = _http_error(response["status"], self.endpoint + "/status", body)
            logger.warning("%s Advisory status unavailable; trying the data request.", message)
            self.notify(f"{message} Advisory status unavailable; trying the data request.")
            if response["status"] == 429 or response.get("retry_after"):
                self._wait_after_http_error(response, message)
            return
        text = body.decode("utf-8", errors="replace")
        if re.search(r"[1-9]\d* slots? available", text) or re.search(
            r"(?m)^Rate limit:\s*0\s*$", text
        ):
            return
        timestamps = re.findall(r"Slot available after: (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", text)
        if timestamps:
            now = dt.datetime.now(dt.UTC)
            delay = min((dt.datetime.fromisoformat(s) - now).total_seconds() for s in timestamps)
            self.wait(max(1.0, math.ceil(delay)))
        else:
            self.notify("Server did not advertise slot availability; trying the data request.")

    def overpass(self, data):
        import osmnx as ox

        for attempt in range(1, 3):
            try:
                logger.info("Overpass attempt %d/2 at %s", attempt, self.endpoint)
                self.notify(f"Overpass request · attempt {attempt}/2 · {self.endpoint}")
                if ox.settings.overpass_rate_limit:
                    self._slot()
                query = dict(data)
                seconds = max(1, int(min(self.remaining, self.request_timeout) - 2))
                query["data"] = re.sub(r"\[timeout:\d+\]", f"[timeout:{seconds}]", query["data"])
                response, body = self._request(self.endpoint + "/interpreter", data=query)
                status = response["status"]
                if status != 200:
                    message = _http_error(status, self.endpoint + "/interpreter", body)
                    logger.warning("%s", message)
                    self.notify(message)
                    # Even the last failed attempt must honor the cooldown before
                    # a caller can fall back to another layer on this acquisition.
                    if status == 429:
                        self._wait_after_http_error(response, message)
                    if status in {429, 502, 503, 504} and attempt < 2:
                        logger.warning(
                            "Overpass HTTP %s from %s; retrying once within the remaining budget",
                            status,
                            self.endpoint,
                        )
                        if status != 429:
                            self._wait_after_http_error(response, message)
                        continue
                    raise AcquisitionError(message)
                try:
                    return validate_overpass(json.loads(body))
                except (ValueError, UnicodeError) as exc:
                    raise AcquisitionError("Overpass returned malformed JSON") from exc
            except AcquisitionError as exc:
                if attempt == 2 or self.remaining <= 0:
                    raise
                logger.warning("Overpass attempt %d failed at %s: %s", attempt, self.endpoint, exc)
                self.notify(
                    f"Request failed: {exc}. Retrying once using the remaining network budget."
                )
                self.wait(5.0)
        raise AssertionError("unreachable")


@contextlib.contextmanager
def acquisition_scope(**kwargs):
    """Serialize temporary OSMnx adapter/settings changes and restore on all exits."""
    import osmnx as ox
    from osmnx import _overpass

    if not ox.__version__.startswith("2."):
        raise RuntimeError("ORCHAV OSM acquisition requires OSMnx 2.x")
    with _lock:
        context = Acquisition(**kwargs)
        saved = {
            name: getattr(ox.settings, name)
            for name in (
                "overpass_url",
                "overpass_rate_limit",
                "requests_timeout",
                "max_query_area_size",
            )
        }
        original = _overpass._overpass_request
        token = _active.set(context)
        try:
            ox.settings.overpass_url = context.endpoint
            _overpass._overpass_request = context.overpass
            yield context
        finally:
            context.close()
            _overpass._overpass_request = original
            for name, value in saved.items():
                setattr(ox.settings, name, value)
            _active.reset(token)


def download_file(url, destination, *, timeout=90.0, max_bytes=1024 * 1024 * 1024):
    """Atomic optional-dataset download using the same cancellable transport."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = active_acquisition() or Acquisition(timeout=timeout, request_timeout=timeout)
    fd, name = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    os.close(fd)
    temporary = Path(name)
    try:
        response = context.download(url, temporary, max_bytes=max_bytes, timeout=timeout)
        if response["status"] != 200:
            raise AcquisitionError(f"Dataset HTTP {response['status']}: {url}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        if active_acquisition() is None:
            context.close()


def merge_tags(groups):
    merged = {}
    for tags in groups:
        for key, value in tags.items():
            if value is True or merged.get(key) is True:
                merged[key] = True
            else:
                values = [value] if isinstance(value, str) else list(value)
                merged[key] = sorted(set(merged.get(key, [])) | set(values))
    return merged


def select_features(gdf, tags):
    """Same OR tag semantics as OSMnx, after its complete geometry parser."""
    if gdf is None:
        return None
    mask = None
    for key, value in tags.items():
        if key not in gdf:
            continue
        selected = (
            gdf[key].notna()
            if value is True
            else gdf[key].isin([value] if isinstance(value, str) else value)
        )
        mask = selected if mask is None else mask | selected
    if mask is None or not mask.any():
        return None
    return gdf.loc[mask].dropna(axis="columns", how="all").copy()
