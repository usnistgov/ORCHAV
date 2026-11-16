"""Bounded raster-tile requests shared by Builder and visualization workers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from email.utils import parsedate_to_datetime

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "ORCHAV/0.2 MapViewer (+https://github.com/usnistgov/ORCHAV)"


def tile_expiry(headers, now):
    """Use provider freshness, falling back to OSM's minimum seven-day cache."""
    control = headers.get("Cache-Control", "").lower()
    if "no-cache" in control or "no-store" in control:
        return now
    match = re.search(r"max-age=(\d+)", control)
    if match:
        return now + max(0, int(match[1]) - int(headers.get("Age", "0")))
    if headers.get("Expires"):
        try:
            return parsedate_to_datetime(headers["Expires"]).timestamp()
        except (ValueError, TypeError, OverflowError):
            pass
    return now + 7 * 86400


def fetch_tile(session, url, cache, cancelled):
    """Fetch one visible tile with bounded I/O, conditional revalidation and disk reuse."""
    from PIL import Image

    key = hashlib.sha256(url.encode()).hexdigest()
    png, meta = cache / (key + ".png"), cache / (key + ".json")
    try:
        record = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    now = time.time()
    if png.is_file() and record.get("expires", 0) > now:
        try:
            with Image.open(png, formats=["PNG"]) as source:
                if source.size != (256, 256):
                    raise ValueError("Invalid cached tile size")
                return source.convert("RGB"), True
        except (OSError, ValueError):
            record = {}
    headers = {"User-Agent": USER_AGENT}
    if png.is_file():
        for key, name in (("etag", "If-None-Match"), ("modified", "If-Modified-Since")):
            if record.get(key):
                headers[name] = record[key]
    with session.get(url, headers=headers, timeout=(5, 8), stream=True) as response:
        if response.status_code == 304 and png.is_file():
            data = png.read_bytes()
        else:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(32768):
                if cancelled():
                    raise InterruptedError("Map loading cancelled")
                data.extend(chunk)
                if len(data) > 2 * 1024 * 1024:
                    raise ValueError("Map tile exceeds 2 MiB")
        with Image.open(io.BytesIO(data), formats=["PNG"]) as source:
            if source.size != (256, 256):
                raise ValueError("Map provider must supply 256-pixel raster tiles")
            image = source.convert("RGB")
        if "no-store" not in response.headers.get("Cache-Control", "").lower():
            cache.mkdir(parents=True, exist_ok=True)
            temporary = png.with_suffix(f".{os.getpid()}.tmp")
            image.save(temporary, format="PNG")
            temporary.replace(png)
            meta.write_text(
                json.dumps(
                    {
                        "expires": tile_expiry(response.headers, now),
                        "etag": response.headers.get("ETag", record.get("etag")),
                        "modified": response.headers.get("Last-Modified", record.get("modified")),
                    }
                ),
                encoding="utf-8",
            )
        return image, response.status_code == 304
