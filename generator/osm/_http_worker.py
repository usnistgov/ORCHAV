"""Isolated HTTP download worker. Only the parent publishes cache files.

Kept stdlib-only so Windows process startup does not initialize RT/geometry.
The parent kills and joins this process on cancellation or deadline, including
when DNS resolution or a trickling response would evade socket timeouts.
"""

from __future__ import annotations

import gzip
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache


class CountedResponse:
    """Count wire bytes independently of decoded bytes for length validation."""

    def __init__(self, response):
        self.response = response
        self.count = 0

    def read(self, size=-1):
        data = self.response.read(size)
        self.count += len(data)
        return data


def pin_dns():
    # Overpass status and POST must reach the same round-robin backend.
    # Cache DNS only inside this short-lived worker; retain the URL hostname
    # for TLS verification and never alter the application's global resolver.
    socket.getaddrinfo = lru_cache(maxsize=32)(socket.getaddrinfo)


def download(job):
    data = job.get("data")
    request = urllib.request.Request(
        job["url"],
        data=urllib.parse.urlencode(data).encode() if data is not None else None,
        headers={"User-Agent": "ORCHAV/0.2 OSM scene acquisition", "Accept-Encoding": "gzip"},
    )
    try:
        try:
            response = urllib.request.urlopen(request, timeout=job["timeout"])
        except urllib.error.HTTPError as exc:
            response = exc
        with response, open(job["output"], "wb") as target:
            # Rate limiting is established by headers alone. Do not let a broken
            # or trickling error body hide the 429 and its required cooldown.
            if response.status == 429:
                print(  # noqa: T201 - worker protocol
                    json.dumps(
                        {
                            "status": 429,
                            "bytes": 0,
                            "transfer_bytes": 0,
                            "retry_after": response.headers.get("Retry-After"),
                        }
                    ),
                    flush=True,
                )
                return
            wire = CountedResponse(response)
            encoding = response.headers.get("Content-Encoding", "identity").lower().strip()
            if encoding not in {"identity", "gzip"}:
                raise OSError(f"Unsupported HTTP content encoding: {encoding}")
            decoded = gzip.GzipFile(fileobj=wire) if encoding == "gzip" else wire
            count = 0
            while chunk := decoded.read(65536):
                count += len(chunk)
                if count > job["max_bytes"]:
                    raise OSError("Download exceeds configured byte limit")
                target.write(chunk)
            expected = response.headers.get("Content-Length")
            if expected is not None and wire.count != int(expected):
                raise OSError("Truncated HTTP body")
            print(  # noqa: T201 - worker protocol
                json.dumps(
                    {
                        "status": response.status,
                        "bytes": count,
                        "transfer_bytes": wire.count,
                        "retry_after": response.headers.get("Retry-After"),
                    }
                ),
                flush=True,
            )  # noqa: T201 - worker protocol
    except Exception as exc:
        print(  # noqa: T201 - worker protocol
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True
        )  # noqa: T201 - worker protocol


if __name__ == "__main__":
    pin_dns()
    for line in sys.stdin:
        download(json.loads(line))
