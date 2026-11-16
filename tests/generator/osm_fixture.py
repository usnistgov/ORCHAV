"""Authored primitive Overpass fixtures and a loopback-only HTTP test server.

No live map data is retained. Coordinates merely place rectangular primitives
in a metric-sized test region; they do not describe the named geographic site.
SPDX-License-Identifier: GPL-3.0-only
"""

import json
import math
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


@contextmanager
def fixture_server(payload, *, latency=0.0, mode="ok"):
    """Serve an explicitly supplied synthetic payload on loopback only."""
    records = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, status=200, headers=None):
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            records.append({"method": "GET", "path": self.path})
            if self.path == "/status":
                self.send(b"Connected\nCurrent time\nRate limit\nSlots\n2 slots available now.\n")
            elif mode == "trickle":
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            elif mode == "truncated":
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"short")
                self.close_connection = True
            elif mode == "stall":
                time.sleep(2)
                self.send(b"late")
            else:
                self.send(b"dataset")

        def do_POST(self):
            query = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())["data"][
                0
            ]
            records.append({"method": "POST", "query": query})
            time.sleep(latency)
            if mode == "stall":
                time.sleep(2)
            if mode == "html":
                self.send(b"<html>Error</html>")
                return
            if mode == "partial":
                self.send(
                    json.dumps({**payload, "remark": "runtime error: Query timed out"}).encode()
                )
                return
            if mode == "429":
                self.send(b"busy", 429, {"Retry-After": "120"})
                return
            if mode in {"429-broken", "429-stall"}:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", "100")
                self.end_headers()
                if mode == "429-stall":
                    self.wfile.flush()
                    time.sleep(2)
                else:
                    self.wfile.write(b"private invalid gzip body")
                self.close_connection = True
                return
            if mode == "capacity":
                self.send(
                    b"This Overpass API server is currently at capacity. All query workers "
                    b"are busy\nand the request queue is full.",
                    502,
                    {"Retry-After": "0"},
                )
                return
            if mode == "504" or (
                mode == "504-once" and sum(r["method"] == "POST" for r in records) == 1
            ):
                self.send(b"gateway timeout", 504, {"Retry-After": "0"})
                return
            result = payload
            if '["highway"]' in query:
                roads = [
                    e
                    for e in payload["elements"]
                    if e["type"] == "way" and e.get("tags", {}).get("highway")
                ]
                ids = {n for e in roads for n in e["nodes"]}
                result = {
                    **payload,
                    "elements": [
                        e for e in payload["elements"] if e["type"] == "node" and e["id"] in ids
                    ]
                    + roads,
                }
            self.send(json.dumps(result).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def synthetic_payload():
    """Construct a small grid with tagged boxes, a courtyard, roads and vegetation."""
    elements = []
    next_node = 1
    next_way = 1000

    def node(x, y, **tags):
        nonlocal next_node
        number = next_node
        next_node += 1
        item = {
            "type": "node",
            "id": number,
            "lat": 38.8887 + y / 111000,
            "lon": -77.0261 + x / (111000 * math.cos(math.radians(38.8887))),
        }
        if tags:
            item["tags"] = tags
        elements.append(item)
        return number

    def way(nodes, **tags):
        nonlocal next_way
        number = next_way
        next_way += 1
        elements.append({"type": "way", "id": number, "nodes": nodes, "tags": tags})
        return number

    def box(x0, y0, x1, y1, **tags):
        ids = [node(x0, y0), node(x1, y0), node(x1, y1), node(x0, y1)]
        return way(ids + ids[:1], **tags)

    for x in (-85, -55, 30, 65):
        for y in (-85, -55, 30, 65):
            box(
                x,
                y,
                x + 18,
                y + 18,
                **{
                    "building": "yes",
                    "height": "12",
                    "building:material": "brick",
                    "roof:shape": "flat",
                },
            )
    outer = box(5, 5, 26, 26)
    inner = box(10, 10, 20, 20)
    elements.append(
        {
            "type": "relation",
            "id": 9000,
            "tags": {
                "type": "multipolygon",
                "building": "yes",
                "height": "9",
                "building:material": "concrete",
            },
            "members": [
                {"type": "way", "ref": outer, "role": "outer"},
                {"type": "way", "ref": inner, "role": "inner"},
            ],
        }
    )
    box(
        -84,
        -84,
        -76,
        -76,
        **{
            "building:part": "yes",
            "height": "18",
            "min_height": "12",
            "building:material": "glass",
        },
    )
    box(-98, -98, 98, 98, landuse="residential")
    box(30, -40, 50, -20, natural="water")
    box(65, -40, 88, -20, natural="wood")
    way([node(-95, -45), node(-65, -45)], barrier="wall", height="2")
    node(-35, 45, natural="tree", height="6")
    grid = [[node(x, y) for y in (-115, 0, 115)] for x in (-115, 0, 115)]
    for indices in (*zip(*grid), *grid):
        way(list(indices), highway="residential", name="Fixture street")
    return {
        "version": 0.6,
        "generator": "ORCHAV deterministic test fixture (synthetic)",
        "elements": elements,
    }
