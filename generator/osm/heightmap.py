# Adapted from NVlabs/sionna-large-radio-maps
# (sionna_lrm/scene/heightmap.py), which carries the following notice:
#
#   Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#
# See generator/osm/NOTICE.md for the full list of changes.

"""Terrain elevation from USGS heightmap tiles via AWS S3.

Public S3 GeoTIFFs use the bounded HTTP downloader and an on-disk tile cache.
No AWS credentials or SDK transfer workers are needed.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .acquisition import Acquisition, active_acquisition

logger = logging.getLogger(__name__)

TILE_PIXEL_SIZE = 512
TILE_DOWNLOAD_WORKERS = 8
MAX_PARENT_TILE_ATTEMPTS = 10
INVALID_ELEVATION_MIN_M = -1e6
INVALID_ELEVATION_MAX_M = 1e4


class HeightMap:
    """Terrain heightmap built from USGS elevation tiles on AWS S3.

    Downloads slippy-map elevation GeoTIFF tiles and provides bilinear
    height queries in either UTM or WGS84 coordinates.

    Parameters
    ----------
    utm_zone : CRS or str
        Target UTM CRS for coordinate transforms.
    bbox : tuple
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84.
    zoom : int
        Slippy-map zoom level (higher = more detail, default 14).
    """

    def __init__(
        self,
        utm_zone,
        bbox: tuple[float, float, float, float],
        zoom: int = 14,
    ):
        context = active_acquisition()
        self._cancel_event = threading.Event()
        self._parent_cancel = context.cancel if context else None
        self._deadline = time.monotonic() + (context.remaining if context else 180.0)
        self._cache = (
            Path(os.environ.get("ORCHAV_CACHE_DIR", "~/.orchav/cache")).expanduser() / "elevation"
        )

        min_lon, min_lat, max_lon, max_lat = bbox
        z = zoom

        x0 = int(self._lon2tile(min_lon, z))
        x1 = int(self._lon2tile(max_lon, z))
        y0 = int(self._lat2tile(max_lat, z))
        y1 = int(self._lat2tile(min_lat, z))

        self.to_utm = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)

        n_rows = y1 - y0 + 1
        n_cols = x1 - x0 + 1
        tex = np.zeros((n_rows * TILE_PIXEL_SIZE, n_cols * TILE_PIXEL_SIZE, 1))
        logger.info(
            "Loading heightmap: tiles (%d,%d)-(%d,%d) at zoom %d",
            x0,
            y0,
            x1,
            y1,
            z,
        )

        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=TILE_DOWNLOAD_WORKERS) as executor:
                futures = [
                    executor.submit(self._get_tile, z, x0 + col, y0 + row)
                    for row in range(n_rows)
                    for col in range(n_cols)
                ]
                try:
                    for future in futures:
                        x, y, tile = future.result(
                            timeout=max(0.001, self._deadline - time.monotonic())
                        )
                        col, row = x - x0, y - y0
                        tex[
                            row * TILE_PIXEL_SIZE : (row + 1) * TILE_PIXEL_SIZE,
                            col * TILE_PIXEL_SIZE : (col + 1) * TILE_PIXEL_SIZE,
                            0,
                        ] = tile
                except BaseException:
                    self._cancel_event.set()
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            if context:
                context.remaining -= time.monotonic() - started

        # Store as plain numpy array for bilinear interpolation.
        # This avoids Mitsuba variant issues (scalar_rgb vs cuda_ad_rgb).
        self._tex_np = tex[:, :, 0].astype(np.float32)

        self.map_min_lon = self._tile2lon(x0, z)
        self.map_min_lat = self._tile2lat(y1 + 1, z)
        self.map_max_lon = self._tile2lon(x1 + 1, z)
        self.map_max_lat = self._tile2lat(y0, z)

    def height_from_utm(self, x, y):
        """Query height at UTM coordinates (scalar or array)."""
        x_wgs, y_wgs = self.to_wgs84.transform(x, y)
        return self.height_from_wgs84(x_wgs, y_wgs)

    def height_from_wgs84(self, lon, lat):
        """Query height at WGS84 lon/lat coordinates (scalar or array)."""
        u = (np.asarray(lon) - self.map_min_lon) / (self.map_max_lon - self.map_min_lon)
        v = (np.asarray(lat) - self.map_min_lat) / (self.map_max_lat - self.map_min_lat)
        return self._sample_bilinear(u, v)

    def heights_from_utm_batch(self, x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
        """Query heights at arrays of UTM coordinates.

        Returns a numpy float32 array of the same length as the inputs.
        """
        lon_array, lat_array = self.to_wgs84.transform(x_array, y_array)
        u = (lon_array - self.map_min_lon) / (self.map_max_lon - self.map_min_lon)
        v = (lat_array - self.map_min_lat) / (self.map_max_lat - self.map_min_lat)
        return self._sample_bilinear(u, v)

    def _sample_bilinear(self, u, v):
        """Bilinear interpolation on the heightmap texture using numpy.

        Works with both scalar and array inputs, independent of Mitsuba
        variant (scalar_rgb, cuda_ad_rgb, etc.).
        """
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
        v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
        h, w = self._tex_np.shape
        fx = u * (w - 1)
        fy = (1.0 - v) * (h - 1)  # v=0 is bottom (min_lat), tex row 0 is top

        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        tx = fx - x0
        ty = fy - y0

        tex = self._tex_np
        h00 = tex[y0, x0]
        h10 = tex[y0, x1]
        h01 = tex[y1, x0]
        h11 = tex[y1, x1]
        result = (
            (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
        )
        return np.asarray(result, dtype=np.float32)

    # -- tile download helpers ------------------------------------------------

    @staticmethod
    def _resize_to_512(im: np.ndarray) -> np.ndarray:
        """Resize a height tile to 512x512 using bilinear interpolation."""
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(im)
        return np.array(pil_img.resize((512, 512), PILImage.BILINEAR), dtype=np.float32)

    def _download_tile(self, z, x, y):
        from PIL import Image

        cache_path = self._cache / str(z) / str(x) / f"{y}.tif"
        if cache_path.exists():
            try:
                with Image.open(cache_path) as image:
                    im = np.array(image, dtype=np.float32)
                if im.shape != (TILE_PIXEL_SIZE, TILE_PIXEL_SIZE):
                    im = self._resize_to_512(im)
                return im
            except OSError:
                cache_path.unlink(missing_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Each worker inherits the provider deadline and cancellation signal.
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Terrain acquisition budget exhausted")
        request = Acquisition(
            timeout=remaining,
            request_timeout=min(45.0, remaining),
            cancel=lambda: (
                self._cancel_event.is_set() or (self._parent_cancel and self._parent_cancel())
            ),
        )
        url = f"https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{z}/{x}/{y}.tif"
        with tempfile.TemporaryDirectory(prefix=".tile-", dir=cache_path.parent) as tmp:
            downloaded = Path(tmp) / "tile.tif"
            try:
                response = request.download(url, downloaded, max_bytes=16 * 1024 * 1024)
            finally:
                request.close()
            if response["status"] != 200:
                raise OSError(f"Terrain tile HTTP {response['status']}: {url}")
            with Image.open(downloaded) as image:
                im = np.array(image, dtype=np.float32)
            os.replace(downloaded, cache_path)
        if im.shape != (TILE_PIXEL_SIZE, TILE_PIXEL_SIZE):
            im = self._resize_to_512(im)
        return im

    def _get_tile(self, z, x, y):
        im = self._download_tile(z, x, y)
        attempts = 0
        while np.any(
            (im < INVALID_ELEVATION_MIN_M) | np.isnan(im) | (im > INVALID_ELEVATION_MAX_M)
        ):
            attempts += 1
            logger.warning(
                "Tile %d,%d at zoom %d has invalid pixels, filling from zoom %d",
                x,
                y,
                z,
                z - attempts,
            )
            if attempts >= MAX_PARENT_TILE_ATTEMPTS:
                logger.error("Cannot fill invalid pixels for tile %d,%d", x, y)
                raise OSError(f"Cannot fill invalid terrain pixels for tile {z}/{x}/{y}")
            scale = 2**attempts
            parent = self._download_tile(z - attempts, x // scale, y // scale)
            im = parent[
                (y % scale) * TILE_PIXEL_SIZE // scale : (y % scale + 1) * TILE_PIXEL_SIZE // scale,
                (x % scale) * TILE_PIXEL_SIZE // scale : (x % scale + 1) * TILE_PIXEL_SIZE // scale,
            ]
        if im.shape != (TILE_PIXEL_SIZE, TILE_PIXEL_SIZE):
            im = self._resize_to_512(im)
        return x, y, im

    # -- slippy-map tile math -------------------------------------------------

    @staticmethod
    def _lat2tile(lat, z):
        lat_rad = np.deg2rad(lat)
        n = 2**z
        return np.floor(n * 0.5 * (1 - np.arcsinh(np.tan(lat_rad)) / np.pi))

    @staticmethod
    def _lon2tile(lon, z):
        n = 2**z
        return np.floor(n * 0.5 * (1 + np.deg2rad(lon) / np.pi))

    @staticmethod
    def _tile2lon(x, z):
        return x / (2**z) * 360 - 180

    @staticmethod
    def _tile2lat(y, z):
        n = 2**z
        return np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
