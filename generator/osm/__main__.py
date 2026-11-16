"""CLI entry point: ``python -m generator.osm``."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from .metadata import normalize_osm_config
from .scene_builder import build_scene, build_scene_from_center


def parse_maps_url(url: str) -> tuple[float, float] | None:
    """Extract (lat, lon) from a Google Maps URL.

    Supported formats::

        https://www.google.com/maps/@40.748,-73.985,17z
        https://maps.google.com/?ll=40.748,-73.985
        https://www.google.com/maps/place/.../@40.748,-73.985,17z/...
        https://maps.app.goo.gl/... (not supported — requires redirect)

    Returns (lat, lon) or None if parsing fails.
    """
    # Format: /@lat,lon,zoom
    m = re.search(r"/@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Format: ?ll=lat,lon
    m = re.search(r"[?&]ll=(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Format: ?q=lat,lon
    m = re.search(r"[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if m:
        return float(m.group(1)), float(m.group(2))

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Sionna RT scene from OpenStreetMap data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for scene.xml and mesh/ files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--terrain",
        action="store_true",
        help="Use public elevation GeoTIFFs for terrain.",
    )
    parser.add_argument(
        "--terrain-zoom",
        type=int,
        default=14,
        help="Heightmap tile zoom level (default: 14).",
    )
    parser.add_argument(
        "--ground-pad",
        type=float,
        default=200.0,
        help="Ground mesh padding in meters (default: 200).",
    )
    parser.add_argument(
        "--ground-resolution",
        type=float,
        default=50.0,
        help="Ground mesh grid resolution in meters (default: 50).",
    )
    parser.add_argument(
        "--osm-server",
        help="Custom Overpass API endpoint URL.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("LAT", "LON"),
        help="Center point in WGS84 degrees.",
    )
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Bounding box in WGS84 degrees.",
    )
    group.add_argument(
        "--url",
        type=str,
        help="Google Maps URL (extracts lat/lon automatically).",
    )

    parser.add_argument(
        "--size",
        type=float,
        default=500.0,
        help="Scene side length in meters (used with --center, default: 500).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Scene east-west width in meters (default: same as --size).",
    )

    parser.add_argument(
        "--acquisition-timeout",
        type=float,
        default=180.0,
        help="Cumulative acquisition budget in seconds (default: 180).",
    )
    parser.add_argument(
        "--overpass-timeout",
        type=int,
        default=45,
        help="Maximum individual request duration in seconds (default: 45).",
    )
    parser.add_argument("--cache-dir", type=str, help="OSM query cache directory.")
    parser.add_argument("--refresh-cache", action="store_true", help="Revalidate OSM query caches.")
    parser.add_argument(
        "--no-height-enrichment",
        action="store_true",
        help="Use OSM height tags and deterministic defaults only.",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    common_kwargs = dict(
        osm_acquisition_timeout=args.acquisition_timeout,
        osm_overpass_timeout=args.overpass_timeout,
        osm_cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        osm_refresh_cache=args.refresh_cache,
        height_enrichment=not args.no_height_enrichment,
        terrain=args.terrain,
        terrain_zoom=args.terrain_zoom,
        ground_pad_m=args.ground_pad,
        ground_resolution_m=args.ground_resolution,
        osm_server=args.osm_server,
    )

    # Parse Google Maps URL into center coordinates.
    if args.url:
        parsed = parse_maps_url(args.url)
        if parsed is None:
            parser.error(f"Could not extract lat/lon from URL: {args.url}")
        args.center = list(parsed)
        logger.info("Parsed URL → center: %.6f, %.6f", args.center[0], args.center[1])

    if args.center:
        metadata_config = normalize_osm_config(
            {
                "center": list(args.center),
                "size_m": args.size,
                "width_m": args.width,
                "terrain": args.terrain,
                "height_enrichment": not args.no_height_enrichment,
                "terrain_zoom": args.terrain_zoom,
                "ground_pad_m": args.ground_pad,
                "ground_resolution_m": args.ground_resolution,
                "osm_server": args.osm_server,
            }
        )
        scene_xml = build_scene_from_center(
            lat=args.center[0],
            lon=args.center[1],
            size_m=args.size,
            output_dir=args.output,
            width_m=args.width,
            metadata_config=metadata_config,
            **common_kwargs,
        )
    else:
        metadata_config = normalize_osm_config(
            {
                "bbox": list(args.bbox),
                "terrain": args.terrain,
                "height_enrichment": not args.no_height_enrichment,
                "terrain_zoom": args.terrain_zoom,
                "ground_pad_m": args.ground_pad,
                "ground_resolution_m": args.ground_resolution,
                "osm_server": args.osm_server,
            }
        )
        scene_xml = build_scene(
            bbox=tuple(args.bbox),
            output_dir=args.output,
            metadata_config=metadata_config,
            **common_kwargs,
        )

    logger.info("Scene generated: %s", scene_xml)


if __name__ == "__main__":
    main()
