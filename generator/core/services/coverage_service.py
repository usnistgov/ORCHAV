"""Coverage service boundary for YAML normalization and output delegation.

``CoverageService`` adapts scenario coverage settings into
``SimulationConfig.coverage`` and then delegates the actual coverage-map
computation, HDF5 writing, and optional figures to the coverage/io/viz layers.
It does not own coverage physics or storage schema details.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...figures.coverage import (
    create_coverage_distribution_figure,
    create_coverage_metric_guide,
    create_coverage_visualization,
)
from ...io.storage.coverage_writer import save_coverage_map
from ..configuration import CoverageConfig
from ..configuration.defaults import (
    DEFAULT_COVERAGE_HEIGHTS_M,
    DEFAULT_COVERAGE_METRIC,
    DEFAULT_COVERAGE_METRICS_DERIVED,
    DEFAULT_COVERAGE_METRICS_STORE,
    DEFAULT_COVERAGE_QUALITY_PRESET,
    DEFAULT_COVERAGE_SAVE_COMPRESSION,
    DEFAULT_COVERAGE_SAVE_PATH,
    DEFAULT_COVERAGE_STRIDE,
    DEFAULT_COVERAGE_TX_MODE,
)
from ..coverage import compute_coverage_map
from .base import BaseService
from .scene_service import SceneService

if TYPE_CHECKING:
    from ...io.storage.coverage_publication import CoveragePublication

CoverageBboxXY = tuple[tuple[float, float], tuple[float, float]]


def _parse_bbox_xy(value: Any) -> CoverageBboxXY:
    """Validate coverage ``bbox_xy`` as ``[[x_min, x_max], [y_min, y_max]]``."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("coverage.grid.bbox_xy must be [[x_min, x_max], [y_min, y_max]]")
    if len(value) != 2:
        raise ValueError("coverage.grid.bbox_xy must contain exactly two axis ranges")

    normalized: list[tuple[float, float]] = []
    for pair in value:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple)):
            raise ValueError("coverage.grid.bbox_xy ranges must be [min, max] pairs")
        if len(pair) != 2:
            raise ValueError("coverage.grid.bbox_xy ranges must contain exactly two values")
        normalized.append((float(pair[0]), float(pair[1])))

    return (normalized[0], normalized[1])


class CoverageService(BaseService):
    """Generator-side service for coverage generation and coverage-file output."""

    def compute_coverage(
        self,
        scene_service: SceneService,
        scenario_configuration: Any | None = None,
        *,
        publication: CoveragePublication | None = None,
    ) -> str | None:
        """Compute the coverage map when the scenario enables coverage.

        The scenario object still carries YAML-shaped names such as
        ``bbox_xy``, ``heights_m``, and nested ``save``/``solver`` blocks.  This
        method translates those names once into ``CoverageConfig`` so the core
        coverage implementation can work with a stable config object.

        Returns the saved HDF5 path, or ``None`` when coverage or its data
        persistence is intentionally disabled. Computation and enabled-output
        failures raise instead of sharing the disabled sentinel.
        """
        if scenario_configuration:
            cov_cfg = getattr(scenario_configuration, "coverage_cfg", None)
            if cov_cfg:
                if self.simulation_config.coverage is None:
                    self.simulation_config.coverage = CoverageConfig()

                self.simulation_config.coverage.enabled = bool(cov_cfg.get("enabled", False))
                bbox = cov_cfg.get("bbox_xy", "auto")
                self.simulation_config.coverage.bbox = None
                self.simulation_config.coverage.bbox_xy = None
                if bbox != "auto" and bbox is not None:
                    # ``bbox_xy`` is the documented two-axis YAML shape.  The
                    # coverage engine also expects a 3D bbox, so derive the z
                    # range from the configured coverage heights.
                    bbox_xy = _parse_bbox_xy(bbox)
                    self.simulation_config.coverage.bbox_xy = bbox_xy
                    heights_for_bbox = cov_cfg.get("heights_m") or DEFAULT_COVERAGE_HEIGHTS_M
                    z_values = [float(h) for h in heights_for_bbox]
                    self.simulation_config.coverage.bbox = (
                        bbox_xy[0],
                        bbox_xy[1],
                        (min(z_values), max(z_values)),
                    )

                res = cov_cfg.get("resolution_m")
                if res:
                    res_tuple = tuple(map(float, res))
                    if len(res_tuple) != 2:
                        raise ValueError("coverage.grid.resolution_m must be [dx, dy]")
                    self.simulation_config.coverage.resolution = res_tuple

                heights = cov_cfg.get("heights_m")
                if heights is not None:
                    self.simulation_config.coverage.heights = [float(h) for h in heights]

                self.simulation_config.coverage.stride = int(
                    cov_cfg.get("stride", DEFAULT_COVERAGE_STRIDE)
                )
                primary_metric = (cov_cfg.get("metrics", {}) or {}).get("derived", [])
                self.simulation_config.coverage.metric = str(
                    primary_metric[0] if primary_metric else DEFAULT_COVERAGE_METRIC
                )
                tx_mode = str(cov_cfg.get("tx_mode", DEFAULT_COVERAGE_TX_MODE))
                self.simulation_config.coverage.tx_combination = tx_mode
                self.simulation_config.coverage.tx_mode = tx_mode

                # Keep logical metric requests explicit. The HDF5 writer stores
                # canonical coverage data and records recipes for derivable
                # metrics instead of materializing every requested layer.
                metrics = cov_cfg.get("metrics", {}) or {}
                self.simulation_config.coverage.metrics_store = list(
                    metrics.get("store", DEFAULT_COVERAGE_METRICS_STORE)
                )
                self.simulation_config.coverage.metrics_derived = list(
                    metrics.get(
                        "derived",
                        DEFAULT_COVERAGE_METRICS_DERIVED,
                    )
                )

                tx_index = cov_cfg.get("tx_selected")
                self.simulation_config.coverage.tx_selected = tx_index
                if tx_index is not None and isinstance(tx_index, int):
                    self.simulation_config.coverage.tx_index = int(tx_index)

                # Solver settings are coverage-specific.  They intentionally do
                # not share the ray-tracing quality block because coverage grid
                # solves have different cost and noise tradeoffs.
                solver_cfg = cov_cfg.get("solver", {}) or {}
                self.simulation_config.coverage.solver_settings = dict(solver_cfg)
                self.simulation_config.coverage.quality = str(
                    solver_cfg.get("preset") or DEFAULT_COVERAGE_QUALITY_PRESET
                )

                rr_depth = solver_cfg.get("rr_depth")
                if rr_depth is not None:
                    self.simulation_config.coverage.rr_depth = int(rr_depth)
                rr_prob = solver_cfg.get("rr_prob")
                if rr_prob is not None:
                    self.simulation_config.coverage.rr_prob = float(rr_prob)
                stop_threshold = solver_cfg.get("stop_threshold")
                if stop_threshold is not None:
                    self.simulation_config.coverage.stop_threshold = float(stop_threshold)
                seed = solver_cfg.get("seed")
                if seed is not None:
                    self.simulation_config.coverage.seed = int(seed)

                # Transmit precoding vector
                precoding_vec = cov_cfg.get("precoding_vec")
                if precoding_vec is not None:
                    self.simulation_config.coverage.precoding_vec = [
                        float(v) for v in precoding_vec
                    ]
                save_cfg = cov_cfg.get("save", {}) or {}
                self.simulation_config.coverage.save_path = DEFAULT_COVERAGE_SAVE_PATH
                self.simulation_config.coverage.save_compression = str(
                    save_cfg.get("compression", DEFAULT_COVERAGE_SAVE_COMPRESSION)
                )

        if not (self.simulation_config.coverage and self.simulation_config.coverage.enabled):
            return None

        self.logger.info("Coverage mode enabled - computing coverage map")

        coverage_data = compute_coverage_map(
            scene=scene_service.scene,
            tx_list=scene_service.tx_list,
            rx_list=scene_service.rx_list,
            target_objects=[tm.target_object for tm in scene_service.target_managers],
            coverage_config=self.simulation_config.coverage,
            simulation_config=self.simulation_config,
            scenario_context=scenario_configuration,
        )

        if coverage_data is None:
            raise RuntimeError("Coverage map computation failed")

        output_file = (
            publication.stage(coverage_data, scenario_configuration)
            if publication is not None
            else save_coverage_map(coverage_data, scenario_configuration)
        )
        return output_file

    def generate_summary_figures(
        self,
        coverage_file: Path,
        scenario_configuration: Any,
        *,
        summary_root: Path,
        strict: bool = False,
    ) -> list[Path]:
        """Write requested coverage figures below a supplied summary root.

        The offline pipeline supplies its private summary staging directory and
        uses strict mode. This keeps coverage HDF5 persistence authoritative
        while preventing a partially rendered summary tree from being
        published.
        """
        save_root = scenario_configuration.coverage_cfg.get("save", {}) or {}
        fig_spec = save_root.get("figure", {}) or {}
        if not fig_spec or not bool(fig_spec.get("enabled", False)):
            return []

        out_dir = Path(summary_root) / "coverage"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_format = fig_spec.get("format", "png")
        filename = fig_spec.get("filename", "coverage_maps")
        output_path = out_dir / f"{filename}.{out_format}"

        interpolation = str(fig_spec.get("interpolation", "nearest"))
        overlay_scene = bool(fig_spec.get("overlay_scene", False))
        overlay_alpha = float(fig_spec.get("overlay_alpha", 0.3))
        overlay_style = str(fig_spec.get("overlay_style", "outline"))
        generated: list[Path] = []

        try:
            generated.extend(
                Path(path)
                for path in create_coverage_visualization(
                    coverage_file=coverage_file,
                    output_path=output_path,
                    interpolation=interpolation,
                    overlay_scene=overlay_scene,
                    overlay_alpha=overlay_alpha,
                    overlay_style=overlay_style,
                    scenario_context=scenario_configuration,
                    show_tx=bool(fig_spec.get("show_tx", True)),
                )
            )
            metric_specs = list(fig_spec.get("metrics", []) or [])
            if metric_specs:
                metric_filename = str(fig_spec.get("metric_filename", "coverage_metrics"))
                metric_output_path = out_dir / f"{metric_filename}.{out_format}"
                generated.extend(
                    Path(path)
                    for path in create_coverage_metric_guide(
                        coverage_file=coverage_file,
                        output_path=metric_output_path,
                        metrics=metric_specs,
                        columns=int(fig_spec.get("columns", 3)),
                        interpolation=interpolation,
                        overlay_scene=overlay_scene,
                        overlay_alpha=overlay_alpha,
                        overlay_style=overlay_style,
                        scenario_context=scenario_configuration,
                        show_tx=bool(fig_spec.get("show_tx", True)),
                    )
                )
            distribution_spec = fig_spec.get("distribution", {}) or {}
            distribution_metrics = list(distribution_spec.get("metrics", []) or [])
            if bool(distribution_spec.get("enabled", False)) and distribution_metrics:
                distribution_filename = str(
                    distribution_spec.get("filename", "coverage_distributions")
                )
                distribution_output_path = out_dir / f"{distribution_filename}.{out_format}"
                generated.extend(
                    Path(path)
                    for path in create_coverage_distribution_figure(
                        coverage_file=coverage_file,
                        output_path=distribution_output_path,
                        metrics=distribution_metrics,
                        bins=int(distribution_spec.get("bins", 40)),
                    )
                )
        except (OSError, ValueError, RuntimeError) as exc:
            if strict:
                raise RuntimeError("Coverage summary figure generation failed") from exc
            self.logger.warning("Coverage figure generation skipped: %s", exc)

        return generated
