from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from generator.core.configuration import SimulationConfig
from generator.core.services.coverage_service import CoverageService, _parse_bbox_xy


def test_parse_bbox_xy_accepts_two_axis_ranges():
    assert _parse_bbox_xy([[0, 10], ["-5.5", 7]]) == ((0.0, 10.0), (-5.5, 7.0))


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 1, 2, 3],
        [[0, 1, 2], [3, 4, 5]],
        [[0, 1]],
        "0,1,2,3",
        [[0, 1], "2,3"],
    ],
)
def test_parse_bbox_xy_rejects_invalid_shapes(bbox):
    with pytest.raises(ValueError, match="coverage.grid.bbox_xy"):
        _parse_bbox_xy(bbox)


class TestCoverageService:
    @pytest.fixture
    def mock_simulation_config(self):
        config = MagicMock(spec=SimulationConfig)
        config.coverage = None
        return config

    @patch("generator.core.services.coverage_service.compute_coverage_map")
    @patch("generator.core.services.coverage_service.save_coverage_map")
    def test_compute_coverage_delegation(self, mock_save, mock_compute, mock_simulation_config):
        service = CoverageService(mock_simulation_config)

        # Setup services
        mock_scene_service = MagicMock()
        mock_scene_service.tx_list = [1]
        mock_scene_service.rx_list = [2]
        mock_scene_service.target_managers = []

        # Setup config to enable coverage
        scenario_config = MagicMock()
        scenario_config.coverage_cfg = {"enabled": True, "resolution": [1.0, 1.0]}

        # Setup compute return
        mock_compute.return_value = {"data": "test"}
        mock_save.return_value = "coverage.h5"

        # Execute
        result = service.compute_coverage(
            mock_scene_service, scenario_configuration=scenario_config
        )

        # Verify
        mock_compute.assert_called_once()
        mock_save.assert_called_once_with({"data": "test"}, scenario_config)
        assert result == "coverage.h5"

        # Verify config update
        assert mock_simulation_config.coverage.enabled is True
        assert mock_simulation_config.coverage.resolution == (1.0, 1.0)

    def test_compute_coverage_sets_valid_bbox(self, mock_simulation_config):
        service = CoverageService(mock_simulation_config)
        mock_scene_service = MagicMock()
        scenario_config = MagicMock()
        scenario_config.coverage_cfg = {
            "enabled": False,
            "bbox_xy": [[-1, 2], [3, 4]],
            "heights_m": [1, 8],
        }

        result = service.compute_coverage(mock_scene_service, scenario_config)

        assert result is None
        assert mock_simulation_config.coverage.bbox_xy == ((-1.0, 2.0), (3.0, 4.0))
        assert mock_simulation_config.coverage.bbox == ((-1.0, 2.0), (3.0, 4.0), (1.0, 8.0))

    @patch("generator.core.services.coverage_service.compute_coverage_map")
    def test_compute_coverage_raises_when_solver_returns_no_result(
        self, mock_compute, mock_simulation_config
    ):
        service = CoverageService(mock_simulation_config)
        mock_scene_service = MagicMock()
        mock_scene_service.target_managers = []
        scenario_config = MagicMock()
        scenario_config.coverage_cfg = {"enabled": True}
        mock_compute.return_value = None

        with pytest.raises(RuntimeError, match="Coverage map computation failed"):
            service.compute_coverage(mock_scene_service, scenario_config)

    @patch("generator.core.services.coverage_service.create_coverage_visualization")
    @patch("generator.core.services.coverage_service.save_coverage_map")
    @patch("generator.core.services.coverage_service.compute_coverage_map")
    def test_primary_write_failure_propagates_before_optional_figure(
        self,
        mock_compute,
        mock_save,
        mock_figure,
        mock_simulation_config,
    ):
        service = CoverageService(mock_simulation_config)
        mock_scene_service = MagicMock()
        mock_scene_service.target_managers = []
        scenario_config = MagicMock()
        scenario_config.coverage_cfg = {
            "enabled": True,
            "save": {"figure": {"enabled": True}},
        }
        mock_compute.return_value = {"path_gain_linear": "computed"}
        mock_save.side_effect = PermissionError("coverage destination is read-only")

        with pytest.raises(PermissionError, match="read-only"):
            service.compute_coverage(mock_scene_service, scenario_config)

        mock_figure.assert_not_called()

    @patch("generator.core.services.coverage_service.create_coverage_visualization")
    @patch("generator.core.services.coverage_service.save_coverage_map")
    @patch("generator.core.services.coverage_service.compute_coverage_map")
    def test_coverage_figures_are_deferred_to_the_summary_publication_stage(
        self,
        mock_compute,
        mock_save,
        mock_figure,
        mock_simulation_config,
        tmp_path,
    ):
        service = CoverageService(mock_simulation_config)
        mock_scene_service = MagicMock()
        mock_scene_service.target_managers = []
        scenario_config = MagicMock()
        scenario_config.root = tmp_path
        scenario_config.coverage_cfg = {
            "enabled": True,
            "save": {"figure": {"enabled": True}},
        }
        output_path = tmp_path / "coverage.h5"
        output_path.touch()
        mock_compute.return_value = {"path_gain_linear": "computed"}
        mock_save.return_value = str(output_path)
        mock_figure.side_effect = OSError("figure backend unavailable")

        result = service.compute_coverage(mock_scene_service, scenario_config)

        assert result == str(output_path)
        mock_figure.assert_not_called()

        generated = service.generate_summary_figures(
            output_path,
            scenario_config,
            summary_root=tmp_path / "summary-staging",
        )

        assert generated == []
        mock_figure.assert_called_once()
        assert mock_figure.call_args.kwargs["output_path"] == (
            tmp_path / "summary-staging" / "coverage" / "coverage_maps.png"
        )

    @patch("generator.core.services.coverage_service.create_coverage_visualization")
    def test_strict_coverage_figure_failure_propagates(self, mock_figure, tmp_path):
        service = CoverageService(MagicMock(spec=SimulationConfig))
        scenario_config = SimpleNamespace(coverage_cfg={"save": {"figure": {"enabled": True}}})
        mock_figure.side_effect = OSError("figure backend unavailable")

        with pytest.raises(RuntimeError, match="Coverage summary figure generation failed"):
            service.generate_summary_figures(
                tmp_path / "coverage" / "coverage_maps.h5",
                scenario_config,
                summary_root=tmp_path / "summary-staging",
                strict=True,
            )

    @patch("generator.core.services.coverage_service.create_coverage_distribution_figure")
    @patch("generator.core.services.coverage_service.create_coverage_metric_guide")
    @patch("generator.core.services.coverage_service.create_coverage_visualization")
    def test_summary_figure_helpers_generate_all_heights(
        self,
        mock_visualization,
        mock_metric_guide,
        mock_distribution,
        tmp_path,
    ):
        service = CoverageService(MagicMock(spec=SimulationConfig))
        scenario_config = SimpleNamespace(
            coverage_cfg={
                "save": {
                    "figure": {
                        "enabled": True,
                        "metrics": ["best_path_loss_db"],
                        "distribution": {
                            "enabled": True,
                            "metrics": ["best_path_loss_db"],
                        },
                    }
                }
            }
        )
        map_paths = [
            tmp_path / "coverage_maps_height-01_1.5m.png",
            tmp_path / "coverage_maps_height-02_30m.png",
        ]
        metric_paths = [
            tmp_path / "coverage_metrics_height-01_1.5m.png",
            tmp_path / "coverage_metrics_height-02_30m.png",
        ]
        distribution_paths = [
            tmp_path / "coverage_distributions_height-01_1.5m.png",
            tmp_path / "coverage_distributions_height-02_30m.png",
        ]
        mock_visualization.return_value = map_paths
        mock_metric_guide.return_value = metric_paths
        mock_distribution.return_value = distribution_paths

        generated = service.generate_summary_figures(
            tmp_path / "coverage" / "coverage_maps.h5",
            scenario_config,
            summary_root=tmp_path / "summary-staging",
            strict=True,
        )

        assert generated == map_paths + metric_paths + distribution_paths
        assert "height_index" not in mock_metric_guide.call_args.kwargs
        assert "height_index" not in mock_distribution.call_args.kwargs

    @patch("generator.core.services.coverage_service.save_coverage_map")
    @patch("generator.core.services.coverage_service.compute_coverage_map")
    def test_disabled_primary_persistence_remains_successful(
        self,
        mock_compute,
        mock_save,
        mock_simulation_config,
    ):
        service = CoverageService(mock_simulation_config)
        mock_scene_service = MagicMock()
        mock_scene_service.target_managers = []
        scenario_config = MagicMock()
        scenario_config.coverage_cfg = {
            "enabled": True,
            "save": {"data": {"enabled": False}},
        }
        mock_compute.return_value = {"path_gain_linear": "computed"}
        mock_save.return_value = None

        assert service.compute_coverage(mock_scene_service, scenario_config) is None
