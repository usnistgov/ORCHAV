"""On-demand Angle requests remain usable after visiting ordinary result plots."""

from dataclasses import replace

import numpy as np

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_sensing_panel_fallback import _typed_dashboard_payload
from visualizer.src.sensing.dashboard_data import DashboardSession
from visualizer.src.sensing.sensing_window import SensingWindow


def test_current_angle_current_loads_missing_products_and_reuses_visited_result(
    qtbot, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    payload = _typed_dashboard_payload(scene_frame_id=0, sample_time_s=0.0)
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    writer.append(replace(build_standard_mpc_frame(frame_idx=0), sensing=payload))
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    reads = []
    load_projection = provider.load_frame_projection

    def load(step, projection):
        reads.append((step, projection))
        return load_projection(step, projection)

    monkeypatch.setattr(provider, "load_frame_projection", load)
    session = DashboardSession(provider)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True

    def request(view):
        session.request((0, view, "primary", True, "auto"))
        qtbot.waitUntil(session.pending.done, timeout=3000)
        result = session.pending.result()
        session.pending = None
        assert result is not None
        assert result.effective_display_step == 0
        window.update_from_request(result)
        return result

    try:
        window._set_view_mode("analysis")
        current = request("analysis")
        assert "range_doppler_angle_by_pair" not in current.display_frame["sensing"]

        # Switching layouts can first reuse the Current snapshot. Its missing
        # cube must show pending state until the selected-view worker completes.
        window._set_view_mode("angular")
        assert window._angular_shape is None
        assert "Loading saved angular product" in window._angular_status
        angle = request("angular")
        assert angle.display_data_token is current.display_data_token
        assert not window.advance_playback_context(current)
        assert window._angular_status == "typed primary"
        assert window._angular_shape == (1, 3, 4, 8)
        from visualizer.src.sensing.prepared_angle import prepare_angular

        assert "range_doppler_angle_by_pair" not in angle.display_frame["sensing"]
        np.testing.assert_array_equal(
            angle.prepared_angular.range_azimuth_power,
            prepare_angular(payload).range_azimuth_power,
        )

        read_count = len(reads)
        window._set_view_mode("analysis")
        revisited = request("analysis")
        assert revisited.display_data_token is current.display_data_token
        assert len(reads) == read_count
    finally:
        session.close()
        provider.close()
