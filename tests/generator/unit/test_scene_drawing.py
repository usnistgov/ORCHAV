from types import SimpleNamespace

import numpy as np

from generator.figures.scene import drawing


class RecordingAxis:
    def __init__(self):
        self.calls = []

    def imshow(self, *args, **kwargs):
        self.calls.append(("imshow", args, kwargs))

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))


def test_floor_offset_is_capped_for_normal_and_tall_scenes():
    assert drawing._scaled_floor_offset(2.0) == 0.1
    assert drawing._scaled_floor_offset(20.0) == 0.1


def test_floor_offset_scales_with_shallow_scene_height():
    assert np.isclose(drawing._scaled_floor_offset(0.08), 0.004)


def test_floor_offset_is_zero_for_flat_or_invalid_negative_height():
    assert drawing._scaled_floor_offset(0.0) == 0.0
    assert drawing._scaled_floor_offset(-1.0) == 0.0


def test_rasterized_floor_plan_keeps_shallow_mesh():
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.08, 0.0, 0.08],
                [0.0, 0.08, 0.08],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
    )
    scene_geometry = [{"mesh": mesh, "name": "shallow", "color": [0.1, 0.2, 0.3]}]

    result = drawing.create_rasterized_floor_plan(
        scene_geometry,
        resolution=0.01,
        blur_sigma=0,
        edge_enhancement=False,
    )

    assert result is not None
    image = result[0]
    assert np.any(image[:, :, 3] > 0)


def test_raster_cache_key_records_policy_only_for_automatic_z_range(monkeypatch):
    captured = []

    def capture_cache_path(_scene_geometry, artifact, params):
        captured.append((artifact, params))
        return None

    monkeypatch.setattr(drawing, "_summary_cache_path", capture_cache_path)
    monkeypatch.setattr(drawing, "create_rasterized_floor_plan", lambda *args, **kwargs: None)

    drawing._cached_rasterized_floor_plan(
        [object()], resolution=0.5, blur_sigma=0.0, edge_enhancement=False
    )
    drawing._cached_rasterized_floor_plan(
        [object()],
        resolution=0.5,
        blur_sigma=0.0,
        edge_enhancement=False,
        z_range=(0.0, 1.0),
    )

    assert captured[0][0] == "raster"
    assert captured[0][1]["auto_z_range_policy"] == "scaled-floor-v1"
    assert captured[1][0] == "raster"
    assert "auto_z_range_policy" not in captured[1][1]


def test_rasterized_mode_uses_vector_fallback_when_rasterization_fails(monkeypatch):
    scene_geometry = [object()]
    ax = RecordingAxis()

    monkeypatch.setattr(drawing, "_cached_rasterized_floor_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        drawing,
        "_build_scene_geometry_2d_commands",
        lambda geometry: (("plot", ([0.0, 1.0], [0.0, 1.0]), {"color": "black"}),),
    )

    drawing.plot_scene_geometry_2d(ax, scene_geometry, rendering_mode="rasterized")

    assert [call[0] for call in ax.calls] == ["plot"]


def test_rasterized_floor_plan_reports_progress_for_large_mesh(monkeypatch):
    progress_instances = []

    class FakeProgress:
        def __init__(self, first_step, total_steps, label):
            self.first_step = first_step
            self.total_steps = total_steps
            self.label = label
            self.updates = []
            self.closed = False
            progress_instances.append(self)

        def update(self, step_idx):
            self.updates.append(step_idx)

        def newline(self):
            self.closed = True

    def fake_rasterize_triangles(
        vertices,
        triangles,
        occupancy_r,
        occupancy_g,
        occupancy_b,
        occupancy_alpha,
        z_buffer,
        color,
        x_min,
        y_min,
        resolution,
        progress_callback=None,
    ):
        occupancy_alpha[0, 0] = 1.0
        for _ in triangles:
            progress_callback()

    triangle_count = 3
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3]]),
    )
    scene_geometry = [{"mesh": mesh, "name": "test", "color": [0.1, 0.2, 0.3]}]

    monkeypatch.setattr(drawing, "MIN_RASTER_PROGRESS_PRIMITIVES", 1)
    monkeypatch.setattr(drawing, "StderrProgress", FakeProgress)
    monkeypatch.setattr(drawing, "_rasterize_triangles", fake_rasterize_triangles)

    result = drawing.create_rasterized_floor_plan(scene_geometry, blur_sigma=0)

    assert result is not None
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.total_steps == triangle_count
    assert progress.label == "Scene raster"
    assert progress.updates[-1] == triangle_count - 1
    assert progress.closed is True


def test_scene_figure_cache_stats_track_vector_and_raster_entries(monkeypatch):
    drawing.clear_scene_figure_caches()
    baseline = drawing.get_scene_figure_cache_stats()
    scene_geometry = [object()]
    x_values = np.asarray([0.0, 1.0], dtype=np.float64)
    y_values = np.asarray([1.0, 0.0], dtype=np.float64)

    def record_vector(axis, _geometry):
        axis.plot(x_values, y_values, color="black")

    monkeypatch.setattr(drawing, "_plot_scene_geometry_2d_impl", record_vector)
    first_commands = drawing._build_scene_geometry_2d_commands(scene_geometry)
    second_commands = drawing._build_scene_geometry_2d_commands(scene_geometry)

    image = np.ones((4, 5, 4), dtype=np.float32)
    raster = (image, 0.0, 5.0, 0.0, 4.0)
    monkeypatch.setattr(drawing, "_summary_cache_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        drawing,
        "create_rasterized_floor_plan",
        lambda *_args, **_kwargs: raster,
    )
    raster_kwargs = {
        "resolution": 0.5,
        "blur_sigma": 0.0,
        "edge_enhancement": False,
    }
    first_raster = drawing._cached_rasterized_floor_plan(scene_geometry, **raster_kwargs)
    second_raster = drawing._cached_rasterized_floor_plan(scene_geometry, **raster_kwargs)

    assert first_commands is second_commands
    assert first_raster is second_raster
    stats = drawing.get_scene_figure_cache_stats()
    assert stats["vector_entries"] == 1
    assert stats["vector_current_bytes"] >= x_values.nbytes + y_values.nbytes
    assert stats["vector_peak_bytes"] >= stats["vector_current_bytes"]
    assert stats["vector_hits"] == baseline["vector_hits"] + 1
    assert stats["vector_misses"] == baseline["vector_misses"] + 1
    assert stats["vector_writes"] == baseline["vector_writes"] + 1
    assert stats["raster_entries"] == 1
    assert stats["raster_current_bytes"] >= image.nbytes
    assert stats["raster_peak_bytes"] >= stats["raster_current_bytes"]
    assert stats["raster_hits"] == baseline["raster_hits"] + 1
    assert stats["raster_misses"] == baseline["raster_misses"] + 1
    assert stats["raster_writes"] == baseline["raster_writes"] + 1
    assert stats["current_bytes"] > 0
    assert stats["peak_bytes"] >= stats["current_bytes"]
    assert stats["evictions"] == 0

    drawing.clear_scene_figure_caches()
    cleared = drawing.get_scene_figure_cache_stats()
    assert cleared["vector_entries"] == 0
    assert cleared["raster_entries"] == 0
    assert cleared["current_bytes"] == 0
    assert cleared["peak_bytes"] >= stats["current_bytes"]
    assert cleared["clears"] == stats["clears"] + 1
