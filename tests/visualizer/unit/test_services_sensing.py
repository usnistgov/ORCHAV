from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid5

import numpy as np
import pytest

from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    UnsupportedSensingSchemaError,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
)
from shared.frames import (
    FrameProjection,
    FrameReadRequest,
    StandardMPCFrame,
    project_standard_mpc_frame,
)
from shared.frames.provider_base import DataProvider, ProviderInfo
from shared.sensing.products import (
    add_product_spec,
    attach_product_descriptors,
    build_product_descriptor,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.fixtures.sensing_v3 import (
    build_visualizer_sensing_payload,
    materialize_visualizer_product_specs,
)
from visualizer.src.sensing.cir_provider import (
    StoredCIRProvider,
    extract_timestamped_fast_time_samples,
)
from visualizer.src.sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    InspectionDisposition,
)
from visualizer.src.services.sensing_service import (
    _RANGE_PROFILE_CONTRACT,
    SensingReplayCacheState,
    SensingReplayTelemetry,
    SensingReplayTelemetrySnapshot,
    SensingService,
    _require_single_pair_product,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _frame_cache(frames: dict[int, dict]) -> SimpleNamespace:
    """Return a cache double exposing deterministic storage-row inventory."""
    return SimpleNamespace(
        get_frame=lambda step: frames.get(step),
        frame_cache=SimpleNamespace(keys=lambda: list(frames)),
    )


def _multi_sample_sensing(
    products: dict,
    *,
    sample_times_s: list[float],
    strategy: str,
) -> dict:
    """Build a valid multi-sample sensing payload for provider tests."""
    typed_products = dict(products)
    legacy_cir = typed_products.pop("cir", None)
    legacy_slow_time = typed_products.pop("slow_time_matrix", None)
    range_size: int | None = None
    if legacy_cir is not None:
        cir = np.asarray(legacy_cir, dtype=np.complex128)
        range_size = int(cir.shape[-1])
        typed_products["cir_by_pair_rx_element"] = cir[np.newaxis, np.newaxis, :]
    if legacy_slow_time is not None:
        slow_time = np.asarray(legacy_slow_time, dtype=np.complex128)
        range_size = int(slow_time.shape[-1])
        typed_products["slow_time_by_pair_rx_element"] = slow_time[np.newaxis, np.newaxis, :, :]
    if range_size is not None:
        typed_products["range_axis_m"] = np.arange(range_size, dtype=np.float64) ** 2 * 0.125

    base = build_visualizer_sensing_payload()
    base_temporal = base["temporal"]
    clock = base_temporal["clock"]
    acquisition = base_temporal["acquisition"]
    result = base_temporal["result"]
    samples = [
        {
            "sample_index": index,
            "sample_time_s": sample_time_s,
            "valid": True,
            "phase_coherent": strategy == "coherent_cpi_fft",
            "source": {
                "source_generation_id": clock["clock_id"],
                "scene_frame_id": "provider-test",
                "sample_offset": index,
            },
        }
        for index, sample_time_s in enumerate(sample_times_s)
    ]
    temporal = build_sensing_temporal_contract(
        clock_id=clock["clock_id"],
        acquisition_id=str(
            uuid5(
                UUID(acquisition["acquisition_id"]),
                f"{strategy}:{sample_times_s!r}",
            )
        ),
        acquisition_config_id=canonical_sensing_hash(
            {"strategy": strategy, "sample_times_s": sample_times_s}
        ),
        pair_id=acquisition["pair_id"],
        tx_sensor_id=acquisition["tx_sensor_id"],
        rx_sensor_id=acquisition["rx_sensor_id"],
        samples=samples,
        processing_run_id=result["processing_run_id"],
        processing_recipe_id=result["processing_recipe_id"],
        reference_time_s=(sample_times_s[0] + sample_times_s[-1]) / 2.0,
        reference_time_policy="test_acquisition_center",
        display_scene_frame_id=0,
        display_policy="exact",
        display_reason="Timestamped provider test fixture.",
    )
    base.update(
        {
            "schema_version": SENSING_SCHEMA_VERSION,
            "temporal": temporal,
            "processing_strategy": strategy,
            **typed_products,
        }
    )
    resolved = base["config_resolved"]
    sample_source = "coherent_one_solve" if strategy == "coherent_cpi_fft" else "independent_solves"
    resolved["acquisition"].update(
        {
            "sample_source": sample_source,
            "rt_solve_stride_scene_steps": (
                len(sample_times_s) if strategy == "coherent_cpi_fft" else 1
            ),
            "sample_period_s": (
                sample_times_s[1] - sample_times_s[0] if len(sample_times_s) > 1 else 0.001
            ),
            "sample_rate_hz": (
                1.0 / (sample_times_s[1] - sample_times_s[0]) if len(sample_times_s) > 1 else 1000.0
            ),
            "cpi_length_samples": len(sample_times_s),
            "cpi_hop_samples": len(sample_times_s) if strategy == "coherent_cpi_fft" else 1,
            "overlap_samples": 0 if strategy == "coherent_cpi_fft" else len(sample_times_s) - 1,
            "phase_contract": (
                "sionna_single_solve_time_axis"
                if strategy == "coherent_cpi_fft"
                else "independent_solve_phase_continuity_controlled_specular_only"
            ),
        }
    )
    resolved["processing"].update(
        {
            "mode": "coherent_cpi" if strategy == "coherent_cpi_fft" else "sequential_stft",
            "strategy": strategy,
            "fft_size_doppler": len(sample_times_s),
            "result_stride_samples": (len(sample_times_s) if strategy == "coherent_cpi_fft" else 1),
        }
    )
    resolved["fft_size_doppler"] = len(sample_times_s)
    resolved["processing_mode"] = resolved["processing"]["mode"]
    base["config"] = deepcopy(resolved)
    materialize_visualizer_product_specs(base)
    return base


def _add_range_doppler_variant(
    sensing: dict,
    *,
    data_key: str,
    values: np.ndarray,
    variant: str,
    representation: str,
    range_semantics: str = "monostatic_range",
) -> None:
    """Attach one arbitrarily named RD fixture through its exact descriptor."""
    sensing[data_key] = np.asarray(values)
    add_product_spec(
        sensing,
        data_key=data_key,
        kind="range_doppler",
        variant=variant,
        axes=(
            catalog_axis("pair", size=1, catalog_key="pair_axis"),
            dense_axis(
                "doppler_frequency",
                size=int(values.shape[1]),
                unit="Hz",
                coordinate_key="doppler_axis_hz",
                coordinate_semantics="fftshifted_doppler_frequency",
            ),
            dense_axis(
                "range",
                size=int(values.shape[2]),
                unit="m",
                coordinate_key="range_axis_m",
                coordinate_semantics=range_semantics,
                reference_frame="tx_target_rx_path",
            ),
        ),
        representation=representation,
        unit="1",
        combine={"policy": "siso_identity", "normalization": "identity"},
    )
    materialize_product_specs(sensing)


def _detector_stage_sensing(
    *,
    scene_frame_id: int = 5,
    sample_time_s: float = 5.0,
    range_doppler_fill: float = 1.0,
) -> dict:
    """Return one complete typed Phase-4 detector-stage result."""
    shape = (1, 2, 3)
    sensing = build_visualizer_sensing_payload(
        {
            "detections": [
                {
                    "cluster_id": 0,
                    "doppler_bin": 0,
                    "range_bin": 1,
                    "range_m": 1.0,
                    "snr_db": 3.0,
                }
            ],
            "doppler_axis_hz": np.array([-2.0, 2.0], dtype=np.float64),
            "range_axis_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "range_doppler_by_pair": np.full(
                shape,
                range_doppler_fill,
                dtype=np.complex64,
            ),
        },
        scene_frame_id=scene_frame_id,
        sample_time_s=sample_time_s,
    )
    values = {
        "range_doppler_detection_input_power_by_pair": np.ones(shape, dtype=np.float64),
        "range_doppler_cfar_noise_power_by_pair": np.full(shape, 0.5, dtype=np.float64),
        "range_doppler_cfar_threshold_power_by_pair": np.full(shape, 0.75, dtype=np.float64),
        "range_doppler_cfar_training_cell_count_by_pair": np.full(shape, 8, dtype=np.int32),
        "range_doppler_cfar_eligible_mask_by_pair": np.ones(shape, dtype=np.bool_),
        "range_doppler_cfar_mask_by_pair": np.zeros(shape, dtype=np.bool_),
        "range_doppler_nms_mask_by_pair": np.zeros(shape, dtype=np.bool_),
        "range_doppler_snr_db_by_pair": np.full(shape, 3.0, dtype=np.float64),
        "range_doppler_snr_valid_mask_by_pair": np.ones(shape, dtype=np.bool_),
        "range_doppler_cluster_labels_by_pair": np.full(shape, -1, dtype=np.int32),
    }
    values["range_doppler_cfar_mask_by_pair"][0, 0, 1] = True
    values["range_doppler_nms_mask_by_pair"][0, 0, 1] = True
    values["range_doppler_cluster_labels_by_pair"][0, 0, 1] = 0
    sensing.update(values)
    axes = (
        catalog_axis("pair", size=1, catalog_key="pair_axis"),
        dense_axis(
            "doppler_frequency",
            size=2,
            unit="Hz",
            coordinate_key="doppler_axis_hz",
        ),
        dense_axis("range", size=3, unit="m", coordinate_key="range_axis_m"),
    )
    specs = (
        (
            "range_doppler_detection_input_power_by_pair",
            "range_doppler_detection_input_power",
            "power_linear",
            ("range_doppler_by_pair",),
        ),
        (
            "range_doppler_cfar_noise_power_by_pair",
            "range_doppler_cfar_noise_power",
            "power_linear",
            ("range_doppler_detection_input_power_by_pair",),
        ),
        (
            "range_doppler_cfar_training_cell_count_by_pair",
            "range_doppler_cfar_training_cell_count",
            "integer_count",
            ("range_doppler_detection_input_power_by_pair",),
        ),
        (
            "range_doppler_cfar_eligible_mask_by_pair",
            "range_doppler_cfar_eligible_mask",
            "boolean_mask",
            (
                "range_doppler_detection_input_power_by_pair",
                "range_doppler_cfar_training_cell_count_by_pair",
            ),
        ),
        (
            "range_doppler_cfar_threshold_power_by_pair",
            "range_doppler_cfar_threshold_power",
            "power_linear",
            (
                "range_doppler_cfar_noise_power_by_pair",
                "range_doppler_cfar_eligible_mask_by_pair",
            ),
        ),
        (
            "range_doppler_cfar_mask_by_pair",
            "range_doppler_cfar_mask",
            "boolean_mask",
            (
                "range_doppler_detection_input_power_by_pair",
                "range_doppler_cfar_threshold_power_by_pair",
                "range_doppler_cfar_eligible_mask_by_pair",
            ),
        ),
        (
            "range_doppler_snr_db_by_pair",
            "range_doppler_snr_db",
            "log_power_ratio_db",
            (
                "range_doppler_detection_input_power_by_pair",
                "range_doppler_cfar_noise_power_by_pair",
                "range_doppler_cfar_eligible_mask_by_pair",
            ),
        ),
        (
            "range_doppler_snr_valid_mask_by_pair",
            "range_doppler_snr_valid_mask",
            "boolean_mask",
            (
                "range_doppler_cfar_noise_power_by_pair",
                "range_doppler_cfar_eligible_mask_by_pair",
            ),
        ),
        (
            "range_doppler_nms_mask_by_pair",
            "range_doppler_nms_mask",
            "boolean_mask",
            (
                "range_doppler_detection_input_power_by_pair",
                "range_doppler_cfar_mask_by_pair",
                "range_doppler_snr_valid_mask_by_pair",
            ),
        ),
        (
            "range_doppler_cluster_labels_by_pair",
            "range_doppler_cluster_labels",
            "integer_label",
            (
                "range_doppler_nms_mask_by_pair",
                "range_doppler_snr_db_by_pair",
                "range_doppler_snr_valid_mask_by_pair",
            ),
        ),
    )
    for data_key, kind, representation, source_data_keys in specs:
        add_product_spec(
            sensing,
            data_key=data_key,
            kind=kind,
            axes=axes,
            representation=representation,
            unit="dB" if representation == "log_power_ratio_db" else "1",
            source_data_keys=source_data_keys,
        )
    materialize_product_specs(sensing)
    return sensing


class _CompleteFrameOnlyProvider(DataProvider):
    """Provider that exercises the default complete-frame projection path."""

    def __init__(self, frame: StandardMPCFrame) -> None:
        self.frame = frame
        self.complete_loads = 0
        self.requests: list[FrameReadRequest] = []

    def list_frames(self) -> list[int]:
        return [self.frame.frame_index]

    def has_frame(self, step: int) -> bool:
        return step == self.frame.frame_index

    def load_frame(self, step: int) -> StandardMPCFrame:
        if not self.has_frame(step):
            raise KeyError(step)
        self.complete_loads += 1
        return self.frame

    def load_frame_projection(self, step: int, request: FrameReadRequest) -> FrameProjection:
        self.requests.append(request)
        return super().load_frame_projection(step, request)


class _CountingMultiFrameProvider(DataProvider):
    """Count selective reads across an immutable multi-frame inventory."""

    def __init__(self, frames: dict[int, StandardMPCFrame]) -> None:
        self.frames = frames
        self.requests: list[tuple[int, FrameReadRequest]] = []

    def list_frames(self) -> list[int]:
        return sorted(self.frames)

    def has_frame(self, step: int) -> bool:
        return step in self.frames

    def load_frame(self, step: int) -> StandardMPCFrame:
        try:
            return self.frames[step]
        except KeyError:
            raise KeyError(step) from None

    def load_frame_projection(self, step: int, request: FrameReadRequest) -> FrameProjection:
        self.requests.append((step, request))
        return super().load_frame_projection(step, request)


class _GenerationSwitchingProvider(_CountingMultiFrameProvider):
    """Switch generations immediately after the first dense projection."""

    def __init__(
        self,
        frames: dict[int, StandardMPCFrame],
        next_frames: dict[int, StandardMPCFrame],
    ) -> None:
        super().__init__(frames)
        self.next_frames = next_frames
        self.generation_id = "generation-a"
        self.switch_count = 0

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name=type(self).__name__,
            source="unit-test",
            total_frames=len(self.frames),
            generation_id=self.generation_id,
            frame_set_id="stable-frame-set",
        )

    def load_frame_projection(self, step: int, request: FrameReadRequest) -> FrameProjection:
        projection = super().load_frame_projection(step, request)
        if self.switch_count == 0 and any(
            name.startswith("range_doppler_") and name != "range_doppler_map"
            for name in request.sensing_products
        ):
            self.frames = self.next_frames
            self.generation_id = "generation-b"
            self.switch_count += 1
        return projection


def test_immutable_inventory_avoids_per_frame_enumeration_and_invalidates_refresh():
    class SnapshotProvider(_CountingMultiFrameProvider):
        snapshot = tuple(range(24576))

        @property
        def frame_ids_snapshot(self):
            return self.snapshot

        @property
        def info(self):
            return ProviderInfo(name="test", source="test")

        def list_frames(self):
            pytest.fail("Immutable inventory should not be enumerated during playback")

    provider = SnapshotProvider({})
    service = SensingService(SimpleNamespace(frame_loader=SimpleNamespace(provider=provider)))
    first = service.candidate_storage_frame_ids(0)
    token = service._refresh_metadata_index_token(first)
    for step in (1, 128, 4096):
        assert service.candidate_storage_frame_ids(step) is first
        assert service._refresh_metadata_index_token(first) is token
        assert service._current_metadata_index_token(first)[-1] is first
    # Same-length replacement and a same-ID refresh must both retire derived data.
    for snapshot in (tuple(range(1, 24577)), tuple(list(provider.snapshot))):
        provider.snapshot = snapshot
        candidate = service.candidate_storage_frame_ids(128)
        refreshed = service._refresh_metadata_index_token(candidate)
        assert refreshed is not token
        assert set(candidate) == set(snapshot)
        token = refreshed


def test_nonzero_hdf5_inventory_reuses_index_across_navigation_and_selection(tmp_path, monkeypatch):
    from threading import Event

    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.frames.providers import Hdf5Provider

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    for step in (64, 65, 128, 129):
        frame = replace(build_standard_mpc_frame(frame_idx=step), timestamp_s=step * 0.01)
        if step in (65, 129):
            frame = replace(
                frame,
                sensing=build_visualizer_sensing_payload(
                    {
                        "range_axis_m": np.arange(3.0),
                        "doppler_axis_hz": np.array([-1.0, 1.0]),
                        "range_doppler_power_by_pair": np.full((1, 2, 3), float(step)),
                    },
                    scene_frame_id=step - 1,
                    sample_time_s=(step - 1) * 0.01,
                ),
            )
        writer.append(frame)
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    progress = []
    service.configure_background_reader(Event(), lambda *event: progress.append(event))
    try:
        candidates = service.candidate_storage_frame_ids(0)
        assert set(candidates) == {64, 65, 128, 129}
        for step in (0, 64, 129, 999):
            assert service.candidate_storage_frame_ids(step) is candidates
        rows = service.dashboard_result_rows()
        assert {storage for storage, _ in rows} == {65, 129}
        index_token = service.dashboard_inventory_token()
        for scene_id in (64, 128, 64):
            assert service.sensing_playback_frame_ids() == (64, 128)
            selection = service.resolve_display_sensing_frame(
                service.dashboard_scene_frame(scene_id),
                scene_frame_id=scene_id,
                storage_frame_id=scene_id,
                product_names=("range_doppler",),
            )
            assert selection is not None
            assert selection.storage_frame_id == scene_id + 1
            assert selection.bound_scene_frame_id == scene_id
            np.testing.assert_array_equal(
                selection.display_frame["sensing"]["range_doppler_power_by_pair"],
                np.full((1, 2, 3), float(scene_id + 1)),
            )
            assert service.dashboard_result_rows() is rows
            assert service.dashboard_inventory_token() is index_token
        assert sum(stage == "Indexing results" and done == 0 for stage, done, _ in progress) == 1
    finally:
        provider.close()


@pytest.mark.parametrize("snapshot", [None, (64,)])
def test_dynamic_and_independently_mapped_sources_keep_candidate_union(snapshot):
    class Provider(_CountingMultiFrameProvider):
        @property
        def frame_ids_snapshot(self):
            return snapshot

    provider = Provider({64: build_standard_mpc_frame(frame_idx=64)})
    service = SensingService(
        SimpleNamespace(
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=SimpleNamespace(list_frames=lambda: [128]),
            cache_service=_frame_cache({256: {}}),
        )
    )
    assert set(service.candidate_storage_frame_ids(512)) == {64, 128, 256, 512}
    provider.frames[65] = build_standard_mpc_frame(frame_idx=65)
    assert set(service.candidate_storage_frame_ids(512)) == {64, 65, 128, 256, 512}


def test_sensing_playback_uses_bindings_including_empty_detections_and_skips_missing_pose():
    frames = {step: build_standard_mpc_frame(frame_idx=step) for step in (2, 11, 29, 40, 51)}
    # Deliberately irregular storage rows, duplicate bindings, and an unavailable
    # bound pose (0). Neither a stride nor the result's storage ID is the answer.
    for storage, bound in ((29, 2), (40, 11), (51, 11)):
        frames[storage] = replace(
            frames[storage],
            sensing=build_visualizer_sensing_payload(
                {"detections": []}, scene_frame_id=bound, sample_time_s=float(bound)
            ),
        )
    frames[11] = replace(
        frames[11],
        sensing=build_visualizer_sensing_payload({"detections": []}, scene_frame_id=0),
    )
    provider = _CountingMultiFrameProvider(frames)
    service = SensingService(SimpleNamespace(frame_loader=SimpleNamespace(provider=provider)))
    assert service.sensing_playback_frame_ids() == (2, 11)
    assert all(not request.all_sensing_products for _, request in provider.requests)
    assert all(request.sensing_products == {"products"} for _, request in provider.requests)


def test_private_registration_keeps_heavy_runtime_modules_lazy() -> None:
    """Registry discovery must not import disabled panels or renderer adapters."""
    probe = """
import sys
from pathlib import Path

if (Path.cwd() / "RELEASE_PROVENANCE.json").is_file():
    # A curated export tested from a private editable environment must not
    # discover that environment's intentionally omitted extension package.
    sys.modules["visualizer_extensions"] = None

from visualizer.src.io.frame_source_extensions import registered_frame_source_modes

registered_frame_source_modes()
forbidden = {
    'visualizer.src.beamforming.ns3_mode',
    'visualizer.src.io.measurement_frame_source',
    'visualizer.src.io.ns3_beamforming_loader',
    'visualizer.src.panels.sensing_panel',
    'visualizer.src.services.sensing_service',
    'visualizer.src.renderers.pygfx.sensing',
}
loaded = sorted(forbidden.intersection(sys.modules))
if loaded:
    raise SystemExit('eager private imports: ' + ', '.join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_history_load_fallback_does_not_retain_unrequested_products() -> None:
    """Complete-frame providers return only the requested detection product."""

    detections = [{"track_id": 4, "x_m": 1.0, "y_m": 2.0, "z_m": 0.0}]
    frame = replace(
        build_standard_mpc_frame(frame_idx=3),
        sensing=build_visualizer_sensing_payload(
            {
                "detections": detections,
                "range_axis_m": np.arange(1024, dtype=np.float64),
                "range_profile_by_pair": np.arange(1024, dtype=np.float32)[None, :],
            },
            scene_frame_id=3,
            sample_time_s=3.0,
        ),
    )
    provider = _CompleteFrameOnlyProvider(frame)
    visualizer = SimpleNamespace(
        cache_service=SimpleNamespace(get_frame=lambda _step: None),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=None,
    )

    loaded = SensingService(visualizer)._load_history_frame(3)

    assert loaded is not None
    assert loaded["sensing"]["detections"] == detections
    assert loaded["sensing"]["schema_version"] == SENSING_SCHEMA_VERSION == 4
    assert "temporal" in loaded["sensing"]
    assert "range_profile_by_pair" not in loaded["sensing"]
    assert "range_axis_m" not in loaded["sensing"]
    assert provider.complete_loads == 1
    assert provider.requests[0].sensing_products == frozenset({"detections"})


def test_history_load_projects_direct_standard_frame_without_visual_geometry() -> None:
    """A direct source retains detections without constructing a visual payload."""

    detections = [{"track_id": 8, "x_m": 2.0, "y_m": 3.0, "z_m": 0.0}]
    frame = replace(
        build_standard_mpc_frame(frame_idx=5),
        sensing=build_visualizer_sensing_payload(
            {
                "detections": detections,
                "range_axis_m": np.arange(2048, dtype=np.float64),
                "range_profile_by_pair": np.arange(2048, dtype=np.float32)[None, :],
            },
            scene_frame_id=5,
            sample_time_s=5.0,
        ),
    )
    visualizer = SimpleNamespace(
        cache_service=SimpleNamespace(get_frame=lambda _step: None),
        frame_loader=None,
        frame_source=SimpleNamespace(load_frame=lambda _step: frame),
    )

    loaded = SensingService(visualizer)._load_history_frame(5)

    assert loaded is not None
    assert loaded["sensing"]["detections"] == detections
    assert loaded["sensing"]["schema_version"] == SENSING_SCHEMA_VERSION == 4
    assert "temporal" in loaded["sensing"]
    assert "range_profile_by_pair" not in loaded["sensing"]
    assert "range_axis_m" not in loaded["sensing"]


def test_selective_range_doppler_load_translates_alias_to_typed_product_and_axes() -> None:
    """Dashboard aliases must load only usable v4 products and exact coordinates."""
    range_axis = np.array([0.0, 0.3, 1.6], dtype=np.float64)
    doppler_axis = np.array([-11.0, 4.0], dtype=np.float64)
    power = np.array([[1.0, 3.0, 7.0], [2.0, 5.0, 9.0]], dtype=np.float32)
    frame = replace(
        build_standard_mpc_frame(frame_idx=3),
        sensing=build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": range_axis,
                "doppler_axis_hz": doppler_axis,
                "range_doppler_power_by_pair": power[np.newaxis, ...],
            },
            scene_frame_id=3,
            sample_time_s=3.0,
        ),
    )
    provider = _CompleteFrameOnlyProvider(frame)
    visualizer = SimpleNamespace(
        cache_service=SimpleNamespace(get_frame=lambda _step: None),
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=None,
    )
    service = SensingService(visualizer)

    loaded = service._load_sensing_candidate(3, ("range_doppler_map",))

    assert loaded is not None
    sensing = loaded["sensing"]
    np.testing.assert_array_equal(sensing["range_doppler_power_by_pair"], power[None])
    np.testing.assert_array_equal(sensing["range_axis_m"], range_axis)
    np.testing.assert_array_equal(sensing["doppler_axis_hz"], doppler_axis)
    assert "range_doppler_map" not in sensing
    assert [request.sensing_products for request in provider.requests] == [
        frozenset({"products"}),
        {
            "range_doppler_power_by_pair",
            "pair_axis",
            "range_axis_m",
            "doppler_axis_hz",
        },
    ]
    result = service.compute_range_doppler(
        frame=loaded,
        sampling_frequency=1.0,
        prf_hz=1.0,
    )
    assert result is not None
    np.testing.assert_array_equal(result.power_linear, power)
    np.testing.assert_array_equal(result.range_axis_m, range_axis)
    np.testing.assert_array_equal(result.doppler_axis_hz, doppler_axis)


def test_replay_metadata_index_loads_arrays_only_for_selected_event_and_reuses_them() -> None:
    def product_frame(storage_id: int, bound_scene_id: int) -> StandardMPCFrame:
        return replace(
            build_standard_mpc_frame(frame_idx=storage_id),
            sensing=build_visualizer_sensing_payload(
                {
                    "range_axis_m": np.array([0.0, 0.5, 1.0], dtype=np.float64),
                    "doppler_axis_hz": np.array([-2.0, 2.0], dtype=np.float64),
                    "range_doppler_power_by_pair": np.full((1, 2, 3), storage_id, dtype=np.float32),
                },
                scene_frame_id=bound_scene_id,
                sample_time_s=float(bound_scene_id),
            ),
        )

    provider = _CountingMultiFrameProvider(
        {
            10: product_frame(10, 5),
            20: product_frame(20, 6),
        }
    )
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        )
    )
    scene = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": {},
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }

    first = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
    )
    first_request_count = len(provider.requests)
    second = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
    )

    assert first is not None and second is not None
    assert first.storage_frame_id == second.storage_frame_id == 10
    assert first.replay_event is not None and second.replay_event is not None
    assert first.replay_reason is None and second.replay_reason is None
    assert first.replay_event.manifest_id == second.replay_event.manifest_id
    assert first.replay_event.key.reference_time_s == first.reference_time_s == 5.0
    assert first.replay_event.scene_binding.storage_frame_id == first.storage_frame_id
    assert first.replay_event.scene_binding.bound_scene_frame_id == first.bound_scene_frame_id
    assert first.replay_event.scene_binding.current_scene_frame_id == first.current_scene_frame_id
    assert first.replay_event.scene_binding.fresh is True
    assert first.replay_event.preflight.disposition == "ready"
    assert first.replay_event.pair_pose.tx_position_m == (1.0, 2.0, 3.0)
    assert first.replay_event.pair_pose.rx_position_m == (4.0, 5.0, 6.0)
    assert (
        first.display_frame["_sensing_display_provenance"]["replay_event_id"]
        == first.replay_event.manifest_id
    )
    assert len(provider.requests) == first_request_count
    dense_requests = [
        (step, request)
        for step, request in provider.requests
        if "range_doppler_power_by_pair" in request.sensing_products
    ]
    assert len(dense_requests) == 1
    assert dense_requests[0][0] == 10
    assert all(
        "range_doppler_power_by_pair" not in request.sensing_products
        for step, request in provider.requests
        if step == 20
    )
    request_count_before_inspection = len(provider.requests)
    unavailable = service.load_detection_stage_bundle(first)
    assert unavailable.disposition is InspectionDisposition.UNAVAILABLE
    assert unavailable.reason == "detector_stages_not_recorded"
    assert len(provider.requests) == request_count_before_inspection


@pytest.mark.parametrize("projection_fails", [False, True])
def test_empty_metadata_projection_does_not_reload_complete_frame(projection_fails) -> None:
    """Only unavailable projections require the complete-frame fallback."""
    frame = build_standard_mpc_frame(frame_idx=5)
    provider = _CountingMultiFrameProvider({5: frame})
    if projection_fails:

        def unavailable(*_args):
            raise OSError("projection unavailable")

        provider.load_frame_projection = unavailable
    complete_loads = []

    def load_frame(step):
        complete_loads.append(step)
        return frame

    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: {"sensing": {}}),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=SimpleNamespace(load_frame=load_frame),
        )
    )
    assert service._load_sensing_metadata_candidate(5) is None
    assert service._load_sensing_metadata_candidate(5) is None
    assert complete_loads == ([5] if projection_fails else [])
    assert len(provider.requests) == (0 if projection_fails else 1)


@pytest.mark.parametrize("batch_error", [None, OSError, ValueError, UnsupportedSensingSchemaError])
def test_metadata_index_uses_batch_and_preserves_validation_failures(batch_error) -> None:
    provider = _CountingMultiFrameProvider(
        {step: build_standard_mpc_frame(frame_idx=step) for step in (3, 4, 5)}
    )
    batches, closed = [], []

    def batch_loader(steps, request):
        batches.append(tuple(steps))
        assert request.sensing_products == frozenset({"products"})
        try:
            yield project_standard_mpc_frame(provider.frames[3], request)
            if batch_error:
                raise batch_error("invalid or unavailable metadata")
            for step in (4, 5):
                yield project_standard_mpc_frame(provider.frames[step], request)
        finally:
            closed.append(True)

    provider.iter_frame_projections = batch_loader
    service = SensingService(
        SimpleNamespace(
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
            cache_service=None,
        )
    )
    storage_ids = (3, 4, 5)
    token = service._refresh_metadata_index_token(storage_ids)
    if batch_error in (ValueError, UnsupportedSensingSchemaError):
        with pytest.raises(batch_error):
            service._indexed_sensing_results(storage_ids, token)
        assert service._sensing_result_index is None
    else:
        assert service._indexed_sensing_results(storage_ids, token) == ()
        assert service._indexed_sensing_results(storage_ids, token) == ()
        assert [step for step, _ in provider.requests] == ([4, 5] if batch_error else [])
    assert batches == [storage_ids]
    assert closed == [True]


def test_metadata_index_does_not_prefetch_complete_cached_sensing_frame() -> None:
    sensing = build_visualizer_sensing_payload({}, scene_frame_id=5, sample_time_s=5.0)
    provider = _CountingMultiFrameProvider({5: build_standard_mpc_frame(frame_idx=5)})

    def forbid_batch(*args):
        raise AssertionError("Cached sensing metadata must not be read again")

    provider.iter_frame_projections = forbid_batch
    service = SensingService(
        SimpleNamespace(
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
            cache_service=SimpleNamespace(get_frame=lambda step: {"sensing": sensing}),
        )
    )
    storage_ids = (5,)
    token = service._refresh_metadata_index_token(storage_ids)
    rows = service._indexed_sensing_results(storage_ids, token)
    assert len(rows) == 1
    assert rows[0][0] == 5
    assert provider.requests == []


def test_replay_metadata_index_projects_cached_full_frame_without_dense_arrays() -> None:
    """A full visualizer cache hit must not become a dense metadata-index entry."""
    dense = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    nested_array = np.arange(2048, dtype=np.float64)
    nested_dense_list = list(range(2048))
    sensing = build_visualizer_sensing_payload(
        {
            "range_axis_m": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "doppler_axis_hz": np.array([-2.0, 2.0], dtype=np.float64),
            "range_doppler_power_by_pair": dense,
            "detections": [{"range_m": 1.0, "track_id": 7}],
        },
        scene_frame_id=5,
        sample_time_s=5.0,
    )
    sensing["opaque_debug"] = {
        "nested": {
            "samples": nested_array,
            "serialized_samples": nested_dense_list,
        }
    }
    sensing["config_resolved"]["opaque_dense_list"] = nested_dense_list
    sensing["config"]["opaque_dense_list"] = nested_dense_list
    cached = {
        "_source": {"frame_idx": 10, "timestamp": 5.0},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": sensing,
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }
    service = SensingService(
        SimpleNamespace(
            cache_service=_frame_cache({10: cached}),
            frame_loader=None,
            frame_source=None,
        )
    )
    scene = {**cached, "sensing": {}}

    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
        candidate_storage_ids=(10,),
    )
    metadata = service._load_sensing_metadata_candidate(10)

    assert selection is not None
    assert selection.storage_frame_id == 10
    np.testing.assert_array_equal(
        selection.product_frame["sensing"]["range_doppler_power_by_pair"],
        dense,
    )
    assert metadata is not None and set(metadata) == {"sensing"}
    assert "products" in metadata["sensing"]
    assert "range_doppler_power_by_pair" not in metadata["sensing"]
    assert "range_axis_m" not in metadata["sensing"]
    assert "doppler_axis_hz" not in metadata["sensing"]
    assert "config" not in metadata["sensing"]
    assert "config_resolved" not in metadata["sensing"]
    assert "detections" not in metadata["sensing"]
    assert "opaque_debug" not in metadata["sensing"]
    assert service._declares_any_sensing_product(metadata, ("detections",))

    def nested_values(value):
        yield value
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from nested_values(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from nested_values(nested)

    retained = tuple(nested_values(metadata))
    assert not any(isinstance(value, np.ndarray) for value in retained)
    assert not any(isinstance(value, list) and len(value) > 16 for value in retained)
    assert service.replay_cache_state() == SensingReplayCacheState(
        metadata_entries=1,
        selected_product_entries=1,
        selected_product_capacity=2,
        detection_stage_entries=0,
        detection_stage_capacity=2,
        validated_event_entries=1,
    )


def test_replay_resolution_retries_without_mixing_provider_generations() -> None:
    """A generation switch during a dense read discards and retries that attempt."""
    storage_id = 10
    scene_id = 5
    first_frame = replace(
        build_standard_mpc_frame(frame_idx=storage_id),
        sensing=_detector_stage_sensing(
            scene_frame_id=scene_id,
            sample_time_s=float(scene_id),
            range_doppler_fill=1.0,
        ),
    )
    second_frame = replace(
        build_standard_mpc_frame(frame_idx=storage_id),
        sensing=_detector_stage_sensing(
            scene_frame_id=scene_id,
            sample_time_s=float(scene_id),
            range_doppler_fill=9.0,
        ),
    )
    provider = _GenerationSwitchingProvider(
        {storage_id: first_frame},
        {storage_id: second_frame},
    )
    telemetry = SensingReplayTelemetry()
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        ),
        replay_telemetry=telemetry,
    )
    scene = {
        "_source": {"frame_idx": scene_id, "timestamp": float(scene_id)},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": {},
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }

    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=scene_id,
        storage_frame_id=scene_id,
        product_names=("range_doppler_map",),
        candidate_storage_ids=(storage_id,),
    )

    assert selection is not None and selection.replay_event is not None
    assert provider.switch_count == 1
    np.testing.assert_array_equal(
        selection.product_frame["sensing"]["range_doppler_by_pair"],
        np.full((1, 2, 3), 9.0, dtype=np.complex64),
    )
    dense_requests = [
        request
        for _, request in provider.requests
        if "range_doppler_by_pair" in request.sensing_products
    ]
    assert len(dense_requests) == 2
    assert service.replay_telemetry_snapshot().selected_dense_loads == 2
    assert service.replay_cache_state().selected_product_entries == 1

    bundle = service.load_detection_stage_bundle(selection)
    assert bundle.disposition is InspectionDisposition.AVAILABLE
    stage_request_count = len(provider.requests)
    assert service.replay_cache_state().detection_stage_entries == 1

    provider.generation_id = "generation-c"
    stale_bundle = service.load_detection_stage_bundle(selection)

    assert stale_bundle.disposition is InspectionDisposition.UNAVAILABLE
    assert stale_bundle.reason == "sensing_inventory_changed_since_selection"
    assert len(provider.requests) == stage_request_count
    assert service.replay_cache_state().selected_product_entries == 0
    assert service.replay_cache_state().detection_stage_entries == 0


def test_scene_layers_carry_the_selected_canonical_replay_identity() -> None:
    scene = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "range_m": 1.0,
                        "x_m": 1.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                    }
                ]
            },
            scene_frame_id=5,
            sample_time_s=5.0,
        ),
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))
    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("detections",),
    )

    assert selection is not None and selection.replay_event is not None
    provenance = selection.display_frame["_sensing_display_provenance"]
    expected_identity = {
        **selection.replay_event.key.to_manifest(),
        "manifest_id": selection.replay_event.manifest_id,
        "scene_binding": selection.replay_event.scene_binding.to_manifest(),
    }
    assert provenance["replay_event_identity"] == expected_identity

    layers = service.build_scene_overlays(selection.display_frame, step=5)

    assert layers
    for layer in layers:
        assert layer.metadata["replay_event_identity"] == expected_identity
        render_metadata = layer.to_render_object().metadata
        assert render_metadata["type"] == "sensing_overlay"
        assert render_metadata["replay_event_identity"] == expected_identity


def test_detection_stage_bundle_selectively_loads_once_and_caches_by_replay_event() -> None:
    sensing = _detector_stage_sensing(scene_frame_id=5)
    provider = _CountingMultiFrameProvider(
        {
            10: replace(
                build_standard_mpc_frame(frame_idx=10),
                sensing=sensing,
            )
        }
    )
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        )
    )
    scene = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": {},
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }

    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
    )

    assert selection is not None and selection.replay_event is not None
    stage_data_keys = {
        data_key
        for data_key, descriptor in sensing["products"].items()
        if descriptor["kind"] in DETECTION_STAGE_KINDS
    }
    assert stage_data_keys
    assert stage_data_keys.isdisjoint(selection.product_frame["sensing"])
    request_count_before_stage_load = len(provider.requests)

    first = service.load_detection_stage_bundle(selection)
    second = service.load_detection_stage_bundle(selection)

    assert first is second
    assert first.disposition is InspectionDisposition.AVAILABLE
    assert len(first.products) == len(DETECTION_STAGE_KINDS)
    assert first.event is not None
    assert first.event.processing_run_id == selection.replay_event.key.processing_run_id
    assert len(provider.requests) == request_count_before_stage_load + 1
    stage_requests = [
        (storage_id, request)
        for storage_id, request in provider.requests
        if stage_data_keys.issubset(request.sensing_products)
    ]
    assert len(stage_requests) == 1
    assert stage_requests[0][0] == selection.storage_frame_id == 10


def test_pinned_seek_reuses_stage_cache_by_scientific_event_identity() -> None:
    sensing = _detector_stage_sensing(scene_frame_id=5)
    provider = _CountingMultiFrameProvider(
        {
            10: replace(
                build_standard_mpc_frame(frame_idx=10),
                sensing=sensing,
            )
        }
    )
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        )
    )
    bound_scene = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": {},
        "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }
    later_scene = {
        **bound_scene,
        "_source": {"timestamp": 6.0},
    }

    exact = service.resolve_display_sensing_frame(
        bound_scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
        candidate_storage_ids=(10,),
    )
    assert exact is not None and exact.replay_event is not None
    exact_bundle = service.load_detection_stage_bundle(exact)
    request_count_after_stage_load = len(provider.requests)

    pinned = service.resolve_display_sensing_frame(
        later_scene,
        scene_frame_id=6,
        storage_frame_id=6,
        product_names=("range_doppler_map",),
        pin_freshest=True,
        candidate_storage_ids=(10,),
        load_scene_frame=lambda _scene_id: bound_scene,
    )

    assert pinned is not None and pinned.replay_event is not None
    assert pinned.stale_for_current_scene is True
    assert pinned.replay_event.manifest_id == exact.replay_event.manifest_id
    assert pinned.replay_event.view_manifest_id != exact.replay_event.view_manifest_id
    assert service.load_detection_stage_bundle(pinned) is exact_bundle
    assert len(provider.requests) == request_count_after_stage_load


def test_replay_telemetry_proves_indexed_forward_reverse_load_invariants() -> None:
    """Cold, warm, forward, and reverse seeks retain data with bounded reads."""

    def scene(scene_id: int, time_s: float) -> dict:
        return {
            "_source": {"frame_idx": scene_id, "timestamp": time_s},
            "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
            "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
            "sensing": {},
            "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
            "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        }

    sensing_a = _detector_stage_sensing(
        scene_frame_id=5,
        sample_time_s=5.0,
        range_doppler_fill=1.25,
    )
    sensing_b = _detector_stage_sensing(
        scene_frame_id=6,
        sample_time_s=6.0,
        range_doppler_fill=2.5,
    )
    provider = _CountingMultiFrameProvider(
        {
            10: replace(build_standard_mpc_frame(frame_idx=10), sensing=sensing_a),
            20: replace(build_standard_mpc_frame(frame_idx=20), sensing=sensing_b),
        }
    )
    telemetry = SensingReplayTelemetry()
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        ),
        replay_telemetry=telemetry,
    )
    candidates = (10, 20)

    first = service.resolve_display_sensing_frame(
        scene(5, 5.0),
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
        candidate_storage_ids=candidates,
    )
    assert first is not None and first.replay_event is not None
    first_bundle = service.load_detection_stage_bundle(first)
    assert first_bundle.disposition is InspectionDisposition.AVAILABLE
    first_request_count = len(provider.requests)
    assert first_request_count == len(candidates) + 2
    assert service.replay_telemetry_snapshot() == SensingReplayTelemetrySnapshot(
        metadata_probes=len(candidates),
        selected_dense_loads=1,
        replay_builds=1,
        stage_loads=1,
        index_builds=1,
    )

    warm = service.resolve_display_sensing_frame(
        scene(5, 5.0),
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
        candidate_storage_ids=candidates,
    )
    assert warm is not None and warm.replay_event is not None
    assert service.load_detection_stage_bundle(warm) is first_bundle
    assert len(provider.requests) == first_request_count

    forward = service.resolve_display_sensing_frame(
        scene(6, 6.0),
        scene_frame_id=6,
        storage_frame_id=6,
        product_names=("range_doppler_map",),
        candidate_storage_ids=candidates,
    )
    assert forward is not None and forward.replay_event is not None
    forward_bundle = service.load_detection_stage_bundle(forward)
    assert forward_bundle.disposition is InspectionDisposition.AVAILABLE
    assert len(provider.requests) == first_request_count + 2

    before_reverse = len(provider.requests)
    reverse = service.resolve_display_sensing_frame(
        scene(5, 5.0),
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
        candidate_storage_ids=candidates,
    )
    assert reverse is not None and reverse.replay_event is not None
    reverse_bundle = service.load_detection_stage_bundle(reverse)
    assert reverse_bundle is first_bundle
    assert len(provider.requests) == before_reverse

    assert reverse.replay_event.manifest_id == first.replay_event.manifest_id
    np.testing.assert_array_equal(
        reverse.product_frame["sensing"]["range_doppler_by_pair"],
        first.product_frame["sensing"]["range_doppler_by_pair"],
    )
    for reverse_product, first_product in zip(
        reverse_bundle.products, first_bundle.products, strict=True
    ):
        assert reverse_product.data_key == first_product.data_key
        np.testing.assert_array_equal(reverse_product.values, first_product.values)
    assert len(reverse_bundle.products) == len(DETECTION_STAGE_KINDS)

    snapshot = service.replay_telemetry_snapshot()
    assert snapshot == SensingReplayTelemetrySnapshot(
        metadata_probes=2,
        # Warm selection now reuses the result index without revisiting each
        # scene row through the metadata loader.
        metadata_cache_hits=0,
        selected_dense_loads=2,
        selected_dense_cache_hits=2,
        replay_builds=2,
        replay_cache_hits=2,
        stage_loads=2,
        stage_cache_hits=2,
        index_builds=1,
        index_memory_hits=3,
    )
    service.reset_replay_telemetry()
    assert service.replay_telemetry_snapshot() == SensingReplayTelemetrySnapshot()


def test_replay_dense_caches_are_bounded_lrus_with_warm_recent_selection() -> None:
    """Long playback retains only two recent dense results and stage bundles."""

    def scene(scene_id: int) -> dict:
        time_s = float(scene_id)
        return {
            "_source": {"frame_idx": scene_id, "timestamp": time_s},
            "rx_orientations": np.array([[0.4, 0.5, 0.6]], dtype=np.float64),
            "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
            "sensing": {},
            "tx_orientations": np.array([[0.1, 0.2, 0.3]], dtype=np.float64),
            "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        }

    scene_ids = tuple(range(5, 10))
    storage_ids = tuple(range(10, 60, 10))
    provider = _CountingMultiFrameProvider(
        {
            storage_id: replace(
                build_standard_mpc_frame(frame_idx=storage_id),
                sensing=_detector_stage_sensing(
                    scene_frame_id=scene_id,
                    sample_time_s=float(scene_id),
                    range_doppler_fill=float(scene_id),
                ),
            )
            for scene_id, storage_id in zip(scene_ids, storage_ids, strict=True)
        }
    )
    telemetry = SensingReplayTelemetry()
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        ),
        replay_telemetry=telemetry,
    )

    selections = []
    for scene_id in scene_ids:
        selection = service.resolve_display_sensing_frame(
            scene(scene_id),
            scene_frame_id=scene_id,
            storage_frame_id=scene_id,
            product_names=("range_doppler_map",),
            candidate_storage_ids=storage_ids,
        )
        assert selection is not None and selection.replay_event is not None
        assert service.load_detection_stage_bundle(selection).disposition is (
            InspectionDisposition.AVAILABLE
        )
        selections.append(selection)
        state = service.replay_cache_state()
        assert state.selected_product_entries <= state.selected_product_capacity == 2
        assert state.detection_stage_entries <= state.detection_stage_capacity == 2

    assert service.replay_cache_state() == SensingReplayCacheState(
        metadata_entries=len(storage_ids),
        selected_product_entries=2,
        selected_product_capacity=2,
        detection_stage_entries=2,
        detection_stage_capacity=2,
        validated_event_entries=len(storage_ids),
    )

    requests_before_warm = len(provider.requests)
    loads_before_warm = service.replay_telemetry_snapshot()
    warm = service.resolve_display_sensing_frame(
        scene(scene_ids[-1]),
        scene_frame_id=scene_ids[-1],
        storage_frame_id=scene_ids[-1],
        product_names=("range_doppler_map",),
        candidate_storage_ids=storage_ids,
    )
    assert warm is not None
    assert service.load_detection_stage_bundle(warm) is service.load_detection_stage_bundle(
        selections[-1]
    )
    assert len(provider.requests) == requests_before_warm
    loads_after_warm = service.replay_telemetry_snapshot()
    assert loads_after_warm.selected_dense_loads == loads_before_warm.selected_dense_loads
    assert loads_after_warm.selected_dense_cache_hits == (
        loads_before_warm.selected_dense_cache_hits + 1
    )
    assert loads_after_warm.stage_loads == loads_before_warm.stage_loads
    assert loads_after_warm.stage_cache_hits == loads_before_warm.stage_cache_hits + 2

    requests_before_evicted = len(provider.requests)
    loads_before_evicted = service.replay_telemetry_snapshot()
    evicted = service.resolve_display_sensing_frame(
        scene(scene_ids[0]),
        scene_frame_id=scene_ids[0],
        storage_frame_id=scene_ids[0],
        product_names=("range_doppler_map",),
        candidate_storage_ids=storage_ids,
    )
    assert evicted is not None
    assert service.load_detection_stage_bundle(evicted).disposition is (
        InspectionDisposition.AVAILABLE
    )
    assert len(provider.requests) == requests_before_evicted + 2
    loads_after_evicted = service.replay_telemetry_snapshot()
    assert loads_after_evicted.selected_dense_loads == (
        loads_before_evicted.selected_dense_loads + 1
    )
    assert loads_after_evicted.stage_loads == loads_before_evicted.stage_loads + 1
    assert service.replay_cache_state().selected_product_entries == 2
    assert service.replay_cache_state().detection_stage_entries == 2


def test_replay_telemetry_is_disabled_by_default() -> None:
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))

    service._replay_telemetry.record("metadata_probes")

    assert service.replay_telemetry_snapshot() == SensingReplayTelemetrySnapshot()


def test_detection_stage_bundle_rejects_projection_from_different_replay_event() -> None:
    provider = _CountingMultiFrameProvider(
        {
            10: replace(
                build_standard_mpc_frame(frame_idx=10),
                sensing=_detector_stage_sensing(scene_frame_id=5),
            )
        }
    )
    service = SensingService(
        SimpleNamespace(
            cache_service=SimpleNamespace(get_frame=lambda _step: None),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        )
    )
    scene = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.zeros((1, 3), dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": {},
        "tx_orientations": np.zeros((1, 3), dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }
    selection = service.resolve_display_sensing_frame(
        scene,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_doppler_map",),
    )
    assert selection is not None and selection.replay_event is not None

    provider.frames[10] = replace(
        build_standard_mpc_frame(frame_idx=10),
        sensing=_detector_stage_sensing(scene_frame_id=6),
    )
    bundle = service.load_detection_stage_bundle(selection)

    assert bundle.disposition is InspectionDisposition.INVALID
    assert bundle.reason is not None
    assert "does not match ReplayEvent" in bundle.reason


def test_compute_range_profile_uses_exact_nonuniform_axis_and_power_linear() -> None:
    range_axis = np.array([0.0, 0.15, 0.9, 2.75], dtype=np.float64)
    profile = np.array([[0.25, 1.0, 4.0, 9.0]], dtype=np.float32)
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {
                    "range_mode": "monostatic",
                    "range_resolution_m": 99.0,
                },
                "range_axis_m": range_axis,
                "range_profile_by_pair": profile,
            }
        )
    }

    service = SensingService(SimpleNamespace())
    result = service.compute_range_profile(frame, sampling_frequency=1.0, fft_size=999)

    assert result is not None
    np.testing.assert_array_equal(result.range_axis_m, range_axis)
    np.testing.assert_array_equal(result.power_linear, profile[0])
    assert result.metadata is not None
    assert result.metadata["representation"] == "power_linear"
    assert result.metadata["data_key"] == "range_profile_by_pair"


def test_compute_range_profile_raw_variant_does_not_fall_back_to_primary() -> None:
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": np.array([0.0, 0.4, 1.2], dtype=np.float64),
                "range_profile_by_pair": np.ones((1, 3), dtype=np.float32),
            }
        )
    }

    service = SensingService(SimpleNamespace())
    with pytest.raises(RuntimeError, match="explicit variant 'raw'"):
        service.compute_range_profile(
            frame,
            sampling_frequency=1e9,
            fft_size=8,
            variant="raw",
        )


def test_compute_range_doppler_uses_exact_nonuniform_axes_for_complex_voltage() -> None:
    range_axis = np.array([0.0, 0.2, 1.3, 4.8], dtype=np.float64)
    doppler_axis = np.array([-125.0, -17.0, 41.0], dtype=np.float64)
    cube = np.array(
        [
            [1.0 + 2.0j, 2.0 + 0.0j, 0.0 + 3.0j, 4.0 + 0.0j],
            [2.0 + 1.0j, 1.0 + 0.0j, 3.0 + 4.0j, 0.0 + 2.0j],
            [0.5 + 0.5j, 2.0 + 2.0j, 1.0 + 1.0j, 5.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {
                    "range_mode": "bistatic",
                    "range_resolution_m": 77.0,
                    "prf_hz": 3.0,
                },
                "range_axis_m": range_axis,
                "doppler_axis_hz": doppler_axis,
                "range_doppler_by_pair": cube[np.newaxis, ...],
            }
        )
    }
    service = SensingService(SimpleNamespace())
    result = service.compute_range_doppler(
        frame=frame,
        sampling_frequency=1.0,
        prf_hz=2.0,
        fft_size_range=999,
        fft_size_doppler=999,
    )

    assert result is not None
    np.testing.assert_array_equal(result.range_axis_m, range_axis)
    np.testing.assert_array_equal(result.doppler_axis_hz, doppler_axis)
    np.testing.assert_array_equal(result.cube, cube)
    np.testing.assert_allclose(result.power_linear, np.abs(cube) ** 2)
    assert result.range_axis_label == "Path Length (m)"
    assert result.metadata is not None
    assert result.metadata["data_key"] == "range_doppler_by_pair"
    assert result.metadata["representation"] == "complex_voltage"
    assert result.metadata["detection_velocity_semantics"] == "monostatic_equivalent_velocity"


def test_compute_range_doppler_preserves_power_linear_without_squaring() -> None:
    range_axis = np.array([0.0, 0.7, 2.2], dtype=np.float64)
    doppler_axis = np.array([-9.0, 2.0], dtype=np.float64)
    power = np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]], dtype=np.float32)
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": range_axis,
                "doppler_axis_hz": doppler_axis,
                "range_doppler_power_by_pair": power[np.newaxis, ...],
            }
        )
    }

    result = SensingService(SimpleNamespace()).compute_range_doppler(
        frame=frame,
        sampling_frequency=1e9,
        prf_hz=1000.0,
    )

    assert result is not None
    np.testing.assert_array_equal(result.power_linear, power)
    np.testing.assert_array_equal(result.range_axis_m, range_axis)
    np.testing.assert_array_equal(result.doppler_axis_hz, doppler_axis)
    assert result.cube is None
    assert result.metadata is not None
    assert result.metadata["representation"] == "power_linear"


def test_compute_range_doppler_raw_variant_does_not_fall_back_to_primary() -> None:
    frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": np.array([0.0, 1.0], dtype=np.float64),
                "doppler_axis_hz": np.array([-1.0, 1.0], dtype=np.float64),
                "range_doppler_by_pair": np.ones((1, 2, 2), dtype=np.complex64),
            }
        )
    }

    with pytest.raises(RuntimeError, match="explicit variant 'raw'"):
        SensingService(SimpleNamespace()).compute_range_doppler(
            frame=frame,
            sampling_frequency=1e9,
            prf_hz=1000.0,
            variant="raw",
        )


def test_compute_range_doppler_selects_explicit_raw_descriptor_not_data_key() -> None:
    range_axis = np.array([0.0, 0.35, 1.4], dtype=np.float64)
    doppler_axis = np.array([-8.0, 3.5], dtype=np.float64)
    primary = np.full((1, 2, 3), 2.0, dtype=np.float32)
    raw = np.full((1, 2, 3), 7.0, dtype=np.float32)
    sensing = build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "monostatic"},
            "range_axis_m": range_axis,
            "doppler_axis_hz": doppler_axis,
            "range_doppler_power_by_pair": primary,
        }
    )
    _add_range_doppler_variant(
        sensing,
        data_key="diagnostic_unfiltered_voltage_power",
        values=raw,
        variant="raw",
        representation="power_linear",
    )
    service = SensingService(SimpleNamespace())

    primary_result = service.compute_range_doppler(
        frame={"sensing": sensing},
        sampling_frequency=1.0,
        prf_hz=1.0,
        variant="primary",
    )
    raw_result = service.compute_range_doppler(
        frame={"sensing": sensing},
        sampling_frequency=1.0,
        prf_hz=1.0,
        variant="raw",
    )

    assert primary_result is not None and raw_result is not None
    np.testing.assert_array_equal(primary_result.power_linear, primary[0])
    np.testing.assert_array_equal(raw_result.power_linear, raw[0])
    assert raw_result.metadata is not None
    assert raw_result.metadata["variant"] == "raw"
    assert raw_result.metadata["data_key"] == "diagnostic_unfiltered_voltage_power"


def test_compute_range_doppler_rejects_ambiguous_primary_descriptors() -> None:
    sensing = build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "monostatic"},
            "range_axis_m": np.array([0.0, 0.4, 1.2], dtype=np.float64),
            "doppler_axis_hz": np.array([-3.0, 5.0], dtype=np.float64),
            "range_doppler_power_by_pair": np.ones((1, 2, 3), dtype=np.float32),
        }
    )
    _add_range_doppler_variant(
        sensing,
        data_key="second_primary_product",
        values=np.full((1, 2, 3), 4.0, dtype=np.float32),
        variant="primary",
        representation="power_linear",
    )

    with pytest.raises(ValueError, match="Ambiguous sensing products"):
        SensingService(SimpleNamespace()).compute_range_doppler(
            frame={"sensing": sensing},
            sampling_frequency=1.0,
            prf_hz=1.0,
            variant="primary",
        )


def test_compute_range_doppler_rejects_missing_range_coordinate_semantics() -> None:
    sensing = build_visualizer_sensing_payload()
    sensing["range_axis_m"] = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    sensing["doppler_axis_hz"] = np.array([-4.0, 6.0], dtype=np.float64)
    _add_range_doppler_variant(
        sensing,
        data_key="semantically_unbound_rd",
        values=np.ones((1, 2, 3), dtype=np.float32),
        variant="primary",
        representation="power_linear",
        range_semantics="explicit_range_coordinates",
    )

    with pytest.raises(ValueError, match="requires coordinate_semantics"):
        SensingService(SimpleNamespace()).compute_range_doppler(
            frame={"sensing": sensing},
            sampling_frequency=1.0,
            prf_hz=1.0,
        )


def test_product_contract_rejects_product_without_pair_axis() -> None:
    sensing = build_visualizer_sensing_payload({"config": {"range_mode": "monostatic"}})
    sensing["range_axis_m"] = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    sensing["range_profile_by_pair"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pair_id = sensing["temporal"]["acquisition"]["pair_id"]
    descriptor = build_product_descriptor(
        sensing,
        data_key="range_profile_by_pair",
        kind="range_profile",
        axes=(
            dense_axis(
                "range",
                size=3,
                unit="m",
                coordinate_key="range_axis_m",
            ),
        ),
        representation="power_linear",
        unit="1",
        scale_semantics="relative_unitless",
        pair_ids=(pair_id,),
    )
    with pytest.raises(ValueError, match="explicit pair axis"):
        attach_product_descriptors(sensing, (descriptor,))


def test_compute_range_profile_rejects_multi_pair_product() -> None:
    sensing = build_visualizer_sensing_payload({"config": {"range_mode": "monostatic"}})
    first_pair = dict(sensing["pair_axis"][0])
    second_pair_id = canonical_sensing_hash(
        {"tx_sensor_id": "visualizer-tx-2", "rx_sensor_id": "visualizer-rx-2"}
    )
    sensing["pair_axis"] = [
        first_pair,
        {
            "pair_index": 1,
            "pair_id": second_pair_id,
            "tx_sensor_id": "visualizer-tx-2",
            "rx_sensor_id": "visualizer-rx-2",
        },
    ]
    sensing["range_axis_m"] = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    sensing["range_profile_by_pair"] = np.ones((2, 3), dtype=np.float32)
    sensing["products"] = {
        "range_profile_by_pair": {
            "kind": "range_profile",
            "variant": "primary",
            "representation": "power_linear",
            "axes": [
                catalog_axis("pair", size=2, catalog_key="pair_axis"),
                dense_axis(
                    "range",
                    size=3,
                    unit="m",
                    coordinate_key="range_axis_m",
                ),
            ],
            "pair_ids": [first_pair["pair_id"], second_pair_id],
        }
    }

    with pytest.raises(ValueError, match="requires exactly one explicit pair"):
        _require_single_pair_product(
            sensing,
            contract=_RANGE_PROFILE_CONTRACT,
            variant="primary",
        )


@pytest.mark.parametrize(
    "consume",
    [
        pytest.param(
            lambda service, frame: service.compute_cir(frame, sampling_frequency=1e9),
            id="cir",
        ),
        pytest.param(
            lambda service, frame: service.compute_range_profile(frame, sampling_frequency=1e9),
            id="range-profile",
        ),
        pytest.param(
            lambda service, frame: service.compute_range_doppler(
                frame=frame,
                sampling_frequency=1e9,
                prf_hz=1000.0,
            ),
            id="range-doppler",
        ),
        pytest.param(
            lambda service, frame: service.build_scene_overlays(frame, step=0),
            id="scene-overlays",
        ),
        pytest.param(
            lambda service, frame: service.get_detection_overlay(frame),
            id="detection-overlay",
        ),
    ],
)
def test_sensing_service_rejects_pre_v4_payloads(consume) -> None:
    frame = {
        "rx_positions": np.zeros((1, 3), dtype=np.float32),
        "sensing": {
            "schema_version": 2,
            "range_profile": np.ones(4, dtype=np.float32),
            "range_doppler_map": np.ones((2, 4), dtype=np.float32),
            "detections": [{"range_m": 1.0}],
        },
    }
    provider = SimpleNamespace(compute=lambda *_args, **_kwargs: None)
    service = SensingService(SimpleNamespace(), cir_provider=provider)

    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        consume(service, frame)


def test_stored_cir_provider_direct_boundary_rejects_pre_v4_payload() -> None:
    with pytest.raises(
        UnsupportedSensingSchemaError,
        match="Regenerate this sensing dataset",
    ):
        StoredCIRProvider().compute(
            {
                "sensing": {
                    "schema_version": 2,
                    "cir": np.ones(8, dtype=np.complex64),
                }
            },
            sampling_frequency=1e9,
        )


def test_stored_cir_provider_selects_latest_aligned_coherent_sample_time() -> None:
    rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    sensing = _multi_sample_sensing(
        {"slow_time_matrix": rows},
        sample_times_s=[1.0, 1.25, 1.5],
        strategy="coherent_cpi_fft",
    )

    result = StoredCIRProvider().compute(
        {"sensing": sensing},
        sampling_frequency=1e9,
    )

    assert result is not None
    assert result.sample_time_s == pytest.approx(1.5)
    np.testing.assert_array_equal(result.cir, rows[-1])
    assert result.metadata is not None
    assert result.metadata["sample_time_s"] == pytest.approx(1.5)


def test_typed_slow_time_uses_exact_axes_ids_and_named_unitary_combiner() -> None:
    range_axis = np.array([0.0, 0.2, 1.1, 3.6], dtype=np.float64)
    sample_times = [1.0, 1.125, 1.5]
    element_values = np.stack(
        (
            np.full((3, 4), 1.0 + 0.0j, dtype=np.complex128),
            np.full((3, 4), 3.0 + 0.0j, dtype=np.complex128),
        ),
        axis=0,
    )[np.newaxis, ...]
    sensing = _multi_sample_sensing(
        {
            "range_axis_m": range_axis,
            "slow_time_by_pair_rx_element": element_values,
        },
        sample_times_s=sample_times,
        strategy="coherent_cpi_fft",
    )

    batch = extract_timestamped_fast_time_samples(sensing)
    result = StoredCIRProvider().compute(
        {"sensing": sensing},
        sampling_frequency=123.0,
    )

    assert batch.expanded_acquisition is True
    assert batch.sample_times_s == pytest.approx(sample_times)
    np.testing.assert_array_equal(batch.range_axis_m, range_axis)
    assert batch.pair_id == sensing["temporal"]["acquisition"]["pair_id"]
    assert batch.rx_element_identities == ((0, 0, 0), (1, 0, 1))
    np.testing.assert_allclose(batch.samples[-1], 4.0 / np.sqrt(2.0))
    assert result is not None
    np.testing.assert_allclose(result.cir, 4.0 / np.sqrt(2.0))
    assert result.metadata is not None
    assert result.metadata["data_key"] == "slow_time_by_pair_rx_element"
    assert result.metadata["product_id"] == batch.product_id
    np.testing.assert_array_equal(result.metadata["range_axis_m"], range_axis)


def test_fast_time_consumer_rejects_legacy_array_without_typed_descriptor() -> None:
    sensing = build_visualizer_sensing_payload({"cir": np.ones(4, dtype=np.complex128)})

    with pytest.raises(ValueError, match="legacy cir and slow_time_matrix arrays"):
        extract_timestamped_fast_time_samples(sensing)


def test_fast_time_consumer_rejects_multi_pair_typed_slow_time() -> None:
    sensing = build_visualizer_sensing_payload()
    first_pair = dict(sensing["pair_axis"][0])
    sensing["pair_axis"] = [
        first_pair,
        {
            "pair_index": 1,
            "pair_id": "second-pair",
            "tx_sensor_id": "second-tx",
            "rx_sensor_id": "second-rx",
        },
    ]
    sensing["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]
    sensing["range_axis_m"] = np.array([0.0, 0.3], dtype=np.float64)
    sensing["slow_time_axis_s"] = np.array([0.0], dtype=np.float64)
    sensing["slow_time_by_pair_rx_element"] = np.ones((2, 1, 1, 2), dtype=np.complex128)
    sensing["products"] = {
        "slow_time_by_pair_rx_element": {
            "kind": "delay_domain_slow_time",
            "variant": "primary",
            "representation": "complex_voltage",
            "axes": [
                catalog_axis("pair", size=2, catalog_key="pair_axis"),
                catalog_axis("rx_element", size=1, catalog_key="rx_element_axis"),
                dense_axis(
                    "slow_time",
                    size=1,
                    unit="s",
                    coordinate_key="slow_time_axis_s",
                ),
                dense_axis(
                    "range",
                    size=2,
                    unit="m",
                    coordinate_key="range_axis_m",
                    coordinate_semantics="monostatic_range",
                ),
            ],
            "pair_ids": [first_pair["pair_id"], "second-pair"],
            "product_id": "unsupported-multi-pair-fixture",
            "combine": {"policy": "siso_identity", "normalization": "identity"},
        }
    }

    with pytest.raises(ValueError, match="exactly one pair"):
        extract_timestamped_fast_time_samples(sensing)


def test_cir_consumer_expands_pregrouped_measurement_acquisition() -> None:
    rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    sensing = _multi_sample_sensing(
        {"slow_time_matrix": rows},
        sample_times_s=[1.0, 1.25, 1.5],
        strategy="nist_mpc_csv_import",
    )
    sensing["config_resolved"] = {
        "acquisition": {
            "sample_source": "measured_pulse_stream",
            "cpi_length_samples": 3,
            "cpi_hop_samples": 1,
        },
        "processing": {
            "mode": "measurement_import",
            "strategy": "nist_mpc_csv_import",
        },
    }

    batch = extract_timestamped_fast_time_samples(sensing)

    assert batch.expanded_acquisition is True
    assert batch.sample_times_s == pytest.approx((1.0, 1.25, 1.5))
    assert len(batch.samples) == 3
    np.testing.assert_array_equal(batch.samples[0], rows[0])


def test_cir_consumer_expands_reprocessed_coherent_acquisition() -> None:
    rows = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    sensing = _multi_sample_sensing(
        {"slow_time_matrix": rows},
        sample_times_s=[1.0, 1.25, 1.5],
        strategy="shared_sensing_postprocessor",
    )
    for sample in sensing["temporal"]["acquisition"]["samples"]:
        sample["phase_coherent"] = True
    sensing["config_resolved"] = {
        "acquisition": {
            "sample_source": "coherent_one_solve",
            "cpi_length_samples": 3,
            "cpi_hop_samples": 3,
        },
        "processing": {
            "mode": "offline_reprocess",
            "strategy": "shared_sensing_postprocessor",
        },
        "source_processing": {
            "mode": "coherent_cpi",
            "strategy": "coherent_cpi_fft",
        },
    }

    batch = extract_timestamped_fast_time_samples(sensing)

    assert batch.expanded_acquisition is True
    assert batch.sample_times_s == pytest.approx((1.0, 1.25, 1.5))
    assert len(batch.samples) == 3
    np.testing.assert_array_equal(batch.samples[-1], rows[-1])


def test_cir_consumer_uses_exact_resolved_strategy_when_projection_is_absent() -> None:
    sensing = _multi_sample_sensing(
        {"cir": np.ones(4, dtype=np.complex128)},
        sample_times_s=[1.0],
        strategy="sequential_stft",
    )
    sensing.pop("processing_strategy")

    batch = extract_timestamped_fast_time_samples(sensing)

    assert batch.strategy == "sequential_stft"


def test_cir_consumer_never_sources_strategy_from_outer_projection() -> None:
    sensing = _multi_sample_sensing(
        {"cir": np.ones(4, dtype=np.complex128)},
        sample_times_s=[1.0],
        strategy="sequential_stft",
    )
    sensing.pop("config_resolved")

    with pytest.raises(
        ValueError,
        match=r"requires config_resolved\.processing\.strategy",
    ):
        extract_timestamped_fast_time_samples(sensing)


def test_cir_consumer_rejects_strategy_projection_disagreement() -> None:
    sensing = _multi_sample_sensing(
        {"cir": np.ones(4, dtype=np.complex128)},
        sample_times_s=[1.0],
        strategy="sequential_stft",
    )
    sensing["config_resolved"] = {"processing": {"strategy": "coherent_cpi_fft"}}

    with pytest.raises(ValueError, match="disagrees"):
        extract_timestamped_fast_time_samples(sensing)


def test_sensing_service_history_dedupes_replay_and_resets_on_seek() -> None:
    service = SensingService(SimpleNamespace())

    def compute(sample_time_s: float, amplitude: float):
        return service.compute_cir(
            {
                "sensing": build_visualizer_sensing_payload(
                    {
                        "range_axis_m": np.array([0.0, 0.2, 0.75, 1.8]),
                        "cir_by_pair_rx_element": np.full(
                            (1, 1, 4), amplitude, dtype=np.complex128
                        ),
                    },
                    sample_time_s=sample_time_s,
                )
            },
            sampling_frequency=1e9,
        )

    compute(1.0, 1.0)
    compute(2.0, 2.0)
    compute(2.0, 20.0)

    assert [entry.sample_time_s for entry in service.cir_history] == [1.0, 2.0]
    np.testing.assert_array_equal(service.cir_history[-1].cir, np.full(4, 20.0))

    compute(0.5, 5.0)

    assert [entry.sample_time_s for entry in service.cir_history] == [0.5]
    np.testing.assert_array_equal(service.cir_history[-1].cir, np.full(4, 5.0))


def test_pinned_mismatch_uses_bound_pose_but_current_scene_time_for_age() -> None:
    current_scene = {
        "_source": {"timestamp": 35.0},
        "rx_orientations": np.zeros((1, 3), dtype=np.float32),
        "rx_positions": np.array([[20.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": {},
        "tx_orientations": np.zeros((1, 3), dtype=np.float32),
        "tx_positions": np.array([[200.0, 0.0, 0.0]], dtype=np.float32),
    }
    bound_scene = {
        "_source": {"timestamp": 10.0},
        "rx_orientations": np.array([[0.4, 0.2, -0.1]], dtype=np.float32),
        "rx_positions": np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": {},
        "tx_orientations": np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }
    product = {
        "rx_positions": np.array([[300.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]},
            scene_frame_id=10,
            sample_time_s=10.0,
        ),
    }
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))

    selection = service.resolve_display_sensing_frame(
        current_scene,
        scene_frame_id=20,
        storage_frame_id=20,
        product_names=("detections",),
        pin_freshest=True,
        candidate_storage_ids=(30,),
        load_candidate=lambda _storage_id: product,
        load_scene_frame=lambda _scene_id: bound_scene,
    )

    assert selection is not None
    np.testing.assert_array_equal(selection.display_frame["rx_positions"], [[10.0, 0.0, 0.0]])
    assert selection.current_scene_time_s == pytest.approx(35.0)
    assert selection.scene_result_time_offset_s == pytest.approx(25.0)
    assert selection.stale_age_s == pytest.approx(25.0)
    assert selection.replay_event is not None
    assert selection.replay_reason is None
    assert selection.replay_event.scene_binding.stale is True
    assert selection.replay_event.scene_binding.pinned is True
    assert selection.replay_event.preflight.disposition == "ready_with_warnings"
    assert selection.replay_event.pair_pose.tx_position_m == (1.0, 2.0, 3.0)
    assert selection.replay_event.pair_pose.rx_position_m == (10.0, 0.0, 0.0)


def test_pinned_mismatch_never_selects_a_future_result() -> None:
    current_scene = {
        "_source": {"timestamp": 9.0},
        "sensing": {},
    }
    future_product = {
        "sensing": build_visualizer_sensing_payload(
            {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]},
            scene_frame_id=10,
            sample_time_s=10.0,
        )
    }
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))

    selection = service.resolve_display_sensing_frame(
        current_scene,
        scene_frame_id=9,
        storage_frame_id=9,
        product_names=("detections",),
        pin_freshest=True,
        candidate_storage_ids=(30,),
        load_candidate=lambda _storage_id: future_product,
    )

    assert selection is None


def test_pinning_without_physical_scene_time_remains_exact_only() -> None:
    current_scene = {"sensing": {}}
    mismatched_product = {
        "sensing": build_visualizer_sensing_payload(
            {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]},
            scene_frame_id=10,
            sample_time_s=10.0,
        )
    }
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))

    selection = service.resolve_display_sensing_frame(
        current_scene,
        scene_frame_id=9,
        storage_frame_id=9,
        product_names=("detections",),
        pin_freshest=True,
        candidate_storage_ids=(30,),
        load_candidate=lambda _storage_id: mismatched_product,
    )

    assert selection is None


def test_bound_earlier_scene_reports_signed_offset_without_stale_age() -> None:
    scene_frame = {
        "_source": {"timestamp": 9.0},
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": np.arange(8, dtype=np.float64),
                "range_profile_by_pair": np.ones((1, 8), dtype=np.float32),
            },
            scene_frame_id=5,
            sample_time_s=10.0,
        ),
    }
    service = SensingService(SimpleNamespace(cache_service=None, frame_source=None))

    selection = service.resolve_display_sensing_frame(
        scene_frame,
        scene_frame_id=5,
        storage_frame_id=5,
        product_names=("range_profile",),
    )

    assert selection is not None
    assert selection.scene_result_time_offset_s == pytest.approx(-1.0)
    assert selection.stale_age_s is None
    provenance = selection.display_frame["_sensing_display_provenance"]
    assert provenance["scene_result_time_offset_s"] == pytest.approx(-1.0)
    assert provenance["stale_age_s"] is None


def test_display_provenance_does_not_infer_scene_time_from_numeric_id() -> None:
    scene_frame = {
        "sensing": build_visualizer_sensing_payload(
            {
                "config": {"range_mode": "monostatic"},
                "range_axis_m": np.arange(8, dtype=np.float64),
                "range_profile_by_pair": np.ones((1, 8), dtype=np.float32),
            },
            scene_frame_id=5000,
            sample_time_s=2.0,
        )
    }
    selection = SensingService(
        SimpleNamespace(cache_service=None, frame_source=None)
    ).resolve_display_sensing_frame(
        scene_frame,
        scene_frame_id=5000,
        storage_frame_id=5000,
        product_names=("range_profile",),
    )

    assert selection is not None
    assert selection.current_scene_time_s is None
    assert selection.scene_result_time_offset_s is None
    assert selection.stale_age_s is None


def test_recipe_snapshot_mismatch_is_rejected_before_display_selection() -> None:
    sensing = build_visualizer_sensing_payload(
        {
            "range_axis_m": np.arange(4, dtype=np.float64),
            "range_profile_by_pair": np.ones((1, 4), dtype=np.float32),
        },
        scene_frame_id=5,
        sample_time_s=5.0,
    )
    sensing["config_resolved"]["processing_recipe"]["threshold"] = 99.0
    sensing["config"]["processing_recipe"]["threshold"] = 99.0
    scene_frame = {
        "_source": {"timestamp": 5.0},
        "rx_orientations": np.zeros((1, 3), dtype=np.float64),
        "rx_positions": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
        "sensing": sensing,
        "tx_orientations": np.zeros((1, 3), dtype=np.float64),
        "tx_positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    }

    with pytest.raises(ValueError, match="processing_recipe_id must match"):
        SensingService(
            SimpleNamespace(cache_service=None, frame_source=None)
        ).resolve_display_sensing_frame(
            scene_frame,
            scene_frame_id=5,
            storage_frame_id=5,
            product_names=("range_profile",),
        )


def test_sensing_service_rejects_malformed_v4_contract() -> None:
    service = SensingService(SimpleNamespace())
    frame = {
        "sensing": {
            "schema_version": SENSING_SCHEMA_VERSION,
            "range_profile": np.ones(4, dtype=np.float32),
        }
    }

    with pytest.raises(ValueError, match="Invalid sensing contract"):
        service.compute_range_profile(frame, sampling_frequency=1e9)


def test_detection_overlay_uses_typed_frame_pair_pose_index() -> None:
    sensing = build_visualizer_sensing_payload(
        {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]}
    )
    sensing["pair_axis"][0]["frame_pair_index"] = 1
    frame = {
        "rx_positions": np.array(
            [[100.0, 100.0, 0.0], [10.0, 20.0, 1.0]],
            dtype=np.float64,
        ),
        "rx_orientations": np.array(
            [[0.0, 0.0, 0.0], [np.pi / 2.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        "sensing": sensing,
    }

    overlay = SensingService(SimpleNamespace()).get_detection_overlay(frame)

    assert overlay is not None
    positions, _colors, kind = overlay
    assert kind == "points"
    np.testing.assert_allclose(positions, [[10.0, 21.0, 1.0]], atol=1e-7)


@pytest.mark.parametrize(
    ("orientation", "local_position", "expected_world"),
    [
        ([0.0, np.pi / 2.0, 0.0], [1.0, 0.0, 0.0], [10.0, 20.0, 0.0]),
        ([0.0, 0.0, np.pi / 2.0], [0.0, 1.0, 0.0], [10.0, 20.0, 2.0]),
    ],
)
def test_detection_overlay_applies_full_pitch_and_roll_pose(
    orientation: list[float],
    local_position: list[float],
    expected_world: list[float],
) -> None:
    sensing = build_visualizer_sensing_payload(
        {
            "detections": [
                {
                    "x_m": local_position[0],
                    "y_m": local_position[1],
                    "z_m": local_position[2],
                }
            ]
        }
    )
    frame = {
        "rx_positions": np.array([[10.0, 20.0, 1.0]], dtype=np.float64),
        "rx_orientations": np.array([orientation], dtype=np.float64),
        "sensing": sensing,
    }

    overlay = SensingService(SimpleNamespace()).get_detection_overlay(frame)

    assert overlay is not None
    positions, _colors, kind = overlay
    assert kind == "points"
    np.testing.assert_allclose(positions, [expected_world], atol=1e-7)


def test_bistatic_range_only_overlay_fails_closed() -> None:
    sensing = build_visualizer_sensing_payload(
        {
            "config": {"range_mode": "bistatic"},
            "detections": [{"range_m": 12.0}],
        }
    )
    frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        "sensing": sensing,
    }

    with pytest.raises(ValueError, match="Bistatic range-only"):
        SensingService(SimpleNamespace()).get_detection_overlay(frame)


def test_detection_overlay_rejects_pose_arrays_missing_typed_pair_row() -> None:
    sensing = build_visualizer_sensing_payload(
        {"detections": [{"x_m": 1.0, "y_m": 0.0, "z_m": 0.0}]}
    )
    sensing["pair_axis"][0]["frame_pair_index"] = 1
    frame = {
        "rx_positions": np.array([[10.0, 20.0, 1.0]], dtype=np.float64),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        "sensing": sensing,
    }

    with pytest.raises(ValueError, match="outside the rx_positions array"):
        SensingService(SimpleNamespace()).get_detection_overlay(frame)


def test_build_scene_overlays_tracks_include_velocity_arrows() -> None:
    prev_frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 1,
                        "track_confirmed": True,
                        "x_m": 1.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "track_vx_m_s": 1.0,
                        "track_vy_m_s": 0.0,
                        "track_vz_m_s": 0.0,
                        "snr_db": 15.0,
                    }
                ]
            },
            scene_frame_id=63,
            sample_time_s=63.0,
        ),
    }
    frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 1,
                        "track_confirmed": True,
                        "x_m": 2.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "track_vx_m_s": 1.0,
                        "track_vy_m_s": 0.0,
                        "track_vz_m_s": 0.0,
                        "snr_db": 18.0,
                    }
                ]
            },
            scene_frame_id=127,
            sample_time_s=127.0,
        ),
    }
    cache = _frame_cache({63: prev_frame})
    service = SensingService(SimpleNamespace(cache_service=cache))

    layers = service.build_scene_overlays(
        frame,
        step=127,
        mode="tracks",
        history_frames=2,
        show_velocity_arrows=True,
    )

    names = {layer.name for layer in layers}
    assert service.DETECTION_GEOMETRY_NAME not in names
    assert f"{service.TRACK_POINT_PREFIX}1" in names
    assert f"{service.TRACK_TRAIL_PREFIX}1" in names
    assert f"{service.TRACK_ARROW_PREFIX}1" in names


def test_build_scene_overlays_tracks_prefer_tracker_state_positions() -> None:
    prev_frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 1,
                        "track_confirmed": True,
                        "x_m": 1.0,
                        "y_m": 5.0,
                        "z_m": 0.0,
                        "track_x_m": 1.0,
                        "track_y_m": 0.0,
                        "track_z_m": 0.0,
                        "snr_db": 15.0,
                    }
                ]
            },
            scene_frame_id=63,
            sample_time_s=63.0,
        ),
    }
    frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 1,
                        "track_confirmed": True,
                        "x_m": 2.0,
                        "y_m": 5.0,
                        "z_m": 0.0,
                        "track_x_m": 2.0,
                        "track_y_m": 0.0,
                        "track_z_m": 0.0,
                        "snr_db": 18.0,
                    }
                ]
            },
            scene_frame_id=127,
            sample_time_s=127.0,
        ),
    }
    cache = _frame_cache({63: prev_frame})
    service = SensingService(SimpleNamespace(cache_service=cache))

    layers = service.build_scene_overlays(
        frame,
        step=127,
        mode="full",
        history_frames=2,
        show_velocity_arrows=False,
    )

    detection_layer = next(
        layer for layer in layers if layer.name == service.DETECTION_GEOMETRY_NAME
    )
    point_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_POINT_PREFIX}1")
    trail_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_TRAIL_PREFIX}1")

    np.testing.assert_allclose(detection_layer.payload.points, np.array([[2.0, 5.0, 0.0]]))
    np.testing.assert_allclose(point_layer.payload.points, np.array([[2.0, 0.0, 0.0]]))
    assert trail_layer.payload.line_strip is True
    np.testing.assert_allclose(
        trail_layer.payload.points,
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(trail_layer.payload.lines, np.array([[0, 1]]))


def test_build_scene_overlays_tentative_tracks_preserve_identity_color() -> None:
    prev_frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 7,
                        "track_confirmed": False,
                        "x_m": 1.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "track_vx_m_s": 0.5,
                        "track_vy_m_s": 0.0,
                        "track_vz_m_s": 0.0,
                        "snr_db": 8.0,
                    }
                ]
            },
            scene_frame_id=63,
            sample_time_s=63.0,
        ),
    }
    frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 7,
                        "track_confirmed": False,
                        "x_m": 1.5,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "track_vx_m_s": 0.5,
                        "track_vy_m_s": 0.0,
                        "track_vz_m_s": 0.0,
                        "snr_db": 9.0,
                    }
                ]
            },
            scene_frame_id=127,
            sample_time_s=127.0,
        ),
    }
    cache = _frame_cache({63: prev_frame})
    service = SensingService(SimpleNamespace(cache_service=cache))

    layers = service.build_scene_overlays(
        frame,
        step=127,
        mode="tracks",
        history_frames=2,
        show_velocity_arrows=True,
    )

    point_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_POINT_PREFIX}7")
    trail_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_TRAIL_PREFIX}7")
    arrow_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_ARROW_PREFIX}7")

    tentative_rgb = tuple(float(v) for v in service._track_rgba(7, True)[:3])
    assert point_layer.material is not None
    assert trail_layer.material is not None
    assert arrow_layer.material is not None
    assert point_layer.material.base_color[:3] == tentative_rgb
    assert trail_layer.material.base_color[:3] == tentative_rgb
    assert arrow_layer.material.base_color[:3] == tentative_rgb
    assert trail_layer.material.line_width == 2.25
    np.testing.assert_array_equal(trail_layer.payload.lines, np.array([[0, 1]]))
    assert arrow_layer.payload.lines.shape[0] > 0


def test_build_scene_overlays_tracks_can_disable_velocity_arrows() -> None:
    frame = {
        "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "sensing": build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "track_id": 3,
                        "track_confirmed": True,
                        "x_m": 2.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "track_vx_m_s": 1.0,
                        "track_vy_m_s": 0.0,
                        "track_vz_m_s": 0.0,
                        "snr_db": 18.0,
                    }
                ]
            },
            scene_frame_id=127,
            sample_time_s=127.0,
        ),
    }
    service = SensingService(
        SimpleNamespace(cache_service=SimpleNamespace(get_frame=lambda step: None))
    )

    layers = service.build_scene_overlays(
        frame,
        step=127,
        mode="tracks",
        history_frames=2,
        show_velocity_arrows=False,
    )

    names = {layer.name for layer in layers}
    assert f"{service.TRACK_ARROW_PREFIX}3" not in names


def test_build_scene_overlays_full_history_uses_all_available_sensing_frames() -> None:
    frames = {
        10: {
            "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": build_visualizer_sensing_payload(
                {
                    "detections": [
                        {
                            "track_id": 5,
                            "track_confirmed": True,
                            "x_m": 1.0,
                            "y_m": 0.0,
                            "z_m": 0.0,
                            "snr_db": 10.0,
                        }
                    ]
                },
                scene_frame_id=10,
                sample_time_s=10.0,
            ),
        },
        20: {
            "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": build_visualizer_sensing_payload(
                {
                    "detections": [
                        {
                            "track_id": 5,
                            "track_confirmed": True,
                            "x_m": 2.0,
                            "y_m": 0.0,
                            "z_m": 0.0,
                            "snr_db": 11.0,
                        }
                    ]
                },
                scene_frame_id=20,
                sample_time_s=20.0,
            ),
        },
        30: {
            "rx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "rx_orientations": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "sensing": build_visualizer_sensing_payload(
                {
                    "detections": [
                        {
                            "track_id": 5,
                            "track_confirmed": True,
                            "x_m": 3.0,
                            "y_m": 0.0,
                            "z_m": 0.0,
                            "snr_db": 12.0,
                        }
                    ]
                },
                scene_frame_id=30,
                sample_time_s=30.0,
            ),
        },
    }
    standard_frames = {
        step: replace(
            build_standard_mpc_frame(frame_idx=step),
            sensing=frame["sensing"],
        )
        for step, frame in frames.items()
    }
    frame_source = SimpleNamespace(
        list_frames=lambda: [10, 20, 30],
        load_frame=lambda step: standard_frames[step],
    )
    cache = SimpleNamespace(get_frame=lambda step: None)
    service = SensingService(SimpleNamespace(cache_service=cache, frame_source=frame_source))

    layers = service.build_scene_overlays(
        frames[30],
        step=30,
        mode="tracks",
        history_frames=1,
        use_full_history=True,
        show_velocity_arrows=False,
    )

    trail_layer = next(layer for layer in layers if layer.name == f"{service.TRACK_TRAIL_PREFIX}5")
    assert trail_layer.payload is not None
    assert trail_layer.payload.points.shape[0] == 3
    np.testing.assert_array_equal(trail_layer.payload.lines, np.array([[0, 1], [1, 2]]))


def test_track_history_reads_only_selected_results_and_reuses_validated_records(monkeypatch):
    frames = {i: {"sensing": {}} for i in range(400)}
    for step in (10, 100, 200, 300):
        frames[step] = {
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "sensing": build_visualizer_sensing_payload(
                {"detections": [{"track_id": 1, "x_m": float(step), "y_m": 0.0}]},
                scene_frame_id=step,
                sample_time_s=float(step),
            ),
        }
    service = SensingService(SimpleNamespace(cache_service=_frame_cache(frames)))
    loaded = []
    original = service._load_history_frame

    def load(step):
        loaded.append(step)
        return original(step)

    monkeypatch.setattr(service, "_load_history_frame", load)
    for _ in range(2):
        history = service._history_frames(frames[200], step=200, history_frames=2)
        assert [step for step, _ in history] == [100, 200]
    assert loaded == [100]
    assert len(service._overlay_history_cache) == 1
    storage_ids = service.candidate_storage_frame_ids(200)
    first = service._refresh_metadata_index_token(storage_ids)
    assert service._refresh_metadata_index_token(storage_ids) is first
    # Replacing the immutable source inventory invalidates the history cache.
    service._install_metadata_index_token(("different_generation",))
    assert not service._overlay_history_cache


def test_full_track_trail_keeps_every_intermediate_vertex_and_excludes_future() -> None:
    """A curved trail must not collapse to its first/last points on stride playback."""
    points = [(0.0, 0.0), (1.0, 2.0), (2.0, -1.0), (3.0, 3.0), (4.0, 0.0)]
    frames = {}
    for index, (x, y) in enumerate(points, start=1):
        step = index * 128
        frames[step] = {
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "sensing": build_visualizer_sensing_payload(
                {"detections": [{"track_id": 7, "x_m": x, "y_m": y, "z_m": 0.0}]},
                scene_frame_id=step,
                sample_time_s=float(step),
            ),
        }
    service = SensingService(SimpleNamespace(cache_service=_frame_cache(frames)))
    layers = service.build_scene_overlays(
        frames[512],
        step=512,
        mode="full",
        history_frames=2,
        use_full_history=True,
    )
    trail = next(layer for layer in layers if layer.name == "sensing_track_trail_7")
    np.testing.assert_array_equal(trail.payload.points, [[*p, 0.0] for p in points[:4]])
    np.testing.assert_array_equal(trail.payload.lines, [[0, 1], [1, 2], [2, 3]])


def test_full_history_longer_than_64_results_does_not_reload_at_every_tick(monkeypatch):
    frames = {
        step: {
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "sensing": build_visualizer_sensing_payload(
                {"detections": [{"track_id": 1, "x_m": float(step), "y_m": 0.0}]},
                scene_frame_id=step,
                sample_time_s=float(step),
            ),
        }
        for step in range(80)
    }
    service = SensingService(SimpleNamespace(cache_service=_frame_cache(frames)))
    first = service._history_frames(frames[78], step=78, history_frames=12, use_full_history=True)
    assert [step for step, _ in first] == list(range(79))
    monkeypatch.setattr(service, "_load_history_frame", lambda _: pytest.fail("history reread"))
    warm = service._history_frames(frames[78], step=78, history_frames=12, use_full_history=True)
    assert [step for step, _ in warm] == list(range(79))
    assert all(frame["sensing"]["detections"] for _, frame in warm)


def test_track_history_loads_detections_missing_from_visited_scene_cache() -> None:
    """Normal scene playback must not erase history when overlays open later."""
    frames = {}
    cached = {}
    for step in (128, 256, 384, 512):
        sensing = build_visualizer_sensing_payload(
            {"detections": [{"track_id": 1, "x_m": step / 128.0, "y_m": 0.0}]},
            scene_frame_id=step,
            sample_time_s=float(step),
        )
        frames[step] = replace(build_standard_mpc_frame(frame_idx=step), sensing=sensing)
        cached[step] = {
            "sensing": {},
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
        }
    provider = _CountingMultiFrameProvider(frames)
    service = SensingService(
        SimpleNamespace(
            cache_service=_frame_cache(cached),
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
        )
    )
    current = {**cached[384], "sensing": dict(frames[384].sensing)}
    history = service._history_frames(current, step=384, history_frames=12, use_full_history=True)
    assert [step for step, _frame in history] == [128, 256, 384]
    assert all(frame["sensing"]["detections"][0]["track_id"] == 1 for _, frame in history)


def test_overlay_snapshots_reuse_validation_and_invalidate_pose_style_generation(monkeypatch):
    """Warm playback reuses render payloads, with explicit source/pose invalidation."""

    def result(x):
        return replace(
            build_standard_mpc_frame(frame_idx=128),
            sensing=build_visualizer_sensing_payload(
                {"detections": [{"track_id": 1, "x_m": x, "y_m": 0.0, "z_m": 0.0}]},
                scene_frame_id=128,
                sample_time_s=128.0,
            ),
        )

    provider = _GenerationSwitchingProvider({128: result(1.0)}, {})
    service = SensingService(
        SimpleNamespace(
            frame_loader=SimpleNamespace(provider=provider),
            frame_source=None,
            cache_service=None,
        )
    )
    scene = {"rx_positions": np.zeros((1, 3)), "rx_orientations": np.zeros((1, 3))}
    kwargs = dict(
        step=128,
        mode="tracks",
        history_frames=12,
        use_full_history=False,
        show_velocity_arrows=False,
        pin_freshest=False,
    )
    first = service.resolve_scene_overlay_layers(scene, **kwargs)
    read_count = len(provider.requests)
    original = service.resolve_display_sensing_frame
    monkeypatch.setattr(
        service,
        "resolve_display_sensing_frame",
        lambda *a, **k: pytest.fail("warm view revalidated"),
    )
    assert service.resolve_scene_overlay_layers(scene, **kwargs) is first
    assert len(provider.requests) == read_count
    monkeypatch.setattr(service, "resolve_display_sensing_frame", original)
    scene["rx_positions"][0, 0] = 5.0
    moved = service.resolve_scene_overlay_layers(scene, **kwargs)
    np.testing.assert_allclose(moved[1][0].payload.points, [[6.0, 0.0, 0.0]])
    styled = service.resolve_scene_overlay_layers(scene, **{**kwargs, "mode": "detections"})
    assert len(styled[1]) == 1
    provider.frames[128] = result(3.0)
    provider.generation_id = "new-acquisition"
    changed = service.resolve_scene_overlay_layers(scene, **kwargs)
    np.testing.assert_allclose(changed[1][0].payload.points, [[8.0, 0.0, 0.0]])
    assert len(service._scene_overlay_cache) == 1
    assert not service._current_detection_cache


def test_current_detection_loop_keeps_views_across_intermediate_scene_rows(monkeypatch):
    """A 192-result loop must not reload/revalidate every result on each pass."""
    from visualizer.src.sensing.selection import matches_detection, select_estimate

    frames = {step: build_standard_mpc_frame(frame_idx=step) for step in range(192 * 4)}
    result_steps = list(range(0, 192 * 4, 4))
    for step in result_steps:
        frames[step + 1] = replace(
            frames[step + 1],
            sensing=build_visualizer_sensing_payload(
                {"detections": [{"track_id": 1, "x_m": float(step), "y_m": 0.0, "z_m": 0.0}]},
                scene_frame_id=step,
                sample_time_s=float(step),
            ),
        )
    provider = _GenerationSwitchingProvider(frames, {})
    service = SensingService(
        SimpleNamespace(frame_loader=SimpleNamespace(provider=provider), cache_service=None)
    )
    kwargs = dict(
        mode="detections",
        history_frames=12,
        use_full_history=True,
        show_velocity_arrows=False,
        pin_freshest=False,
    )

    def scene(step):
        return {
            "timestamp_s": float(step),
            "rx_positions": np.zeros((1, 3)),
            "rx_orientations": np.zeros((1, 3)),
            "tx_positions": np.zeros((1, 3)),
            "tx_orientations": np.zeros((1, 3)),
        }

    prepared = {}
    for step in frames:
        view = service.resolve_scene_overlay_layers(scene(step), step=step, **kwargs)
        if step in result_steps:
            assert view is not None
            prepared[step] = view
        else:
            assert view is None
    assert len(service._current_detection_cache) == 192
    reads = len(provider.requests)
    monkeypatch.setattr(
        service,
        "resolve_display_sensing_frame",
        lambda *a, **k: pytest.fail("Repeated current-detection view reloaded or revalidated"),
    )
    for step in result_steps:
        assert (
            service.resolve_scene_overlay_layers(scene(step), step=step, **kwargs) is prepared[step]
        )
    assert len(provider.requests) == reads
    annotations = prepared[0][0]["sensing"]
    assert "samples" not in annotations["temporal"]["acquisition"]
    selected = select_estimate(annotations, detection_index=0)
    assert matches_detection(selected, annotations, 0, annotations["detections"][0])


def test_current_detection_view_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.services.sensing_service._CURRENT_DETECTION_CACHE_CAPACITY", 2
    )
    frames = {
        step: replace(
            build_standard_mpc_frame(frame_idx=step),
            sensing=build_visualizer_sensing_payload(
                {"detections": [{"x_m": float(step), "y_m": 0.0, "z_m": 0.0}]},
                scene_frame_id=step,
                sample_time_s=float(step),
            ),
        )
        for step in range(3)
    }
    provider = _GenerationSwitchingProvider(frames, {})
    service = SensingService(
        SimpleNamespace(frame_loader=SimpleNamespace(provider=provider), cache_service=None)
    )
    for step in frames:
        service.resolve_scene_overlay_layers(
            {"rx_positions": np.zeros((1, 3)), "rx_orientations": np.zeros((1, 3))},
            step=step,
            mode="detections",
            history_frames=12,
            use_full_history=False,
            show_velocity_arrows=False,
            pin_freshest=False,
        )
    assert len(service._current_detection_cache) == 2
