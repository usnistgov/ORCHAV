"""Prepare scenario actors and dispatch file or streaming execution.

``perform_pipeline`` is the generator entry point used by the CLI and tools.
It constructs role-specific scene adapters, resolves references supplied by
Python callers, and delegates execution to the selected backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from shared.frames.frame_set_writer import FrameSetWriter
from shared.logging import configure_logging, get_logger
from shared.scenarios.actors import ActorsSpec, GroupSpec

from ..configuration import ReceiverConfig, SimulationConfig, TransmitterConfig
from .handles import StreamingHandle
from .offline_pipeline import perform_offline_pipeline
from .progress import ProgressInfo

logger = get_logger(__name__)

_OPTIONAL_GRPC_MODULES = ("grpc", "google.protobuf")


def _is_optional_grpc_import_error(exc: ModuleNotFoundError) -> bool:
    """Return whether *exc* identifies the optional gRPC runtime."""
    missing = exc.name or ""
    return missing == "google" or any(
        missing == module or missing.startswith(f"{module}.") for module in _OPTIONAL_GRPC_MODULES
    )


def _load_streaming_pipeline() -> Callable[..., StreamingHandle]:
    """Load streaming support only after the caller selects that backend."""
    try:
        from .streaming import perform_pipeline_streaming
    except ModuleNotFoundError as exc:
        if not _is_optional_grpc_import_error(exc):
            raise
        raise RuntimeError(
            "Generator live streaming requires the optional gRPC transport; "
            'run python -m pip install -e ".[grpc]"'
        ) from exc
    return perform_pipeline_streaming


def perform_pipeline(
    tx_configs: list[TransmitterConfig] | None = None,
    rx_configs: list[ReceiverConfig] | None = None,
    target_configs: list[Any] | None = None,
    simulation_config: SimulationConfig | None = None,
    scenario_configuration: Any | None = None,
    *,
    configure_logging_enabled: bool = True,
    pipeline_mode: str | None = None,
    grpc_port: int | None = None,
    grpc_bind_host: str | None = None,
    blocking: bool = True,
    on_step_complete: Callable[[ProgressInfo], None] | None = None,
    geometry_only: bool = False,
    show_progress: bool = True,
    actors: ActorsSpec | None = None,
    groups: tuple[GroupSpec, ...] | None = None,
    frame_set_writer: FrameSetWriter | None = None,
) -> str | StreamingHandle | None:
    """Resolve actors and run the selected file or streaming backend.

    Callers supply either schema actors with a scenario configuration or a
    complete pair of role-specific TX/RX config lists. An explicit pipeline
    mode overrides scenario hints, and ``frame_set_writer`` belongs only to
    file-output generation. Non-blocking streaming returns a lifecycle handle.
    """
    if configure_logging_enabled and simulation_config is not None:
        configure_logging(level=simulation_config.debug_level)

    if simulation_config is None:
        raise ValueError("simulation_config is required to run the pipeline")

    if actors is not None:
        if scenario_configuration is None:
            raise ValueError("scenario_configuration is required with actors")
        if tx_configs is not None or rx_configs is not None or target_configs is not None:
            raise ValueError("actors cannot be combined with role-specific runtime configs")
        scenario_configuration = replace(
            scenario_configuration,
            actors=actors,
            groups=(
                tuple(groups)
                if groups is not None
                else tuple(getattr(scenario_configuration, "groups", ()) or ())
            ),
        )

    actor_runtime = None
    if tx_configs is None and rx_configs is None and scenario_configuration is not None:
        from ..scenario_actors.runtime import prepare_actor_runtime

        # One prepared actor timeline drives both live scene state and output.
        actor_runtime = prepare_actor_runtime(scenario_configuration)
        tx_configs = list(actor_runtime.transmitters)
        rx_configs = list(actor_runtime.receivers)
        if target_configs is None:
            target_configs = list(actor_runtime.targets)
    elif (tx_configs is None) != (rx_configs is None):
        raise ValueError("explicit Python actor configuration requires both TX and RX lists")

    tx_configs = tx_configs or []
    rx_configs = rx_configs or []
    target_configs = target_configs or []

    if geometry_only and scenario_configuration is not None:
        rt = getattr(scenario_configuration, "raytracing", None)
        if isinstance(rt, dict):
            rt["geometry_only"] = True

    resolved_mode = _infer_pipeline_mode(simulation_config, scenario_configuration, pipeline_mode)
    if resolved_mode in {"streaming", "grpc", "live_grpc", "stream"}:
        if frame_set_writer is not None:
            raise ValueError("frame_set_writer is supported only by file-output generation")
        if int(getattr(simulation_config, "start_step", 0) or 0) > 0:
            raise ValueError(
                "raytracing.start_step is only supported for file-output generation; "
                "live streaming serves requested frames on demand and does not publish "
                "a fresh partial frame set"
            )
        handle = _load_streaming_pipeline()(
            tx_configs,
            rx_configs,
            target_configs,
            simulation_config,
            scenario_configuration=scenario_configuration,
            grpc_port=grpc_port,
            grpc_bind_host=grpc_bind_host,
        )

        if blocking:
            try:
                handle.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Stopping streaming pipeline...")
                handle.shutdown()
                logger.info("Streaming pipeline stopped")
            else:
                handle.close()

        return handle

    return perform_offline_pipeline(
        tx_configs,
        rx_configs,
        target_configs,
        simulation_config,
        scenario_configuration=scenario_configuration,
        frame_set_writer=frame_set_writer,
        on_step_complete=on_step_complete,
        show_progress=show_progress,
    )


def _infer_pipeline_mode(
    simulation_config: SimulationConfig | None,
    scenario_configuration: Any | None,
    explicit_mode: str | None,
) -> str:
    """Return ``'file'`` or ``'streaming'`` from explicit and scenario hints."""

    if explicit_mode:
        return explicit_mode.lower()

    output_mode = str(getattr(simulation_config, "output_mode", "") or "").lower()
    if output_mode in {"grpc", "streaming", "stream", "live_grpc"}:
        return "streaming"
    if output_mode in {"local", "file", "offline"}:
        return "file"

    if scenario_configuration is not None:
        data_mode = str(getattr(scenario_configuration, "data_mode", "") or "").lower()
        if data_mode == "live_grpc":
            return "streaming"

        timeline_cfg = getattr(scenario_configuration, "timeline", {}) or {}
        motion_mode = str(timeline_cfg.get("mode", "") or "").lower()
        if motion_mode == "step":
            return "streaming"

    return "file"
