from __future__ import annotations

from dataclasses import dataclass, field

from generator.core.pipeline.handles import StreamingHandle


@dataclass
class _FakeGrpcServer:
    wait_result: bool
    stop_calls: list[float] = field(default_factory=list)
    stop_event: object | None = None

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        return self.wait_result

    def stop(self, grace: float):
        self.stop_calls.append(grace)
        return self.stop_event


class _FakeStopEvent:
    def __init__(self) -> None:
        self.wait_calls: list[float | None] = []

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)


class _FakeThread:
    def __init__(self) -> None:
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class _FakePipelineContext:
    def __init__(self) -> None:
        self.exit_calls = 0

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.exit_calls += 1


class _FakeGeneratorService:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_streaming_handle_reports_alive_when_grpc_wait_times_out() -> None:
    handle = StreamingHandle(
        server=_FakeGrpcServer(wait_result=True),
        generator_service=None,
        frame_cache=None,
        services={},
    )

    assert handle.is_alive is True


def test_streaming_handle_reports_dead_when_grpc_wait_does_not_time_out() -> None:
    handle = StreamingHandle(
        server=_FakeGrpcServer(wait_result=False),
        generator_service=None,
        frame_cache=None,
        services={},
    )

    assert handle.is_alive is False


def test_streaming_handle_close_marks_handle_dead() -> None:
    handle = StreamingHandle(
        server=_FakeGrpcServer(wait_result=True),
        generator_service=None,
        frame_cache=None,
        services={},
    )

    handle.close()

    assert handle.is_alive is False


def test_streaming_handle_close_owns_complete_idempotent_shutdown() -> None:
    stop_event = _FakeStopEvent()
    server = _FakeGrpcServer(wait_result=True, stop_event=stop_event)
    server_thread = _FakeThread()
    pipeline_context = _FakePipelineContext()
    generator_service = _FakeGeneratorService()
    handle = StreamingHandle(
        server=server,
        generator_service=generator_service,
        frame_cache=None,
        services={},
        server_thread=server_thread,
        pipeline_context=pipeline_context,
    )

    handle.close(timeout_s=1.25)
    handle.shutdown(timeout_s=9.0)

    assert server.stop_calls == [1.25]
    assert stop_event.wait_calls == [1.25]
    assert server_thread.join_calls == [1.25]
    assert generator_service.close_calls == 1
    assert pipeline_context.exit_calls == 1
    assert handle.pipeline_context is None
