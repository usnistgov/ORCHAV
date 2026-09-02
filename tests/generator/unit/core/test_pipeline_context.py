from unittest.mock import MagicMock

import pytest

from generator.core.pipeline.context import PipelineContext


def test_pipeline_context_calls_cleanup_on_success():
    simulation_config = MagicMock()
    cleanup = MagicMock()
    service = MagicMock()
    service.cleanup = cleanup

    with PipelineContext(simulation_config) as ctx:
        ctx.services["dummy"] = service

    cleanup.assert_called_once()


def test_pipeline_context_calls_cleanup_on_exception():
    simulation_config = MagicMock()
    cleanup = MagicMock()
    service = MagicMock()
    service.cleanup = cleanup

    with pytest.raises(RuntimeError, match="boom"):
        with PipelineContext(simulation_config) as ctx:
            ctx.services["dummy"] = service
            raise RuntimeError("boom")

    cleanup.assert_called_once()
