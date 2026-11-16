"""Regression tests for the sequential_stft processor.

Ensures:
1. The SequentialStftProcessor class exists and is importable.
2. The factory accepts "sequential_stft" as processing_mode.
3. SensingConfig validates the mode name.
4. The old naive_stft module no longer exists.
"""

import importlib

import pytest

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import (
    SensingProcessorBase,
    SequentialStftProcessor,
    create_sensing_processor,
)


class TestSequentialStftProcessorExists:
    def test_class_importable(self):
        assert issubclass(SequentialStftProcessor, SensingProcessorBase)

    def test_old_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("generator.core.sensing.strategies.naive_stft")

    def test_new_module_importable(self):
        mod = importlib.import_module("generator.core.sensing.strategies.sequential_stft")
        assert hasattr(mod, "process_sequential_stft")


class TestFactoryProcessingMode:
    def _make_config(self, mode: str) -> SensingConfig:
        return SensingConfig(enabled=True, processing_mode=mode)

    def test_sequential_stft_creates_correct_processor(self):
        cfg = self._make_config("sequential_stft")
        resolved = resolve_sensing_execution_values(
            cfg,
            cir_time_steps=1,
            cir_sampling_frequency_hz=None,
            frame_dt_s=1.0 / cfg.prf_hz,
        )
        proc = create_sensing_processor(cfg, 28e9, resolved)
        assert isinstance(proc, SequentialStftProcessor)


class TestSensingConfigValidation:
    def test_sequential_stft_accepted(self):
        cfg = SensingConfig(enabled=True, processing_mode="sequential_stft")
        assert cfg.processing_mode == "sequential_stft"

    def test_invalid_mode_fails_strictly(self):
        with pytest.raises(ConfigurationError, match="Invalid sensing.processing_mode"):
            SensingConfig(enabled=True, processing_mode="bogus_mode")
