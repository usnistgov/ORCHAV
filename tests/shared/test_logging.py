import logging
import sys
from typing import Callable


def _reset_logging(module_loader: Callable[[], object]):
    """Reset logging state and return a fresh shared.logging module."""
    # Clear existing handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.shutdown()

    # Remove any mocked versions of shared.logging from sys.modules
    # This can happen if other tests mocked the module
    to_remove = [k for k in sys.modules if k.startswith("shared.logging")]
    for k in to_remove:
        del sys.modules[k]

    # Import fresh module
    module = module_loader()
    return module


def _load_module():
    import shared.logging as slog

    return slog


def test_get_logger_namespacing(monkeypatch):
    slog = _reset_logging(_load_module)
    slog.configure_logging(level="INFO")
    logger = slog.get_logger("generator.core.pipeline")
    assert logger.name == "orchav.generator.core.pipeline"


def test_env_level_overrides(monkeypatch):
    monkeypatch.setenv("ORCHAV_LOG_LEVEL", "DEBUG")
    slog = _reset_logging(_load_module)
    slog.configure_logging()
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
    assert slog.resolve_log_level("INFO") == logging.DEBUG
    assert slog.get_current_log_level_name() == "DEBUG"
    monkeypatch.delenv("ORCHAV_LOG_LEVEL", raising=False)


def test_third_party_suppressed_by_default(monkeypatch):
    slog = _reset_logging(_load_module)
    slog.configure_logging(level="INFO")
    mpl_logger = logging.getLogger("matplotlib")
    assert mpl_logger.getEffectiveLevel() == logging.WARNING


def test_idempotent_no_duplicate_handlers(monkeypatch):
    slog = _reset_logging(_load_module)
    slog.configure_logging(level="INFO")
    before = len(logging.getLogger("orchav").handlers)
    slog.configure_logging(level="INFO")
    after = len(logging.getLogger("orchav").handlers)
    assert after == before


def test_default_level_is_warning(monkeypatch):
    monkeypatch.delenv("ORCHAV_LOG_LEVEL", raising=False)
    slog = _reset_logging(_load_module)

    slog.configure_logging()

    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    assert slog.get_current_log_level_name() == "WARNING"


def test_runtime_level_updates_project_loggers_and_handlers(monkeypatch):
    monkeypatch.delenv("ORCHAV_LOG_LEVEL", raising=False)
    slog = _reset_logging(_load_module)
    slog.configure_logging(level="WARNING")
    logger_names = (
        "orchav",
        "orchav.visualizer.runtime",
        "generator",
        "generator.runtime",
        "visualizer",
        "visualizer.runtime",
        "shared",
        "shared.runtime",
    )
    project_loggers = [logging.getLogger(name) for name in logger_names]

    assert slog.set_log_level("DEBUG") == logging.DEBUG

    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG
    assert all(logger.getEffectiveLevel() == logging.DEBUG for logger in project_loggers)
    assert all(handler.level == logging.DEBUG for handler in logging.getLogger().handlers)
    assert all(handler.level == logging.DEBUG for handler in logging.getLogger("orchav").handlers)
    assert slog.get_current_log_level_name() == "DEBUG"
