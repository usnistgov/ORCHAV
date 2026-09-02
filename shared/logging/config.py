"""Logging configuration shared by ORCHAV applications and tools.

``configure_logging`` installs console/file handlers and honors environment
overrides. ``get_logger`` maps package logger names into the ``orchav``
namespace so generator, visualizer, and shared logs can be filtered together.
"""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any, Dict

from .formatters import JsonFormatter

# Named console/file formatting profiles accepted by ``ORCHAV_LOG_FORMAT``.
PROFILES: Dict[str, Dict[str, Any]] = {
    "compact": {"format": "%(levelname)s %(name)s: %(message)s"},
    "minimal": {"format": "%(levelname)s %(message)s"},
    "verbose": {
        "format": "%(asctime)s %(levelname)s [%(threadName)s] %(name)s:%(lineno)d %(funcName)s: %(message)s",
        "datefmt": "%H:%M:%S",
    },
    "json": {"()": JsonFormatter},
}

LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB rotating log file size limit

_NOISY_LOGGERS = ("grpc", "h5py", "matplotlib", "PIL")
_PROJECT_LOGGER_NAMESPACES = ("orchav", "generator", "visualizer", "shared")
_configured = False


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if not level:
        return logging.WARNING
    return getattr(logging, str(level).upper(), logging.WARNING)


def resolve_log_level(level: str | int | None = None) -> int:
    """Resolve a requested level with the process environment taking precedence."""
    configured_level = os.environ.get("ORCHAV_LOG_LEVEL")
    return _coerce_level(configured_level if configured_level is not None else level)


def get_current_log_level_name() -> str:
    """Return the effective project logging level as a display-ready name."""
    return logging.getLevelName(logging.getLogger("orchav").getEffectiveLevel())


def configure_logging(
    level: str | int | None = None,
    format_profile: str | None = None,
    include_third_party: bool = False,
) -> None:
    """Configure process logging from explicit arguments and environment.

    ``ORCHAV_LOG_LEVEL``, ``ORCHAV_LOG_FORMAT``, and ``ORCHAV_LOG_FILE`` take
    precedence over function arguments so command-line tools can be controlled
    from shell wrappers.
    """
    global _configured

    resolved_profile = os.environ.get("ORCHAV_LOG_FORMAT", format_profile or "compact")
    log_file = os.environ.get("ORCHAV_LOG_FILE")

    numeric_level = resolve_log_level(level)
    formatter_key = PROFILES.get(resolved_profile, PROFILES["compact"])

    formatters: Dict[str, Dict[str, Any]] = {"default": formatter_key}
    # Always make verbose available for file output
    formatters.setdefault("verbose", PROFILES["verbose"])

    handlers: Dict[str, Dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": numeric_level,
        }
    }

    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "verbose",
            "level": numeric_level,
            "filename": log_file,
            "maxBytes": LOG_FILE_MAX_BYTES,
            "backupCount": 5,
        }

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "orchav": {"level": numeric_level, "handlers": ["console"], "propagate": False},
            "generator": {"level": numeric_level, "handlers": [], "propagate": True},
            "visualizer": {"level": numeric_level, "handlers": [], "propagate": True},
            "shared": {"level": numeric_level, "handlers": [], "propagate": True},
        },
        "root": {"level": numeric_level, "handlers": ["console"]},
    }

    if log_file:
        logging_config["loggers"]["orchav"]["handlers"].append("file")
        logging_config["root"]["handlers"].append("file")

    if not include_third_party:
        for noisy in _NOISY_LOGGERS:
            logging_config["loggers"][noisy] = {
                "level": "WARNING",
                "handlers": [],
                "propagate": True,
            }

    logging.config.dictConfig(logging_config)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a project logger under the common ``orchav`` namespace."""
    if name.startswith(("generator", "visualizer", "shared")):
        name = f"orchav.{name}"
    return logging.getLogger(name)


def set_log_level(level: str | int) -> int:
    """Update root and all project logger/handler levels at runtime."""
    numeric_level = _coerce_level(level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)

    project_loggers = {logging.getLogger(namespace) for namespace in _PROJECT_LOGGER_NAMESPACES}
    logger_dict = logging.Logger.manager.loggerDict
    for name, logger in list(logger_dict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        if any(
            name == namespace or name.startswith(f"{namespace}.")
            for namespace in _PROJECT_LOGGER_NAMESPACES
        ):
            project_loggers.add(logger)

    for project_logger in project_loggers:
        project_logger.setLevel(numeric_level)
        for handler in project_logger.handlers:
            handler.setLevel(numeric_level)

    return numeric_level
