"""Custom log formatters used by shared logging profiles."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for machine-readable log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
        }
        if record.funcName:
            payload["func"] = record.funcName
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
