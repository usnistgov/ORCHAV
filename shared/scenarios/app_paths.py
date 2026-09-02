"""Load repository-level defaults from ``config/app.toml``.

Scenario YAML owns scenario behavior. ``app.toml`` provides shared fallback
paths and service endpoints used when a scenario does not specify them. Missing
or unreadable configuration resolves to conservative defaults; the bounded
lightweight parser recovers only the simple sections this module owns.
"""

from pathlib import Path
from typing import Any, Dict

from shared.logging import get_logger

logger = get_logger(__name__)

try:
    import tomllib  # type: ignore
except ImportError:
    try:
        import toml as tomllib  # type: ignore
    except ImportError:
        tomllib = None


def load_app_paths(project_root: Path) -> Dict[str, Any]:
    """
    Load app-level path defaults from config/app.toml if available.

    Args:
        project_root: Project root directory

    Returns:
        Dictionary with path defaults
    """
    cfg = project_root / "config" / "app.toml"
    defaults = {
        "ibl": "libraries/ibl",
        "cmgen": "",
        "cmgen_args": "--format=ktx --size=256",
    }
    if not cfg.exists():
        return defaults

    if tomllib is not None:
        try:
            use_binary = getattr(tomllib, "__name__", "") == "tomllib"
            open_mode = "rb" if use_binary else "r"
            with open(cfg, open_mode, encoding=None if use_binary else "utf-8") as f:
                data = tomllib.load(f)
            paths = data.get("paths", {}) or {}
            if "ibl" in paths:
                defaults["ibl"] = str(paths["ibl"])
            if "cmgen" in paths:
                defaults["cmgen"] = str(paths["cmgen"])
            if "cmgen_args" in paths:
                defaults["cmgen_args"] = str(paths["cmgen_args"])
            return defaults
        except (OSError, ValueError, KeyError):
            logger.warning(
                "Could not parse app.toml with toml parser; falling back to lightweight parser"
            )
    else:
        logger.warning("TOML parser not available; using lightweight fallback parser for app.toml")

    return _parse_paths_lightweight(cfg, defaults)


def _parse_paths_lightweight(cfg: Path, defaults: Dict[str, str]) -> Dict[str, str]:
    """Lightweight TOML parser for [paths] section."""
    try:
        in_paths = False
        with open(cfg, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_paths = line[1:-1].strip() == "paths"
                    continue
                if not in_paths:
                    continue
                # Parse key = "value" or key = 'value'
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    # Remove comments after value
                    val = val.split("#", 1)[0].strip()
                    if (val.startswith('"') and '"' in val[1:]) or (
                        val.startswith("'") and "'" in val[1:]
                    ):
                        q = val[0]
                        end = val.find(q, 1)
                        if end != -1:
                            val = val[1:end]
                    # Assign recognized keys
                    if key == "ibl":
                        defaults["ibl"] = val
                    elif key == "cmgen":
                        defaults["cmgen"] = val
                    elif key == "cmgen_args":
                        defaults["cmgen_args"] = val
        return defaults
    except (OSError, ValueError):
        return defaults


def load_live_grpc_endpoints(project_root: Path) -> Dict[str, Any]:
    """
    Load live gRPC endpoints from config/app.toml if available.

    Args:
        project_root: Project root directory

    Returns:
        Dictionary with live gRPC endpoint defaults
    """
    cfg = project_root / "config" / "app.toml"
    defaults = {
        "sionna": "grpc://localhost:50051",
        "http": "http://localhost:8080",
    }
    if not cfg.exists():
        return defaults

    if tomllib is not None:
        try:
            use_binary = getattr(tomllib, "__name__", "") == "tomllib"
            open_mode = "rb" if use_binary else "r"
            with open(cfg, open_mode, encoding=None if use_binary else "utf-8") as f:
                data = tomllib.load(f)
            live_grpc = data.get("live_grpc", {}) or {}
            for k in ["sionna", "http"]:
                if k in live_grpc and live_grpc[k]:
                    defaults[k] = str(live_grpc[k])
            return defaults
        except (OSError, ValueError, KeyError):
            logger.warning("Could not parse app.toml [live_grpc]")

    return _parse_live_grpc_lightweight(cfg, defaults)


def _parse_live_grpc_lightweight(cfg: Path, defaults: Dict[str, str]) -> Dict[str, str]:
    """Lightweight TOML parser for [live_grpc] section."""
    try:
        in_live_grpc = False
        with open(cfg, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_live_grpc = line[1:-1].strip() == "live_grpc"
                    continue
                if not in_live_grpc:
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.split("#", 1)[0].strip().strip('"').strip("'")
                    if key in defaults:
                        defaults[key] = val
        return defaults
    except (OSError, ValueError):
        return defaults
