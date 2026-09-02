"""
Configuration file handlers for the visualizer.

Provides handlers for reading/writing JSON configuration files,
managing recent file lists, and plain text file operations.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from shared.logging import get_logger
except ImportError:
    import logging

    def get_logger(name):
        """Return a basic logger when shared logging is unavailable."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_logger("orchav")


class ConfigFileHandler:
    """Handles configuration file operations."""

    @staticmethod
    def load_config(config_file: str) -> dict[str, Any]:
        """
        Load configuration from a file.

        Args:
            config_file: Path to the configuration file

        Returns:
            Dictionary containing configuration data
        """
        try:
            with open(config_file, encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Loaded config from {config_file}")

            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                raise ValueError(f"Config file {config_file} must contain a JSON object")
            return parsed

        except OSError as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            raise

    @staticmethod
    def save_config(config_file: str, config_data: dict[str, Any]) -> None:
        """
        Save configuration to a file.

        Config files are persisted as JSON objects so they can be loaded by
        ``load_config`` and external tooling without Python ``repr`` parsing.

        Args:
            config_file: Path to the configuration file
            config_data: Configuration data to save
        """
        try:
            config_dir = os.path.dirname(config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, sort_keys=True)
                f.write("\n")
                logger.debug(f"Saved config to {config_file}")

        except OSError as e:
            logger.error(f"Failed to save config to {config_file}: {e}")
            raise


class RecentFilesHandler:
    """Handles recent files configuration management."""

    @staticmethod
    def load_recent_files(config_file: str, max_recent_files: int = 3) -> list[str]:
        """
        Load recent files from configuration file.

        Args:
            config_file: Path to the configuration file
            max_recent_files: Maximum number of recent files to keep

        Returns:
            List of recent file paths
        """
        try:
            if os.path.exists(config_file):
                content = TextFileHandler.read_text_file(config_file)
                config = json.loads(content)
                recent_files = config.get("recent_files", [])

                original_count = len(recent_files)
                recent_files = [f for f in recent_files if os.path.exists(f)]

                if len(recent_files) < original_count:
                    RecentFilesHandler.save_recent_files(config_file, recent_files)

                if len(recent_files) > max_recent_files:
                    recent_files = recent_files[:max_recent_files]

                logger.info(f"Loaded {len(recent_files)} recent files from: {config_file}")
                return recent_files
            else:
                logger.info(f"No config file found at: {config_file}")
                return []

        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load recent files config: {e}")
            return []

    @staticmethod
    def save_recent_files(config_file: str, recent_files: list[str]) -> list[str]:
        """
        Save recent files to configuration file.

        Args:
            config_file: Path to the configuration file
            recent_files: List of recent file paths to save

        Returns:
            The list of recent file paths that was saved.
        """
        try:
            config = {"recent_files": recent_files}

            config_dir = os.path.dirname(config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            TextFileHandler.write_text_file(config_file, json.dumps(config, indent=2))

            logger.info(f"Saved {len(recent_files)} recent files to: {config_file}")

        except OSError as e:
            logger.warning(f"Could not save recent files config: {e}")
        return recent_files

    @staticmethod
    def add_recent_file(config_file: str, file_path: str, max_recent_files: int = 3) -> list[str]:
        """
        Add a file to recent files list and save to config.

        Args:
            config_file: Path to the configuration file
            file_path: Path to add to recent files
            max_recent_files: Maximum number of recent files to keep

        Returns:
            Updated list of recent files
        """
        try:
            recent_files = RecentFilesHandler.load_recent_files(config_file, max_recent_files)

            if file_path in recent_files:
                recent_files.remove(file_path)

            recent_files.insert(0, file_path)

            if len(recent_files) > max_recent_files:
                recent_files = recent_files[:max_recent_files]

            RecentFilesHandler.save_recent_files(config_file, recent_files)

            return recent_files

        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not add recent file: {e}")
            return []


class TextFileHandler:
    """Handles text file operations."""

    @staticmethod
    def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
        """
        Read text from a file.

        Args:
            file_path: Path to the text file
            encoding: File encoding

        Returns:
            File content as string
        """
        try:
            with open(file_path, encoding=encoding) as f:
                content = f.read()
                logger.debug(f"Read text file: {file_path}")
                return content

        except OSError as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise

    @staticmethod
    def write_text_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
        """
        Write text to a file.

        Args:
            file_path: Path to the text file
            content: Content to write
            encoding: File encoding
        """
        try:
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
                logger.debug(f"Wrote text file: {file_path}")

        except OSError as e:
            logger.error(f"Failed to write text file {file_path}: {e}")
            raise
