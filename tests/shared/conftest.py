"""Conftest for shared package tests."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)
