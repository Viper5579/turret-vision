"""Config loading with dot-path access.

WHY a tiny wrapper instead of raw dicts: cfg.get("detection.frame_diff.threshold")
fails loudly with the *full path* in the error when a key is missing, instead of a
bare KeyError('threshold') three dicts deep with no context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(f"config key not found: '{dotted}' (failed at '{part}')")
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> dict:
        val = self.get(dotted)
        if not isinstance(val, dict):
            raise TypeError(f"config key '{dotted}' is not a section")
        return val
