"""Config loading with dot-path access.

WHY a tiny wrapper instead of raw dicts: cfg.get("detection.frame_diff.threshold")
fails loudly with the *full path* in the error when a key is missing, instead of a
bare KeyError('threshold') three dicts deep with no context.

Overlay files: if a `local.yaml` sits next to the loaded config, it is
deep-merged on top. WHY: the tuning UI needs somewhere to persist values
without rewriting default.yaml (which would destroy its comments), and users
need per-machine tweaks that never show up in `git diff`. local.yaml is
gitignored on purpose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

OVERLAY_NAME = "local.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in place) and return base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def set_dotted(data: dict, dotted: str, value: Any) -> None:
    """Set a dot-path key in a nested dict, creating intermediate dicts."""
    node = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class Config:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        overlay = path.with_name(OVERLAY_NAME)
        if overlay.exists() and overlay != path:
            with open(overlay) as f:
                over = yaml.safe_load(f) or {}
            deep_merge(data, over)
            print(f"[config] applied overrides from {overlay}")
        return cls(data)

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


def save_overlay(config_path: str | Path, values: dict[str, Any]) -> Path:
    """Merge dotted-key values into the overlay file next to config_path.

    Existing overlay keys not being written are preserved.
    """
    overlay = Path(config_path).with_name(OVERLAY_NAME)
    data: dict = {}
    if overlay.exists():
        with open(overlay) as f:
            data = yaml.safe_load(f) or {}
    for dotted, value in values.items():
        set_dotted(data, dotted, value)
    header = ("# Machine-local overrides (auto-merged on top of the config it sits\n"
              "# next to). Written by the tuning UI's Save button; safe to hand-edit\n"
              "# or delete. Gitignored on purpose.\n")
    with open(overlay, "w") as f:
        f.write(header)
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)
    return overlay
