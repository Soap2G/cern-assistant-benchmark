"""System registry — load adapters from systems.yaml (config-driven).

Adding a model/framework to the benchmark is a YAML entry + an adapter module;
no code change to the runner. See systems.yaml for the schema.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent
REGISTRY_FILE = ROOT / "systems.yaml"


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else REGISTRY_FILE
    data = yaml.safe_load(p.read_text()) or {}
    systems = data.get("systems", {})
    if not isinstance(systems, dict) or not systems:
        raise ValueError(f"No systems defined in {p}")
    return systems


def system_names() -> list[str]:
    return list(load_registry().keys())


def repeats_for(system: str) -> int:
    return int(load_registry()[system].get("repeats", 1))


def label_for(system: str) -> str:
    return load_registry()[system].get("label", system)


def make_adapter(system: str, record: bool = False):
    """Instantiate the adapter for `system` from the registry.

    Resolves "module.path:ClassName", then calls Class(record=record, **params).
    Adapters accept `record` and arbitrary params (ignored if irrelevant).
    """
    reg = load_registry()
    if system not in reg:
        raise ValueError(
            f"Unknown system '{system}'. Registered: {', '.join(reg)}"
        )
    spec = reg[system]
    dotted = spec["adapter"]
    if ":" not in dotted:
        raise ValueError(f"adapter for '{system}' must be 'module:Class', got {dotted!r}")
    mod_name, cls_name = dotted.split(":", 1)
    module = importlib.import_module(mod_name)
    cls = getattr(module, cls_name)
    params = dict(spec.get("params", {}) or {})
    return cls(record=record, **params)
