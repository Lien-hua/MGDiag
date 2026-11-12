import importlib
from pathlib import Path
from typing import Dict, Type

from extractor.base import DatasetAdapter


_ADAPTERS: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(name: str):
    def decorator(adapter_cls: Type[DatasetAdapter]):
        if name in _ADAPTERS:
            raise ValueError(f"dataset adapter already registered: {name}")
        _ADAPTERS[name] = adapter_cls
        return adapter_cls

    return decorator


def create_adapter(name: str, root: Path) -> DatasetAdapter:
    # Imports are intentionally local so optional dataset dependencies do not
    # affect users of another adapter.
    try:
        importlib.import_module(f"extractor.{name}_adapter")
    except ModuleNotFoundError as exc:
        if exc.name != f"extractor.{name}_adapter":
            raise

    try:
        adapter_cls = _ADAPTERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTERS)) or "none"
        raise ValueError(
            f"unknown dataset adapter {name!r}; registered adapters: {available}"
        ) from exc
    return adapter_cls(root)

