from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PostNASConfig:
    enable: bool
    backend: str
    replace_layers: tuple[int, ...]
    window: int

    @classmethod
    def from_env(cls) -> "PostNASConfig":
        enable = os.environ.get("POSTNAS_ENABLE", "0") == "1"
        backend = os.environ.get("POSTNAS_BACKEND", "full").strip().lower() or "full"
        if backend not in {"full", "local"}:
            raise ValueError(f"POSTNAS_BACKEND must be one of [full, local], got: {backend}")
        window = int(os.environ.get("POSTNAS_WINDOW", "128"))
        if window <= 0:
            raise ValueError(f"POSTNAS_WINDOW must be positive, got: {window}")
        replace_layers_raw = os.environ.get("POSTNAS_REPLACE_LAYERS", "")
        replace_layers = _parse_layer_indices(replace_layers_raw)
        return cls(
            enable=enable,
            backend=backend,
            replace_layers=replace_layers,
            window=window,
        )


def _parse_layer_indices(raw: str) -> tuple[int, ...]:
    text = raw.strip()
    if not text:
        return ()
    values: list[int] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            raise ValueError(f"POSTNAS_REPLACE_LAYERS must use 0-based non-negative indices, got: {idx}")
        values.append(idx)
    return tuple(values)
