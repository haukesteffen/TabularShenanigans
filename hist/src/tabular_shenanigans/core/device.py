from __future__ import annotations

import platform
from typing import Any


def detect_device(runtime_cfg: dict[str, Any]) -> str:
    configured = runtime_cfg.get("runtime", {}).get("device", "cpu")
    if configured != "auto":
        return str(configured)

    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "mps"
    return "cpu"
