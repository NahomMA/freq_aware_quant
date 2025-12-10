import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "demo_config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

