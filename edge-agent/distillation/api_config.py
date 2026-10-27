"""Portable authentication helper; no AutoDL-specific credential path required."""
import os
from pathlib import Path


def resolve_api_key(path: Path | None = None) -> str:
    key = os.environ.get('QWEN_API_KEY')
    if key is not None:
        return key.strip()
    return path.read_text(encoding='utf-8').strip() if path is not None else ''
