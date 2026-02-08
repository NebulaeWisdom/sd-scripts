import json
from typing import Any, Dict, Optional


def build_caption(seed: int, caption: Optional[str]) -> str:
    if caption is None:
        return ""

    if not isinstance(caption, str):
        return str(caption)

    try:
        parsed = json.loads(caption)
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(caption)

    if isinstance(parsed, dict):
        return _build_caption_from_dict(seed, parsed, caption)

    return str(caption)


def _build_caption_from_dict(seed: int, data: Dict[str, Any], original_caption: str) -> str:
    _ = seed
    _ = data

    # Placeholder: implement dict -> caption logic here.
    return str(original_caption)
