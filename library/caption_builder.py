from typing import Optional


def build_caption(seed: int, caption: Optional[str]) -> str:
    if caption is None:
        return ""
    return str(caption)
