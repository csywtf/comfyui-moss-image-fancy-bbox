"""Sequence layout: where the image tokens sit in the transformer's sequence (FR-5, FR-11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

DISCOVERY_TOOL = "tools/discover_moss_layout.py"

TEXT_POSITIONS = ("prefix", "suffix")
IMAGE_ORDERS = ("raster",)
SOURCE_TOKEN_POSITIONS = ("none", "start", "between", "end")

_CONSTANTS = {
    "TEXT_POSITION": TEXT_POSITIONS,
    "IMAGE_ORDER": IMAGE_ORDERS,
    "SOURCE_TOKENS": SOURCE_TOKEN_POSITIONS,
}


class LayoutNotEstablished(RuntimeError):
    """Raised when strict mode is on and the layout constants have not been discovered."""


class LayoutInvalid(ValueError):
    pass


@dataclass(frozen=True)
class Layout:
    text_position: str
    image_order: str
    source_tokens: str

    def __post_init__(self):
        for name, value, allowed in (
            ("TEXT_POSITION", self.text_position, TEXT_POSITIONS),
            ("IMAGE_ORDER", self.image_order, IMAGE_ORDERS),
            ("SOURCE_TOKENS", self.source_tokens, SOURCE_TOKEN_POSITIONS),
        ):
            if value not in allowed:
                raise LayoutInvalid(f"{name} = {value!r} is not one of {allowed}")

    def sequence_length(self, n_image: int, n_text: int, n_source: int) -> int:
        return n_image + n_text + (n_source if self.source_tokens != "none" else 0)

    def image_start(self, n_text: int, n_source: int) -> int:
        src = n_source if self.source_tokens != "none" else 0
        if self.text_position == "prefix":
            return n_text + (src if self.source_tokens in ("start", "between") else 0)
        # suffix: [image][text]; only leading source tokens push the image block back
        return src if self.source_tokens == "start" else 0

    def sequence_mask(self, image_mask: Sequence[float], seq_len: int, n_text: int, n_source: int) -> Optional[List[float]]:
        """Full-sequence mask for one region, or None if ``seq_len`` is not the joint sequence.

        Every position that is not an output-image token (text, source-image tokens)
        is exactly 0.0.
        """
        n_image = len(image_mask)
        if seq_len != self.sequence_length(n_image, n_text, n_source):
            return None
        start = self.image_start(n_text, n_source)
        out = [0.0] * seq_len
        out[start:start + n_image] = list(image_mask)
        return out


def unset_constants(config) -> List[str]:
    return [name for name in _CONSTANTS if getattr(config, name, None) is None]


def load_layout(config=None, strict: bool = True) -> Optional[Layout]:
    """Build the Layout from ``layout_config``.

    Returns None when constants are unset and ``strict`` is False. Raises
    :class:`LayoutNotEstablished` when constants are unset and ``strict`` is True.
    Invalid (set but unrecognised) values always raise.
    """
    if config is None:
        from . import layout_config as config
    missing = unset_constants(config)
    if missing:
        if strict:
            raise LayoutNotEstablished(
                "MOSS-Image sequence layout has not been established. Unset constant(s) in "
                f"moss_regional_lora/layout_config.py: {', '.join(missing)}. "
                f"Run {DISCOVERY_TOOL} on your ComfyUI machine against your MOSS-Image checkpoint "
                "and paste the values it prints into layout_config.py. Refusing to run, because a "
                "misaligned mask applies the LoRA to the wrong tokens and looks like a working node.")
        return None
    return Layout(config.TEXT_POSITION, config.IMAGE_ORDER, config.SOURCE_TOKENS)
