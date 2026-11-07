"""LoRA state-dict parsing (FR-8, FR-9).

Scaling follows ComfyUI's ``comfy/weight_adapter/lora.py`` (``LoRAAdapter.calculate_weight``)::

    scale = strength * (alpha / rank)   if "<prefix>.alpha" is in the file
    scale = strength * 1.0              otherwise

where ``rank = down.shape[0]``. kohya files normally carry ``alpha``; peft/diffusers
files normally do not, so the same rank gives different scales between the two styles.
This module reproduces that rule exactly so the node matches ``LoraLoader`` (FR-8).

Key resolution uses a "key map" from LoRA key prefix to model weight key, the same
structure ComfyUI's ``comfy.lora.model_lora_keys_unet`` returns. Inside ComfyUI the
node merges ComfyUI's own map (which knows model-specific spellings, e.g. MOSS-Image's
separate q/k/v LoRAs onto a fused qkv weight) with :func:`generic_key_map`.
A value is either ``"diffusion_model.<path>.weight"`` or a tuple
``("diffusion_model.<path>.weight", (dim, offset, size))`` for a slice of a fused weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# (up suffix, down suffix) pairs. kohya first, then peft/diffusers spellings.
KOHYA_STYLE = ((".lora_up.weight", ".lora_down.weight"),)
PEFT_STYLE = (
    (".lora_B.weight", ".lora_A.weight"),
    (".lora_B.default.weight", ".lora_A.default.weight"),
    (".lora.up.weight", ".lora.down.weight"),
)
STYLES = KOHYA_STYLE + PEFT_STYLE
UNSUPPORTED_SUFFIXES = (".lora_mid.weight", ".dora_scale", ".hada_w1_a", ".lokr_w1", ".diff", ".diff_b", ".reshape_weight")

DM_PREFIX = "diffusion_model."


@dataclass
class LoraTarget:
    source_prefix: str      # the LoRA key prefix this came from
    weight_key: str         # model weight key, "diffusion_model.<path>.weight"
    module_path: str        # path relative to the diffusion model, "<path>"
    down: Any               # tensor [rank, in_features]   (lora_down / lora_A)
    up: Any                 # tensor [out, rank]           (lora_up / lora_B)
    alpha_scale: float      # alpha / rank, or 1.0 when the file has no alpha
    out_offset: int = 0     # slice of the module's output features this LoRA writes
    out_size: Optional[int] = None


@dataclass
class ParsedLora:
    targets: List[LoraTarget] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)   # (key, reason)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def generic_key_map(module_paths: Iterable[str]) -> Dict[str, Any]:
    """Prefixes accepted for every linear module path of the diffusion model."""
    km: Dict[str, Any] = {}
    for p in module_paths:
        target = f"{DM_PREFIX}{p}.weight"
        under = p.replace(".", "_")
        for alias in (p, f"{DM_PREFIX}{p}", f"transformer.{p}", f"lora_unet_{under}",
                      f"lora_transformer_{under}"):
            km.setdefault(alias, target)
    return km


def _alpha_value(v) -> float:
    return float(v.item()) if hasattr(v, "item") else float(v)


def parse_lora(sd: Dict[str, Any], key_map: Dict[str, Any]) -> ParsedLora:
    """Resolve a LoRA state dict against ``key_map``.

    Keys that do not resolve to a LoRA up/down pair on a mapped weight are returned in
    ``skipped`` with a reason, never raised (FR-9). Whether the target is actually a
    linear module is checked later, against the live model, by the gating layer.
    """
    result = ParsedLora()
    consumed = set()
    for prefix, to in key_map.items():
        style = next(((u, d) for u, d in STYLES if prefix + u in sd), None)
        if style is None:
            continue
        up_key, down_key = prefix + style[0], prefix + style[1]
        alpha_key = prefix + ".alpha"
        consumed.update(k for k in (up_key, down_key, alpha_key) if k in sd)
        if down_key not in sd:
            result.skipped.append((up_key, "up matrix without matching down matrix"))
            continue
        extra = [prefix + s for s in UNSUPPORTED_SUFFIXES if prefix + s in sd]
        if extra:
            consumed.update(extra)
            result.skipped.append((prefix, f"unsupported adapter component(s) {', '.join(s[len(prefix):] for s in extra)}; "
                                           "only plain low-rank LoRA can be applied as a gated delta"))
            continue
        up, down = sd[up_key], sd[down_key]
        if getattr(up, "ndim", 2) != 2 or getattr(down, "ndim", 2) != 2:
            result.skipped.append((prefix, f"not a linear LoRA (up {tuple(up.shape)}, down {tuple(down.shape)})"))
            continue
        rank = down.shape[0]
        if up.shape[1] != rank:
            result.skipped.append((prefix, f"rank mismatch: up {tuple(up.shape)} vs down {tuple(down.shape)}"))
            continue
        alpha_scale = _alpha_value(sd[alpha_key]) / rank if alpha_key in sd else 1.0
        offset, size = 0, None
        if isinstance(to, tuple):
            to, sl = to
            dim, offset, size = sl
            if dim != 0:
                result.skipped.append((prefix, f"slice along dim {dim} is not an output-feature slice"))
                continue
        if not (isinstance(to, str) and to.startswith(DM_PREFIX) and to.endswith(".weight")):
            result.skipped.append((prefix, f"target {to!r} is not a diffusion-model weight"))
            continue
        result.targets.append(LoraTarget(
            source_prefix=prefix, weight_key=to, module_path=to[len(DM_PREFIX):-len(".weight")],
            down=down, up=up, alpha_scale=alpha_scale, out_offset=offset, out_size=size))
    for k in sd:
        if k not in consumed:
            result.skipped.append((k, "key did not resolve to a diffusion-model linear module"))
    return result
