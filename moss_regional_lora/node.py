"""The ComfyUI node (FR-1). Imports ComfyUI lazily so the package stays importable in tests."""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

from .gating import RegionalLoraController, iter_linear_modules
from .layout import load_layout
from .lora_file import generic_key_map, parse_lora
from .regions import parse_regions

log = logging.getLogger("moss_regional_lora")

PATCH_KEY = "moss_regional_lora"

DEFAULT_REGIONS = json.dumps({"version": 1, "regions": [], "groups": [],
                              "editor": {"grid_divisions": 3, "snap": False}}, indent=2)

_LORA_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_LORA_CACHE_SIZE = 6


def _load_lora_file(name: str):
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path_or_raise("loras", name)
    st = os.stat(path)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _LORA_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        _LORA_CACHE.move_to_end(path)
        return hit[1]
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    try:
        import comfy.lora_convert
        sd = comfy.lora_convert.convert_lora(sd)   # same pre-pass as LoraLoader
    except (ImportError, AttributeError):
        pass
    _LORA_CACHE[path] = (stamp, sd)
    while len(_LORA_CACHE) > _LORA_CACHE_SIZE:
        _LORA_CACHE.popitem(last=False)
    return sd


def _key_map(model_patcher, diffusion_model):
    km = generic_key_map(name for name, _ in iter_linear_modules(diffusion_model))
    try:
        import comfy.lora
        km.update(comfy.lora.model_lora_keys_unet(model_patcher.model, {}))
    except Exception as e:  # ComfyUI API drift must not break the node; generic map still works
        log.warning("[MOSS Regional LoRA] ComfyUI key map unavailable (%s); using generic key names only", e)
    return km


def _strip_previous(m):
    """Remove an earlier execution's injection and wrapper so re-execution replaces, not stacks (FR-10)."""
    from comfy.patcher_extension import WrappersMP
    m.remove_injections(PATCH_KEY)
    m.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, PATCH_KEY)


class MossRegionalLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16,
                                  "tooltip": "Width in pixels of the image being generated. Must match the latent."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16,
                                   "tooltip": "Height in pixels of the image being generated. Must match the latent."}),
                "regions": ("STRING", {"multiline": True, "default": DEFAULT_REGIONS,
                                       "tooltip": "Region list as JSON, edited with the canvas below. "
                                                  "Boxes are [y_min, x_min, y_max, x_max] px from the canvas centre."}),
                "vae_downscale": ("INT", {"default": 16, "min": 1, "max": 64,
                                          "tooltip": "Pixels per latent cell. ComfyUI's MOSS-Image latent format "
                                                     "(Flux2) is 16. Confirm with tools/discover_moss_layout.py."}),
                "patch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                                       "tooltip": "Latent cells per transformer token along each axis. "
                                                  "Confirm with tools/discover_moss_layout.py."}),
                "ramp_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Fraction of the steps at which the LoRA starts fading in (0 = from the first step)."}),
                "ramp_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Fraction of the steps at which the LoRA reaches full strength."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "Total sampling steps; set it to the sampler's step count."}),
                "strict": ("BOOLEAN", {"default": True,
                                       "tooltip": "Refuse to run while the sequence layout is unverified, and raise "
                                                  "instead of warning when a mask cannot be aligned."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Applies LoRAs to rectangular regions of a MOSS-Image generation by gating each LoRA's "
                   "delta per token at forward time. Weights are never merged.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The hooks live outside ModelPatcher's weight-patch tracking; never let the cache reuse us (NFR-4).
        return float("nan")

    def apply(self, model, width, height, regions, vae_downscale, patch_size, ramp_start, ramp_end, steps, strict):
        layout = load_layout(strict=strict)          # raises in strict mode if undiscovered (FR-11)
        m = model.clone()
        _strip_previous(m)
        if layout is None:
            msg = ("MOSS-Image layout constants are unset (see tools/discover_moss_layout.py); strict is off, "
                   "so the model is passed through WITHOUT any regional LoRA.")
            log.warning("[MOSS Regional LoRA] %s", msg)
            return {"ui": {"moss_regional_lora": [{"status": "inactive", "message": msg}]}, "result": (m,)}

        specs, notes = parse_regions(regions)
        for n in notes:
            log.info("[MOSS Regional LoRA] %s", n)
        if not specs:
            msg = "no visible regions with a LoRA; model passed through unchanged"
            return {"ui": {"moss_regional_lora": [{"status": "inactive", "message": msg, "notes": notes}]}, "result": (m,)}

        diffusion_model = m.get_model_object("diffusion_model")
        key_map = _key_map(m, diffusion_model)
        targets = []
        parse_skipped = []
        for spec in specs:
            parsed = parse_lora(_load_lora_file(spec.lora), key_map)
            targets.append(parsed.targets)
            parse_skipped += [(spec.lora, k, why) for k, why in parsed.skipped]

        controller = RegionalLoraController(
            specs, targets, layout, width, height, vae_downscale, patch_size,
            ramp_start, ramp_end, steps, strict).bind(diffusion_model)
        bind_skipped = [(specs[i].lora, k, why) for i, k, why in controller.skipped]
        skipped = parse_skipped + bind_skipped

        from comfy.patcher_extension import PatcherInjection, WrappersMP
        m.set_injections(PATCH_KEY, [PatcherInjection(inject=controller.inject, eject=controller.eject)])
        m.add_wrapper_with_key(WrappersMP.APPLY_MODEL, PATCH_KEY, controller.apply_model_wrapper)

        for lora, k, why in skipped[:50]:
            log.info("[MOSS Regional LoRA] skipped %s :: %s (%s)", lora, k, why)
        summary = {
            "status": "active",
            "regions": len(specs),
            "modules": len(controller.gates),
            "applied": controller.applied_count,
            "skipped": len(skipped),
            "notes": notes,
        }
        log.info("[MOSS Regional LoRA] %d region(s), %d LoRA layer(s) on %d module(s), %d key(s) skipped",
                 summary["regions"], summary["applied"], summary["modules"], summary["skipped"])
        if controller.applied_count == 0:
            msg = "none of the selected LoRAs' keys resolved to a linear module of this model"
            if strict:
                raise RuntimeError(f"[MOSS Regional LoRA] {msg}; {len(skipped)} key(s) skipped (see log)")
            log.warning("[MOSS Regional LoRA] %s", msg)
        return {"ui": {"moss_regional_lora": [summary]}, "result": (m,)}


NODE_CLASS_MAPPINGS = {"MossRegionalLora": MossRegionalLora}
NODE_DISPLAY_NAME_MAPPINGS = {"MossRegionalLora": "MOSS-Image Regional LoRA"}
