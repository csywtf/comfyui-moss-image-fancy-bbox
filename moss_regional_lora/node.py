"""The ComfyUI node (FR-1). Imports ComfyUI lazily so the package stays importable in tests."""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

from .gating import RegionalLoraController, iter_linear_modules
from .layout import (IMAGE_ORDERS, NOT_SET, SOURCE_TOKEN_POSITIONS, TEXT_POSITIONS, load_layout,
                     resolve_config)
from .lora_file import generic_key_map, parse_lora
from .layout_prompt import build_layout_prompt
from .regions import parse_document

log = logging.getLogger("moss_regional_lora")

PATCH_KEY = "moss_regional_lora"

# Measured with the MOSS-Image Layout Discovery node on a real MOSS-Image run (ComfyUI's 0.44-era
# MOSS-Image implementation): [image tokens, row by row][prompt][source-image tokens, edit formats that
# put them in the sequence]. Re-run the discovery after a ComfyUI update that changes that implementation.
MEASURED_TEXT_POSITION = "suffix"
MEASURED_IMAGE_ORDER = "raster"
MEASURED_SOURCE_TOKENS = "end"

DEFAULT_REGIONS = json.dumps({"version": 1, "regions": [], "groups": [],
                              "editor": {"grid_divisions": 3, "snap": False}}, indent=2)

_LORA_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_LORA_CACHE_SIZE = 6


def _file_stamp(path: str) -> tuple:
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _lora_fingerprint(regions) -> object:
    """What ComfyUI's cache cannot see from the inputs: the contents of the LoRA files the boxes name.

    The inputs (regions, size, settings, the upstream model) are already part of the cache key, so the
    node re-runs whenever one of them changes. This adds each used file's modification time and size, so
    a LoRA overwritten under the same name also re-runs it. ``NaN`` (always re-run) when the regions
    can't be read here: wired from another node, or a missing ComfyUI.
    """
    if not isinstance(regions, str):
        return float("nan")
    try:
        doc = parse_document(regions)
    except Exception:
        return regions   # the run reports the parse error; re-run whenever the text changes
    try:
        import folder_paths
    except ImportError:
        return float("nan")
    stamps = []
    for name in sorted({spec.lora for spec in doc.loras}):
        path = folder_paths.get_full_path("loras", name)
        try:
            stamps.append(f"{name}={_file_stamp(path)}" if path else f"{name}=missing")
        except OSError:
            stamps.append(f"{name}=unreadable")
    return ";".join(stamps)


def _load_lora_file(name: str):
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path_or_raise("loras", name)
    stamp = _file_stamp(path)
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
                                  "tooltip": "Width in pixels of the image being generated. Must match the latent (checked when sampling)."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16,
                                   "tooltip": "Height in pixels of the image being generated. Must match the latent (checked when sampling)."}),
                "regions": ("STRING", {"multiline": True, "default": DEFAULT_REGIONS,
                                       "tooltip": "Region list as JSON, edited with the canvas below. "
                                                  "Boxes are [y_min, x_min, y_max, x_max] px from the canvas centre."}),
                "ramp_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                                         "tooltip": "Only for boxes without their own strength curve (workflows saved "
                                                    "before curves): fraction of the steps at which the LoRA starts "
                                                    "fading in."}),
                "ramp_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                                       "tooltip": "Only for boxes without their own strength curve: fraction of the "
                                                  "steps at which the LoRA reaches full strength."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "advanced": True,
                                  "tooltip": "Fallback only: the strength curves follow the sampler's own step count. "
                                             "Used if a sampler does not report its schedule."}),
                "strict": ("BOOLEAN", {"default": True,
                                       "tooltip": "Refuse to apply LoRAs while the sequence layout is unverified, and "
                                                  "raise instead of warning when a mask cannot be aligned. Boxes "
                                                  "that only carry a prompt work either way."}),
                # last, so workflows saved before it existed keep their widget values in place
                "caption": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                       "tooltip": "Description of the whole image: the layout prompt's summary "
                                                  "caption. Boxes with a caption are added as layout items."}),
            },
            # Defaults measured by the "MOSS-Image Layout Discovery" node. "not set" falls back to layout_config.py.
            "optional": {
                "text_position": ([NOT_SET, *TEXT_POSITIONS], {
                    "default": MEASURED_TEXT_POSITION, "advanced": True,
                    "tooltip": "Where the prompt sits in the transformer sequence relative to the image tokens. "
                               "Default measured by the MOSS-Image Layout Discovery node; re-run it after a "
                               "ComfyUI update."}),
                "image_order": ([NOT_SET, *IMAGE_ORDERS], {
                    "default": MEASURED_IMAGE_ORDER, "advanced": True,
                    "tooltip": "Order of the image tokens (raster = row by row from the top left). "
                               "Default measured by the MOSS-Image Layout Discovery node; re-run it after a "
                               "ComfyUI update."}),
                "source_tokens": ([NOT_SET, *SOURCE_TOKEN_POSITIONS], {
                    "default": MEASURED_SOURCE_TOKENS, "advanced": True,
                    "tooltip": "Where an edit pipeline's source-image tokens sit in the sequence. "
                               "Default measured by the MOSS-Image Layout Discovery node; re-run it after a "
                               "ComfyUI update."}),
            },
        }

    # new outputs go last, so links in saved workflows keep their slots
    RETURN_TYPES = ("MODEL", "STRING", "INT", "INT")
    RETURN_NAMES = ("model", "prompt", "width", "height")
    OUTPUT_TOOLTIPS = ("The model with each box's LoRA applied inside its box.",
                       "The MOSS-Image layout prompt (same format as MOSS-Image Layout Prompt); "
                       "connect it to the positive text encode.",
                       "The canvas width; connect it to the empty latent node so the image matches the boxes.",
                       "The canvas height; connect it to the empty latent node so the image matches the boxes.")
    FUNCTION = "apply"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Draw boxes for a MOSS-Image generation. Each box can carry a layout-prompt caption, a LoRA "
                   "applied only inside the box (gated per token at forward time, weights never merged), or both. "
                   "Outputs the patched model and the layout prompt.")

    @classmethod
    def IS_CHANGED(cls, regions=None, **kwargs):
        # The patches live on this node's own clone of the model, added through ModelPatcher's injection and
        # wrapper API, so a cached result (same inputs) carries the same hooks and is safe to reuse. Only a
        # LoRA file changed on disk is invisible to the cache key; the fingerprint covers that (NFR-4).
        return _lora_fingerprint(regions)

    def apply(self, model, width, height, regions, ramp_start, ramp_end, steps,
              strict, caption="", text_position=None, image_order=None, source_tokens=None):
        # an API prompt without these options gets the same defaults as the UI
        text_position = MEASURED_TEXT_POSITION if text_position is None else text_position
        image_order = MEASURED_IMAGE_ORDER if image_order is None else image_order
        source_tokens = MEASURED_SOURCE_TOKENS if source_tokens is None else source_tokens
        doc = parse_document(regions)
        for n in doc.notes:
            log.info("[MOSS Regional LoRA] %s", n)
        prompt = build_layout_prompt(caption or "", doc.prompt_items)
        m = model.clone()
        _strip_previous(m)

        def out(summary):
            summary.setdefault("prompt_items", len(doc.prompt_items))
            summary.setdefault("notes", doc.notes)
            return {"ui": {"moss_regional_lora": [summary]}, "result": (m, prompt, width, height)}

        specs = doc.loras
        if not specs:
            # Prompt-only (or empty): nothing touches the transformer, so no layout is needed.
            return out({"status": "prompt-only", "message": "no box has a LoRA; model passed through unchanged"})

        config = resolve_config(text_position, image_order, source_tokens)
        layout = load_layout(config, strict=strict)  # raises in strict mode if undiscovered (FR-11)
        if layout is None:
            msg = ("MOSS-Image layout is not set (Canvas panel, Advanced; values from the MOSS-Image Layout Discovery "
                   "node); strict is off, so the model is passed through WITHOUT any regional LoRA.")
            log.warning("[MOSS Regional LoRA] %s", msg)
            return out({"status": "inactive", "message": msg})

        diffusion_model = m.get_model_object("diffusion_model")
        key_map = _key_map(m, diffusion_model)
        targets = []
        parse_skipped = []
        for spec in specs:
            parsed = parse_lora(_load_lora_file(spec.lora), key_map)
            targets.append(parsed.targets)
            parse_skipped += [(spec.lora, k, why) for k, why in parsed.skipped]

        # The token grid comes from the latent at sampling time: one token per patch_size x patch_size
        # latent cells (MOSS-Image has no patch_size attribute: one token per cell). The latent format's
        # downscale only checks that the latent is the canvas size.
        patch = getattr(diffusion_model, "patch_size", 1)
        latent_format = getattr(getattr(m, "model", None), "latent_format", None)
        downscale = getattr(latent_format, "spacial_downscale_ratio", None)
        controller = RegionalLoraController(
            specs, targets, layout, width, height, ramp_start, ramp_end, steps, strict,
            patch_size=patch if isinstance(patch, int) and patch >= 1 else 1,
            latent_downscale=downscale if isinstance(downscale, int) and downscale >= 1 else None,
        ).bind(diffusion_model)
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
        }
        log.info("[MOSS Regional LoRA] %d LoRA region(s), %d LoRA layer(s) on %d module(s), %d key(s) skipped, "
                 "%d layout item(s)", summary["regions"], summary["applied"], summary["modules"],
                 summary["skipped"], len(doc.prompt_items))
        if controller.applied_count == 0:
            msg = "none of the selected LoRAs' keys resolved to a linear module of this model"
            if strict:
                raise RuntimeError(f"[MOSS Regional LoRA] {msg}; {len(skipped)} key(s) skipped (see log)")
            log.warning("[MOSS Regional LoRA] %s", msg)
        return out(summary)

from .discovery_node import MossLayoutDiscovery  # noqa: E402

NODE_CLASS_MAPPINGS = {"MossRegionalLora": MossRegionalLora, "MossLayoutDiscovery": MossLayoutDiscovery}
NODE_DISPLAY_NAME_MAPPINGS = {"MossRegionalLora": "MOSS-Image Regional LoRA + Layout Prompt",
                              "MossLayoutDiscovery": "MOSS-Image Layout Discovery"}
