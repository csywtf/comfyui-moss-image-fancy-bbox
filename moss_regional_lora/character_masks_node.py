"""The "MOSS-Image Character Masks" ComfyUI node. ComfyUI is imported lazily (tests run without it)."""

from __future__ import annotations

import logging
import os
import uuid

from .character_masks import (DETAIL_VERSION, PARTS, CharacterMasks, Sam3Runner, compute, label_map,
                              render_preview)
from .prompt_writer import WriterSettings, fill_missing

log = logging.getLogger("moss_regional_lora")

UI_KEY = "moss_character_masks"
DEFAULT_DETAIL = '{"version": %d, "boxes": {}}' % DETAIL_VERSION


def _make_runner(sam3_model, sam3_clip, threshold, refine_iterations):
    return Sam3Runner(sam3_model, sam3_clip, threshold, refine_iterations)


def _save_temp_png(array_or_tensor, prefix):
    """Save an [H, W, 3|4] image to ComfyUI's temp folder; returns the /view reference."""
    import numpy as np
    import folder_paths
    from PIL import Image
    a = array_or_tensor
    if not isinstance(a, np.ndarray):
        a = (a.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    Image.fromarray(a).save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


class MossCharacterMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The generated image (after VAE Decode)."}),
                "regions": ("STRING", {"forceInput": True,
                                       "tooltip": "The regions output of MOSS-Image Regional LoRA + Layout Prompt."}),
                "sam3_model": ("MODEL", {"tooltip": "SAM3, from Load Checkpoint with sam3.1_multiplex_fp16.safetensors."}),
                "sam3_clip": ("CLIP", {"tooltip": "The CLIP output of the same Load Checkpoint (encodes the part names)."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "How confident SAM3 must be to accept a part. Lower finds more."}),
                "box_margin": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "How far outside each box (as a fraction of its size) to "
                                                    "look for its parts; characters rarely fill their box exactly."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "advanced": True,
                                              "tooltip": "SAM3 mask refinement passes (0 = raw detector masks)."}),
                "detail": ("STRING", {"multiline": True, "default": DEFAULT_DETAIL,
                                      "tooltip": "Per-box detail settings, edited in this node's editor."}),
            },
            "optional": {
                "prompt_llm": ("CLIP", {"tooltip": "Optional: a text model that writes the prompt of every ticked "
                                                   "part that has none, from the box's caption and a crop of the "
                                                   "part. Any Load CLIP model that can generate text: the MOSS-Image "
                                                   "text encoder (Qwen3.5) works, as do Qwen3, Qwen3-VL and Gemma."}),
                # the prompt writer's settings, edited in the editor's Settings panel
                "llm_max_length": ("INT", {"default": 160, "min": 16, "max": 2048,
                                           "tooltip": "Longest reply, in tokens."}),
                "llm_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                                              "tooltip": "Randomness (0 = always the most likely words)."}),
                "llm_top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "llm_top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "llm_min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "llm_repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "llm_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "llm_use_image": ("BOOLEAN", {"default": True, "tooltip": "Show the model a crop of each part "
                                                                         "(if it can see images)."}),
                "llm_system_prompt": ("STRING", {"multiline": True, "default": "",
                                                 "tooltip": "The prompt writer's instructions (empty: built in)."}),
            },
        }

    RETURN_TYPES = ("MOSS_CHARACTER_MASKS", "IMAGE", "MASK")
    RETURN_NAMES = ("character_masks", "preview", "mask")
    OUTPUT_TOOLTIPS = ("Each box's part masks with its caption, LoRAs and detail settings, for the detailer.",
                       "The image with each character's parts tinted by part.",
                       "Every part of every box as one mask.")
    OUTPUT_NODE = True       # shows its preview even when nothing after it runs
    FUNCTION = "run"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Finds each box's character with SAM3 and masks its face, eyes, hands, hair and skin, "
                   "as ticked per box in this node's editor. Overlapping characters keep their own pixels.")

    def run(self, image, regions, sam3_model, sam3_clip, threshold, box_margin, refine_iterations=2,
            detail=DEFAULT_DETAIL, prompt_llm=None, llm_max_length=160, llm_temperature=0.7, llm_top_p=0.95,
            llm_top_k=64, llm_min_p=0.05, llm_repetition_penalty=1.05, llm_seed=0, llm_use_image=True,
            llm_system_prompt=""):
        runner = _make_runner(sam3_model, sam3_clip, threshold, refine_iterations)
        result: CharacterMasks = compute(image, regions, detail, runner, box_margin)
        settings = WriterSettings(int(llm_max_length), float(llm_temperature), float(llm_top_p), int(llm_top_k),
                                  float(llm_min_p), float(llm_repetition_penalty), int(llm_seed),
                                  bool(llm_use_image), llm_system_prompt or "")
        written = fill_missing(result, image, prompt_llm, settings) if prompt_llm is not None else {}
        previews = [render_preview(image[i], f) for i, f in enumerate(result.frames)]
        import torch
        preview = torch.stack(previews)
        ui = {}
        try:
            f0 = result.frames[0]
            ui = {UI_KEY: [{
                "image": _save_temp_png(image[0, ..., :3].detach().cpu().float(), "moss_masks_image"),
                "labels": _save_temp_png(label_map(f0), "moss_masks_labels"),
                "width": f0.width, "height": f0.height,
                "parts": [p.name for p in PARTS],
                "boxes": [{"id": b.id, "name": b.name, "outline_found": b.outline_found, "found": b.found}
                          for b in f0.boxes],
                "llm": prompt_llm is not None,
                "generated": written,         # the editor stores these as the parts' prompts
            }]}
        except Exception as e:  # the editor's preview is a convenience; never fail the run over it
            log.warning("[MOSS Character Masks] could not save the editor preview (%s)", e)
        n = sum(len(b.parts) for f in result.frames for b in f.boxes)
        missing = [f"{b.name}: {p}" for b in result.frames[0].boxes for p, ok in b.found.items() if not ok]
        log.info("[MOSS Character Masks] %d box(es), %d part mask(s)%s", len(result.frames[0].boxes), n,
                 f"; not found: {', '.join(missing)}" if missing else "")
        return {"ui": ui, "result": (result, preview, result.union())}
