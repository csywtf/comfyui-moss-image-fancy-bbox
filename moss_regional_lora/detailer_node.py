"""The "MOSS-Image Regional Detailer" ComfyUI node. ComfyUI is imported lazily (tests run without it)."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

from .detailer import count_passes, detail

log = logging.getLogger("moss_regional_lora")

_RAW_LORAS: "OrderedDict[str, tuple]" = OrderedDict()
_RAW_LORAS_SIZE = 6


def _raw_lora(name: str):
    """A LoRA file as LoraLoader reads it (load_lora_for_models converts the format itself)."""
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path_or_raise("loras", name)
    st = os.stat(path)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _RAW_LORAS.get(path)
    if hit is not None and hit[0] == stamp:
        _RAW_LORAS.move_to_end(path)
        return hit[1]
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    _RAW_LORAS[path] = (stamp, sd)
    while len(_RAW_LORAS) > _RAW_LORAS_SIZE:
        _RAW_LORAS.popitem(last=False)
    return sd


def _sampler_names():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
    except Exception:  # outside ComfyUI (tests)
        return ["euler"], ["simple"]


class _Sampler:
    """Re-samples one crop with ComfyUI's VAE and sampler, with a pass's prompt and LoRAs (model only, as
    the main node applies them)."""

    def __init__(self, model, clip, vae, negative, steps, cfg, sampler_name, scheduler):
        self.model, self.clip, self.vae = model, clip, vae
        self.steps, self.cfg, self.sampler_name, self.scheduler = steps, cfg, sampler_name, scheduler
        self._models = {}
        self._conds = {}
        self.negative = negative if negative is not None else self._encode("")

    def _encode(self, text):
        if text not in self._conds:
            self._conds[text] = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(text))
        return self._conds[text]

    def _patched(self, loras):
        if loras not in self._models:
            import comfy.sd
            m = self.model
            for name, weight in loras:
                m, _ = comfy.sd.load_lora_for_models(m, None, _raw_lora(name), weight, 0)
            self._models[loras] = m
        return self._models[loras]

    def __call__(self, pixels, noise_mask, p, seed):
        import nodes
        latent = {"samples": self.vae.encode(pixels[None, ..., :3]),
                  "noise_mask": noise_mask.reshape(1, 1, *noise_mask.shape)}
        out = nodes.common_ksampler(self._patched(p.loras), seed, self.steps, self.cfg, self.sampler_name,
                                    self.scheduler, self._encode(p.prompt), self.negative, latent,
                                    denoise=p.denoise)[0]
        img = self.vae.decode(out["samples"])
        img = img.reshape(-1, *img.shape[-3:])        # video-style VAEs return [B, T, H, W, C]
        return img[0]


class MossRegionalDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _sampler_names()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to detail (the one the masks were made from)."}),
                "character_masks": ("MOSS_CHARACTER_MASKS", {"tooltip": "From MOSS-Image Character Masks."}),
                "model": ("MODEL", {"tooltip": "The MOSS-Image model straight from the loader (or after ordinary "
                                               "LoRA loaders), NOT the Regional LoRA node's output: each box's "
                                               "LoRAs are applied here, to its own crops."}),
                "clip": ("CLIP", {"tooltip": "The MOSS-Image text encoder, for the part prompts."}),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000,
                                  "tooltip": "As on a KSampler; with denoise 0.35 about a third of them run."}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Use the same CFG as your KSampler."}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "How much each part is redrawn, unless the part has its own "
                                                 "denoise in the Character Masks editor."}),
                "crop_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16,
                                      "tooltip": "Each part's crop is sampled at about this size on its long side."}),
                "padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Context around each part, as a fraction of its size."}),
                "mask_grow": ("INT", {"default": 6, "min": 0, "max": 128,
                                      "tooltip": "Grow each part's mask by this many pixels before redrawing."}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 256,
                                    "tooltip": "Soften the edge where the redrawn part meets the image (pixels)."}),
                "stop_before_detailing": ("BOOLEAN", {"default": False, "label_on": "stop here",
                                                      "label_off": "detail",
                                                      "tooltip": "Stop here (nothing after this node runs) so you "
                                                                 "can check the masks first. Untick and queue again: "
                                                                 "the image and masks come from ComfyUI's cache and "
                                                                 "only this node runs."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Your negative prompt (empty if not connected)."}),
                "max_upscale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 0.5, "advanced": True,
                                          "tooltip": "Small parts (eyes) are upscaled at most this much; their crop "
                                                     "takes more context instead."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("The image with every masked part redrawn with its character's prompt and LoRAs.",)
    FUNCTION = "run"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Redraws each character's face, eyes, hands, hair and skin (as masked by MOSS-Image Character "
                   "Masks) with that character's caption and LoRAs, one part at a time.")

    def run(self, image, character_masks, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, denoise,
            crop_size, padding, mask_grow, feather, stop_before_detailing, negative=None, max_upscale=4.0):
        if stop_before_detailing:
            from comfy_execution.graph_utils import ExecutionBlocker
            log.info("[MOSS Regional Detailer] stopped before detailing (untick 'stop_before_detailing' to run)")
            return (ExecutionBlocker(None),)       # silently skips everything after this node
        total = count_passes(character_masks, denoise)
        if total == 0:
            log.info("[MOSS Regional Detailer] no part masks to detail; image passed through")
            return (image,)
        import comfy.utils
        bar = comfy.utils.ProgressBar(total)
        sampler = _Sampler(model, clip, vae, negative, steps, cfg, sampler_name, scheduler)

        def on_pass(k, p):
            log.info("[MOSS Regional Detailer] %d/%d: %s %s (denoise %.2f)", k + 1, total, p.box, p.part, p.denoise)
            bar.update(1)

        out = detail(image, character_masks, sampler, denoise=denoise, crop_size=crop_size, padding=padding,
                     max_upscale=max_upscale, grow=mask_grow, feather=feather, seed=seed, on_pass=on_pass)
        return (out.to(image.device, image.dtype),)
