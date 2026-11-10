"""The "MOSS-Image Regional Detailer" ComfyUI node. ComfyUI is imported lazily (tests run without it)."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

from .detailer import count_passes, describe_pass, detail, missing_prompts, needs_reference, plan, stack_previews

log = logging.getLogger("moss_regional_lora")

_RAW_LORAS: "OrderedDict[str, tuple]" = OrderedDict()
_RAW_LORAS_SIZE = 6


def _raw_lora(name: str):
    """A LoRA file as LoraLoader reads it (converted to ComfyUI's format when it is applied)."""
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
    """Re-samples one crop with ComfyUI's VAE and sampler, with a pass's prompt and LoRAs.

    LoRAs are applied to the model only, as the main node applies them, and matched to its layers with the
    main node's key map (ComfyUI's own plus the naming styles it accepts, such as ``transformer.``), so a
    LoRA that works in the main node works here too; patching itself is ComfyUI's (load_lora,
    add_patches), as LoraLoader does."""

    def __init__(self, model, clip, vae, negative, steps, cfg, sampler_name, scheduler, reference=True):
        self.model, self.clip, self.vae = model, clip, vae
        self.reference = reference
        self.steps, self.cfg, self.sampler_name, self.scheduler = steps, cfg, sampler_name, scheduler
        self._models = {}
        self._conds = {}
        self._km = None
        self.negative = negative if negative is not None else self._encode("")

    def _encode(self, text):
        if text not in self._conds:
            self._conds[text] = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(text))
        return self._conds[text]

    def _key_map(self):
        if self._km is None:
            from .node import _key_map
            self._km = _key_map(self.model, self.model.get_model_object("diffusion_model"))
        return self._km

    def _patched(self, loras):
        if loras not in self._models:
            if not loras:
                self._models[loras] = self.model
            else:
                import comfy.lora
                import comfy.lora_convert
                m = self.model.clone()
                for name, weight in loras:
                    patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(_raw_lora(name)), self._key_map())
                    if not patches:
                        log.warning("[MOSS Regional Detailer] no layer of the model matched %s; it has no effect", name)
                    else:
                        log.info("[MOSS Regional Detailer] LoRA %s at %.2f: %d layers patched", name, weight,
                                 len(patches))
                    m.add_patches(patches, weight)
                self._models[loras] = m
        return self._models[loras]

    def __call__(self, pixels, noise_mask, p, seed):
        import nodes
        encoded = self.vae.encode(pixels[None, ..., :3])
        latent = {"samples": encoded, "noise_mask": noise_mask.reshape(1, 1, *noise_mask.shape)}
        out = nodes.common_ksampler(self._patched(p.loras), seed, self.steps, self.cfg, self.sampler_name,
                                    self.scheduler, self._encode(p.prompt), self.negative, latent,
                                    denoise=p.denoise)[0]
        # the same crop decoded without redrawing: what the VAE round trip alone gives, for restoring detail
        return self._decode(out["samples"]), (self._decode(encoded) if self.reference else None)

    def _decode(self, samples):
        img = self.vae.decode(samples)
        return img.reshape(-1, *img.shape[-3:])[0]     # video-style VAEs return [B, T, H, W, C]


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
                "max_upscale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 16.0, "step": 1.0, "advanced": True,
                                          "tooltip": "Small parts (eyes) are upscaled at most this much (a whole "
                                                     "number); their crop takes more context instead."}),
                "restore_detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True,
                                             "tooltip": "Put back the fine detail the crop's round trip (resizing, "
                                                        "VAE) loses, keeping only the model's changes. 1: denoise 0 "
                                                        "returns the image unchanged. Lower it if a high denoise "
                                                        "leaves faint traces of the old features."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "passes")
    OUTPUT_TOOLTIPS = ("The image with every masked part redrawn with its character's prompt and LoRAs.",
                       "One image per pass, in the order of the log: the crop as it went into the sampler (left) "
                       "and as the model redrew it (right), before it is blended back. Connect a Preview Image "
                       "to see what each pass's prompt and LoRAs actually did.")
    FUNCTION = "run"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Redraws each character's face, eyes, hands, hair and skin (as masked by MOSS-Image Character "
                   "Masks) with that character's caption and LoRAs, one part at a time.")

    def run(self, image, character_masks, model, clip, vae, seed, steps, cfg, sampler_name, scheduler, denoise,
            crop_size, padding, mask_grow, feather, stop_before_detailing, negative=None, max_upscale=4.0,
            restore_detail=1.0):
        if stop_before_detailing:
            from comfy_execution.graph_utils import ExecutionBlocker
            log.info("[MOSS Regional Detailer] stopped before detailing (untick 'stop_before_detailing' to run)")
            return (ExecutionBlocker(None), ExecutionBlocker(None))   # silently skips everything after this node
        missing = missing_prompts(list(plan(character_masks, denoise, padding, crop_size / max_upscale,
                                            mask_grow + feather)))
        if missing:
            raise ValueError("MOSS Regional Detailer: these parts have no prompt: " + "; ".join(missing)
                             + ". Write each one in the Character Masks editor (the part's edit button), or untick "
                             "the part. Boxes are named by their place in the box list (region 1 = the first).")
        total = count_passes(character_masks, denoise, padding, crop_size / max_upscale, mask_grow + feather)
        if total == 0:
            log.info("[MOSS Regional Detailer] no part masks to detail; image passed through")
            return (image, stack_previews([]))
        import comfy.utils
        bar = comfy.utils.ProgressBar(total)
        sampler = _Sampler(model, clip, vae, negative, steps, cfg, sampler_name, scheduler,
                           reference=needs_reference(character_masks, denoise, restore_detail))

        def on_pass(k, p):
            log.info("[MOSS Regional Detailer] %d/%d: %s", k + 1, total, describe_pass(p))
            bar.update(1)

        previews = []
        out = detail(image, character_masks, sampler, denoise=denoise, crop_size=crop_size, padding=padding,
                     max_upscale=max_upscale, grow=mask_grow, feather=feather, seed=seed, on_pass=on_pass,
                     restore=restore_detail, keep=previews)
        return out.to(image.device, image.dtype), stack_previews(previews).to(image.device, image.dtype)
