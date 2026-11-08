"""The "MOSS-Image Layout Discovery" node: runs the layout discovery inside ComfyUI (FR-12).

Queue it once with the MOSS-Image MODEL, CLIP and VAE. It shows a report and the probe images; copy
the values it prints into the Advanced group of "MOSS-Image Regional LoRA + Layout Prompt"'s Canvas panel.
Nothing is saved or changed: the node only measures and reports.
"""

from __future__ import annotations

import logging

from .discovery import Discovery, DiscoveryError, Settings, parse_edit_methods

log = logging.getLogger("moss_regional_lora")


class MossLayoutDiscovery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MOSS-Image model, straight from its loader (no other patches)."}),
                "clip": ("CLIP", {"tooltip": "The MOSS-Image text encoder."}),
                "vae": ("VAE", {"tooltip": "The MOSS-Image VAE (decodes the probe images, encodes the edit image)."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a quiet mountain lake at dawn, pine trees on the shore, soft mist",
                                      "tooltip": "Prompt for the single-patch probes and the edit pass."}),
            },
            "optional": {
                "edit_image": ("IMAGE", {"tooltip": "Any image. Connect it to also measure where an edit pipeline's "
                                                    "source-image tokens go (source_tokens)."}),
                "edit_methods": ("STRING", {"default": "default, index_timestep_zero", "advanced": True,
                                            "tooltip": "Comma-separated reference_latents_method values to test; "
                                                       "'default' means unset. index_timestep_zero is the "
                                                       "'ai-toolkit (kv_cache off)' format of Text Encode MOSS-Image Edit."}),
                "probe_strength": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 100.0, "step": 0.5, "advanced": True,
                                             "tooltip": "Size of the one-patch perturbation, relative to the mean token "
                                                        "size. Raise it if the report says the change is not clear."}),
                "patch_size": ("INT", {"default": 0, "min": 0, "max": 16, "advanced": True,
                                       "tooltip": "0 = infer. Tokens per latent cell along each axis."}),
                "blocks": ("STRING", {"default": "", "advanced": True,
                                      "tooltip": "Dotted path of the transformer block list. Empty = auto-detect."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "images")
    OUTPUT_TOOLTIPS = ("The discovery report, ending with the values to set.",
                       "prediction, probe_r0_c0, diff_r0_c0, probe_r0_c1, diff_r0_c1 (one-step predictions and "
                       "where nudging one image token changed them).")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "model_patches/moss_image"
    DESCRIPTION = ("Measures where MOSS-Image puts text, image and source-image tokens, and reports the values to "
                   "set in the Advanced group of MOSS-Image Regional LoRA + Layout Prompt's Canvas panel. Run it once; it "
                   "changes nothing.")

    def run(self, model, clip, vae, width, height, seed, prompt, edit_image=None, edit_methods="default, index_timestep_zero", probe_strength=8.0, patch_size=0, blocks=""):
        import torch
        lines = []

        def emit(line):
            lines.append(line)
            log.info("[MOSS Layout Discovery] %s", line)

        settings = Settings(width=width, height=height, patch_size=patch_size or None, prompt=prompt, seed=seed,
                            probe_strength=probe_strength, edit_methods=parse_edit_methods(edit_methods),
                            blocks=blocks.strip() or None)
        images = []
        try:
            d = Discovery(model, clip, vae, settings, emit)
            d.run_all(edit_image)
            images = [img for _, img in d.images]
        except DiscoveryError as e:
            emit(f"\nERROR: {e}")
        report = "\n".join(lines).strip()
        if images and all(i.shape == images[0].shape for i in images):
            batch = torch.stack(images)
        else:
            batch = torch.zeros(1, 64, 64, 3)
        ui = {"moss_layout_report": [report]}
        try:
            import nodes
            ui["images"] = nodes.PreviewImage().save_images(batch, "moss_layout_discovery")["ui"]["images"]
        except Exception as e:  # preview is a convenience; the IMAGE output still carries them
            log.warning("[MOSS Layout Discovery] could not save preview images: %s", e)
        return {"ui": ui, "result": (report, batch)}
