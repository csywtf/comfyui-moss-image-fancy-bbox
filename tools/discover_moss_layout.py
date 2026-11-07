#!/usr/bin/env python3
"""Establish MOSS-Image's transformer sequence layout on real hardware (FR-12).

Run this on the machine where ComfyUI and the MOSS-Image checkpoint live. It loads the
checkpoint with ComfyUI's own loaders, runs short generations with forward hooks on the
first transformer block, and prints the values to paste into
``moss_regional_lora/layout_config.py``.

Modes (``--mode``):
  shapes  FR-12b/c  observed sequence length vs. the image token count implied by the canvas
  order   FR-12d    short vs. long prompt: is text a prefix or a suffix of the sequence?
  probe   FR-12e    perturb exactly one patch's token, generate, save images, locate the change
  edit    FR-12f    with vs. without a source image: are source-image tokens in the sequence, where?
  all               shapes -> order -> probe (row 0 col 0, then row 0 col 1) -> edit (if --edit-image)

Every mode ends by printing the configuration it established (FR-12g). The same passes are available
inside ComfyUI as the "MOSS-Image Layout Discovery" node; this script is for running them without the
UI. The logic lives in moss_regional_lora/discovery.py.

Example (from the ComfyUI root, with this node in custom_nodes/):
  python custom_nodes/comfyui-moss-image-fancy-bbox/tools/discover_moss_layout.py \\
      --checkpoint moss_image.safetensors --mode all --edit-image input/example.png --out moss_probe
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, NODE_ROOT)

from moss_regional_lora.discovery import (  # noqa: E402
    Discovery, DiscoveryError, Settings, parse_edit_methods, render_config,
)


def load_models(args):
    """FR-12a: load the checkpoint with ComfyUI's own loaders."""
    comfy_root = args.comfyui or os.path.abspath(os.path.join(NODE_ROOT, "..", ".."))
    if not os.path.isfile(os.path.join(comfy_root, "comfy", "sd.py")):
        raise SystemExit(f"ComfyUI not found at {comfy_root}; pass --comfyui /path/to/ComfyUI")
    sys.path.insert(0, comfy_root)
    import folder_paths
    import comfy.sd
    import comfy.utils

    def path(folder, name):
        if os.path.isfile(name):
            return os.path.abspath(name)
        return folder_paths.get_full_path_or_raise(folder, name)

    emb = folder_paths.get_folder_paths("embeddings")
    vae = None
    if args.checkpoint:
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(path("checkpoints", args.checkpoint), embedding_directory=emb)[:3]
    else:
        if not (args.unet and args.clip):
            raise SystemExit("pass --checkpoint, or --unet and --clip (and --vae for probe/edit modes)")
        model = comfy.sd.load_diffusion_model(path("diffusion_models", args.unet))
        clip_type = getattr(comfy.sd.CLIPType, args.clip_type.upper())
        clip = comfy.sd.load_clip([path("text_encoders", args.clip)], embedding_directory=emb, clip_type=clip_type)
    if args.vae:
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path("vae", args.vae)))
    return model, clip, vae


def load_image(path):
    import numpy as np
    import torch
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(img).astype("float32") / 255.0)[None]


def save_png(t, path):
    import numpy as np
    from PIL import Image
    arr = (t.float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["shapes", "order", "probe", "edit", "all"], default="all")
    ap.add_argument("--comfyui", help="ComfyUI root (default: two levels above this node's folder)")
    ap.add_argument("--checkpoint", help="all-in-one checkpoint (models/checkpoints name or a path)")
    ap.add_argument("--unet", help="diffusion model file (models/diffusion_models name or a path)")
    ap.add_argument("--clip", help="text encoder file (models/text_encoders name or a path)")
    ap.add_argument("--clip-type", default="moss_image", help="comfy.sd.CLIPType member for --clip (default moss_image)")
    ap.add_argument("--vae", help="VAE file (models/vae name or a path)")
    ap.add_argument("--blocks", help="dotted path of the transformer block list, if auto-detection fails")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--patch-size", type=int, help="tokens per latent cell along each axis (default: infer)")
    ap.add_argument("--text-position", choices=["prefix", "suffix"], help="for --mode probe/edit when not running order")
    ap.add_argument("--prompt", default="a quiet mountain lake at dawn, pine trees on the shore, soft mist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-row", type=int, default=0)
    ap.add_argument("--probe-col", type=int, default=0)
    ap.add_argument("--probe-strength", type=float, default=8.0, help="perturbation size relative to the mean token norm")
    ap.add_argument("--edit-image", help="an image file for --mode edit")
    ap.add_argument("--edit-method", action="append",
                    help="reference_latents_method to test; repeat for several; 'default' = unset (default: default)")
    ap.add_argument("--atol", type=float, default=0.0, help="tolerance when comparing tokens across passes")
    ap.add_argument("--out", default="moss_layout_probe")
    args = ap.parse_args(argv)

    import torch
    settings = Settings(width=args.width, height=args.height, patch_size=args.patch_size,
                        text_position=args.text_position, prompt=args.prompt, seed=args.seed,
                        probe_strength=args.probe_strength, edit_methods=parse_edit_methods(",".join(args.edit_method or [])),
                        atol=args.atol, blocks=args.blocks)
    try:
        with torch.inference_mode():
            model, clip, vae = load_models(args)
            d = Discovery(model, clip, vae, settings)
            edit_image = load_image(args.edit_image) if args.edit_image else None
            if args.mode == "all":
                d.run_all(edit_image)
            else:
                if args.mode == "shapes":
                    d.mode_shapes()
                elif args.mode == "order":
                    d.mode_order()
                elif args.mode == "probe":
                    d.mode_probe([(args.probe_row, args.probe_col)])
                elif args.mode == "edit":
                    d.mode_edit(edit_image)
                print("\n=== configuration (FR-12g) ===")
                print(render_config(d.findings, edit_ran=d.edit_ran))
    except DiscoveryError as e:
        raise SystemExit(str(e))
    if d.images:
        os.makedirs(args.out, exist_ok=True)
        for name, img in d.images:
            save_png(img, os.path.join(args.out, f"{name}.png"))
        print(f"\nImages saved in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
