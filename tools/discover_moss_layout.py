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

Every mode ends by printing the configuration it established (FR-12g).

Example (from the ComfyUI root, with this node in custom_nodes/):
  python custom_nodes/comfyui-moss-image-fancy-bbox/tools/discover_moss_layout.py \\
      --checkpoint moss_image.safetensors --mode all --edit-image input/example.png --out moss_probe
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_ROOT = os.path.dirname(HERE)
sys.path.insert(0, NODE_ROOT)

SHORT_PROMPT = "a red apple on a wooden table"
LONG_PROMPT = (
    "A wide, highly detailed photograph of a busy harbour at golden hour. Fishing boats with peeling blue and "
    "white paint are tied along a weathered stone quay, their nets piled in tangled heaps beside crates of ice. "
    "Gulls wheel overhead against a sky streaked with orange and violet clouds, and the low sun throws long "
    "shadows across wet cobblestones. In the foreground an old fisherman in a yellow oilskin coat mends a net, "
    "his weathered hands moving quickly, while a small brown dog sleeps beside a coil of rope. Behind him a row "
    "of narrow houses painted ochre, teal and rose climbs the hillside, laundry strung between their windows. "
    "A lighthouse stands at the end of the breakwater, its lamp just beginning to glow. Reflections of the boats "
    "ripple in the green water, broken by the wake of a returning trawler. Crisp focus, natural colour, fine "
    "texture on wood, rope, stone and fabric, gentle film grain, realistic lighting, documentary style. "
) * 2


# ============================================================================ pure analysis (unit-tested)
def identical_runs(a, b, atol: float = 0.0) -> Tuple[int, int]:
    """Leading and trailing token runs that are identical between sequences a [L1, D] and b [L2, D]."""
    import torch
    n = min(a.shape[0], b.shape[0])

    def same(x, y):
        return bool(torch.equal(x, y)) if atol == 0 else bool(torch.allclose(x, y, rtol=0, atol=atol))

    prefix = 0
    while prefix < n and same(a[prefix], b[prefix]):
        prefix += 1
    suffix = 0
    while suffix < n and same(a[a.shape[0] - 1 - suffix], b[b.shape[0] - 1 - suffix]):
        suffix += 1
    return prefix, suffix


def decide_text_position(len_a: int, len_b: int, ctx_a: int, ctx_b: int, prefix_run: int, suffix_run: int,
                         n_image: int) -> Tuple[Optional[str], List[str]]:
    """Two passes, same seed, different prompt lengths. Image tokens do not depend on the prompt before
    the first block, so the side of the sequence that stays identical is the image side."""
    notes = []
    d_seq, d_ctx = len_b - len_a, ctx_b - ctx_a
    if d_ctx == 0:
        return None, ["the two prompts produced the same context length; nothing to compare"]
    if d_seq != d_ctx:
        notes.append(f"sequence length changed by {d_seq} but the text context changed by {d_ctx}: text tokens do "
                     "not enter the sequence one-for-one. This node's layout model does not fit; send this output.")
        return None, notes
    notes.append(f"sequence length delta ({d_seq}) equals the prompt token delta ({d_ctx}): every text token is in the sequence")
    tail_ok, head_ok = suffix_run >= n_image, prefix_run >= n_image
    if tail_ok and not head_ok:
        notes.append(f"the last {suffix_run} tokens are identical across prompts (>= {n_image} image tokens) and the "
                     f"first {prefix_run} are not: the image block is at the END, text is a PREFIX")
        return "prefix", notes
    if head_ok and not tail_ok:
        notes.append(f"the first {prefix_run} tokens are identical across prompts (>= {n_image} image tokens) and the "
                     f"last {suffix_run} are not: the image block is at the START, text is a SUFFIX")
        return "suffix", notes
    notes.append(f"inconclusive: identical leading run {prefix_run}, trailing run {suffix_run}, image tokens {n_image}")
    return None, notes


def decide_source_position(len_without: int, len_with: int, expected_src: int, prefix_run: int, suffix_run: int,
                           n_image: int, n_text: int, text_position: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Same prompt and seed with and without a source image."""
    delta = len_with - len_without
    if delta == 0:
        return "none", ["sequence length is unchanged with a source image: source tokens are not in the sequence"]
    notes = [f"a source image added {delta} tokens (patch-grid count of the source latent: {expected_src})"]
    if delta != expected_src:
        notes.append("the added count does not match the source latent's patch grid; this node cannot count "
                     "source tokens for this model. Send this output.")
        return None, notes
    insert_at = prefix_run
    if prefix_run + suffix_run < len_without:
        notes.append(f"identical runs ({prefix_run} leading + {suffix_run} trailing) do not cover the "
                     f"{len_without}-token sequence; inconclusive")
        return None, notes
    if text_position is None:
        notes.append(f"source tokens inserted at index {insert_at}, but the text position is unknown; run --mode order")
        return None, notes
    first_len = n_text if text_position == "prefix" else n_image
    if insert_at == 0:
        pos = "start"
    elif insert_at == first_len:
        pos = "between"
    elif insert_at == n_text + n_image:
        pos = "end"
    else:
        notes.append(f"source tokens inserted at index {insert_at}, which is not at the start, between text and "
                     "image, or at the end; not supported")
        return None, notes
    notes.append(f"source tokens are inserted at index {insert_at}: SOURCE_TOKENS = {pos!r}")
    return pos, notes


def combine_source_positions(found: Sequence[Optional[str]]) -> Tuple[Optional[str], str]:
    found = list(found)
    if not found or any(f is None for f in found):
        return None, "at least one edit pass was inconclusive"
    in_seq = sorted({f for f in found if f != "none"})
    if not in_seq:
        return "none", "no tested edit method puts source tokens in the sequence"
    if len(in_seq) == 1:
        return in_seq[0], "the node accepts both the with- and without-source sequence lengths"
    return None, f"different edit methods place source tokens differently ({in_seq}); not representable"


def interpret_probe(found_rc: Tuple[int, int], probe_rc: Tuple[int, int], rows: int, cols: int) -> Tuple[Optional[bool], str]:
    """Where did a perturbation of image token (row * cols + col) land?"""
    fr, fc = found_rc
    r, c = probe_rc
    if (fr, fc) == (r, c):
        if r == c:
            return True, (f"the change landed on patch ({fr}, {fc}) as raster order predicts. This patch is on the "
                          "diagonal, so a transposed (column-major) grid would land here too: also run "
                          "--probe-row 0 --probe-col 1.")
        return True, f"the change landed on patch ({fr}, {fc}), exactly where raster order (index = row * cols + col) puts it"
    k = r * cols + c
    if (fr, fc) == (k % rows, k // rows):
        return False, "the change landed where a COLUMN-MAJOR (transposed) grid puts it: image tokens are not raster order"
    if (fr, fc) == (r, cols - 1 - c):
        return False, "the change landed mirrored left-right: columns run right-to-left"
    if (fr, fc) == (rows - 1 - r, c):
        return False, "the change landed mirrored top-bottom: rows run bottom-to-top"
    return False, (f"the change landed on patch ({fr}, {fc}) instead of ({r}, {c}): a tiled, windowed or otherwise "
                   "non-raster token order, or the layout used for the probe is wrong. Raster order is not confirmed.")


def strongest_cell(diff, cell: int) -> Tuple[Tuple[int, int], float, float]:
    """diff [H, W] -> ((row, col) of the cell with the largest mean change, its mean, the median cell mean)."""
    H, W = diff.shape
    rows, cols = H // cell, W // cell
    cells = diff[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).mean(dim=(1, 3))
    idx = int(cells.argmax())
    return (idx // cols, idx % cols), float(cells.max()), float(cells.median())


@dataclass
class Findings:
    vae_downscale: Optional[int] = None
    patch_size: Optional[int] = None
    text_position: Optional[str] = None
    image_order: Optional[str] = None
    source_tokens: Optional[str] = None
    source_note: str = ""
    problems: List[str] = field(default_factory=list)


def render_config(f: Findings) -> str:
    def val(v, why):
        return repr(v) if v is not None else f"None  # NOT ESTABLISHED: {why}"
    src = f.source_tokens
    src_line = val(src, "run --mode edit --edit-image <file>") + (f"  # {f.source_note}" if src and f.source_note else "")
    lines = [
        "# ---- paste into moss_regional_lora/layout_config.py ----",
        f"TEXT_POSITION = {val(f.text_position, 'run --mode order')}",
        f"IMAGE_ORDER = {val(f.image_order, 'run --mode probe (both probes must confirm raster)')}",
        f"SOURCE_TOKENS = {src_line}",
        "",
        "# ---- node inputs (MOSS-Image Regional LoRA) ----",
        f"vae_downscale = {val(f.vae_downscale, 'run --mode shapes')}",
        f"patch_size = {val(f.patch_size, 'run --mode shapes')}",
    ]
    if f.problems:
        lines += ["", "# ---- problems (send this whole output back) ----"] + [f"# - {p}" for p in f.problems]
    return "\n".join(lines)


# ============================================================================ ComfyUI-facing code
def find_blocks(dm, override: Optional[str]):
    """FR-12b: try `blocks`, then `double_blocks`, then report what the module tree contains."""
    import torch.nn as nn
    if override:
        m = dm
        for part in override.split("."):
            m = m[int(part)] if part.isdigit() else getattr(m, part)
        print(f"[blocks] using --blocks {override}: {type(m).__name__} with {len(m)} entries")
        return override, m
    for name in ("blocks", "double_blocks"):
        m = getattr(dm, name, None)
        if isinstance(m, nn.ModuleList) and len(m) > 0:
            print(f"[blocks] found '{name}' ({len(m)} x {type(m[0]).__name__})")
            return name, m
    lists = [(n, m) for n, m in dm.named_modules() if isinstance(m, nn.ModuleList) and len(m) > 0]
    print("[blocks] no 'blocks' or 'double_blocks' attribute on the diffusion model. ModuleLists found:")
    for n, m in lists:
        print(f"           {n:<40} {len(m):>4} x {type(m[0]).__name__}")
    direct = [(n, m) for n, m in lists if "." not in n]
    if len(direct) == 1:
        n, m = direct[0]
        print(f"[blocks] using '{n}': it is the only ModuleList directly on the diffusion model "
              "(nested lists such as a text embedder's blocks see only part of the sequence). "
              "Override with --blocks if this is wrong.")
        return n, m
    raise SystemExit("[blocks] cannot choose a block list automatically; re-run with --blocks <path> from the list above")


class Recorder:
    """Forward pre-hooks on the diffusion model and on its first block.

    The block can run more than once per model call (MOSS-Image, for example, runs the blocks over
    source-image tokens alone to precompute their keys/values), so each block call is tagged with the
    model call it belongs to, and the *main* pass of a model call is its longest block input.
    """

    def __init__(self, dm, block, probe=None):
        self.forwards: List[dict] = []      # one entry per diffusion-model call
        self.block_calls: List[dict] = []
        self.probe = probe                  # callable(x) -> x, applied to every block-0 input
        self.h1 = dm.register_forward_pre_hook(self._model_hook, with_kwargs=True)
        self.h2 = block.register_forward_pre_hook(self._block_hook, with_kwargs=True)

    def _model_hook(self, module, args, kwargs):
        x = args[0]
        ctx = kwargs.get("context", args[2] if len(args) > 2 else None)
        refs = kwargs.get("ref_latents") or []
        self.forwards.append({"latent": tuple(x.shape), "context": tuple(ctx.shape) if ctx is not None else None,
                              "refs": [tuple(r.shape) for r in refs], "ref_kv": kwargs.get("ref_kv") is not None})

    def _block_hook(self, module, args, kwargs):
        x = args[0] if args else kwargs.get("x", kwargs.get("img"))
        fwd = len(self.forwards) - 1        # -1: outside any model call (e.g. conditioning precompute)
        self.block_calls.append({"forward": fwd, "shape": tuple(x.shape),
                                 "x": x.detach().float().cpu() if fwd == 0 else None})
        if self.probe is not None:
            x = self.probe(x)
            if args:
                return (x,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["x" if "x" in kwargs else "img"] = x
            return args, kwargs
        return None

    def main_call(self, forward: int = 0) -> dict:
        calls = [c for c in self.block_calls if c["forward"] == forward]
        if not calls:
            raise SystemExit("[hooks] the first block never ran inside a model call; check --blocks")
        return max(calls, key=lambda c: c["shape"][1])

    def remove(self):
        self.h1.remove()
        self.h2.remove()


class Runner:
    def __init__(self, args):
        self.args = args
        comfy_root = args.comfyui or os.path.abspath(os.path.join(NODE_ROOT, "..", ".."))
        if not os.path.isfile(os.path.join(comfy_root, "comfy", "sd.py")):
            raise SystemExit(f"ComfyUI not found at {comfy_root}; pass --comfyui /path/to/ComfyUI")
        sys.path.insert(0, comfy_root)
        import folder_paths  # noqa: F401  (registers model folders)
        import comfy.sd
        import comfy.utils
        self.folder_paths = folder_paths
        emb = folder_paths.get_folder_paths("embeddings")
        if args.checkpoint:  # FR-12a: ComfyUI's own loaders
            path = self._path("checkpoints", args.checkpoint)
            self.model, self.clip, self.vae, _ = comfy.sd.load_checkpoint_guess_config(path, embedding_directory=emb)[:4]
        else:
            if not (args.unet and args.clip):
                raise SystemExit("pass --checkpoint, or --unet and --clip (and --vae for probe/edit modes)")
            self.model = comfy.sd.load_diffusion_model(self._path("diffusion_models", args.unet))
            clip_type = getattr(comfy.sd.CLIPType, args.clip_type.upper())
            self.clip = comfy.sd.load_clip([self._path("text_encoders", args.clip)], embedding_directory=emb, clip_type=clip_type)
            self.vae = None
        if args.vae:
            self.vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(self._path("vae", args.vae)))
        self.dm = self.model.get_model_object("diffusion_model")
        lf = self.model.get_model_object("latent_format")
        self.vae_downscale = int(getattr(lf, "spacial_downscale_ratio", 8))
        self.latent_channels = int(getattr(lf, "latent_channels", 4))
        print(f"[model] {type(self.dm).__name__}; latent format {type(lf).__name__}: "
              f"{self.latent_channels} channels, {self.vae_downscale} px per latent cell")
        self.block_path, self.blocks = find_blocks(self.dm, args.blocks)
        self.findings = Findings(vae_downscale=self.vae_downscale)
        self.n_text_seen = None

    def _path(self, folder, name):
        if os.path.isfile(name):
            return os.path.abspath(name)
        return self.folder_paths.get_full_path_or_raise(folder, name)

    def encode(self, prompt, refs=None, method=None):
        tokens = self.clip.tokenize(prompt)
        cond = self.clip.encode_from_tokens_scheduled(tokens)
        if refs:
            import node_helpers
            values = {"reference_latents": refs}
            if method:
                values["reference_latents_method"] = method
            cond = node_helpers.conditioning_set_values(cond, values)
        return cond

    def latent(self):
        import torch
        a = self.args
        return torch.zeros(1, self.latent_channels, a.height // self.vae_downscale, a.width // self.vae_downscale)

    def generate(self, prompt, steps, cfg, probe=None, refs=None, method=None):
        import comfy.sample
        lat = self.latent()
        lat = comfy.sample.fix_empty_latent_channels(self.model, lat)
        noise = comfy.sample.prepare_noise(lat, self.args.seed)
        pos = self.encode(prompt, refs, method)
        neg = self.encode("", refs, method)
        rec = Recorder(self.dm, self.blocks[0], probe)
        try:
            samples = comfy.sample.sample(self.model, noise, steps, cfg, self.args.sampler, self.args.scheduler,
                                          pos, neg, lat, denoise=1.0, seed=self.args.seed)
        finally:
            rec.remove()
        return samples, rec

    def grid(self, patch):
        a = self.args
        cell = self.vae_downscale * patch
        return a.height // cell, a.width // cell

    # ------------------------------------------------------------------ modes
    def mode_shapes(self):
        print("\n=== shapes (FR-12b, FR-12c) ===")
        _, rec = self.generate(SHORT_PROMPT, 1, 1.0)
        main, fwd = rec.main_call(), rec.forwards[0]
        L = main["shape"][1]
        n_ctx = fwd["context"][1] if fwd["context"] else 0
        lh, lw = fwd["latent"][-2:]
        print(f"[shapes] first block ({self.block_path}[0]) input shape {main['shape']}  -> sequence length {L}")
        print(f"[shapes] latent {fwd['latent']}, text context {fwd['context']}")
        candidates = [self.args.patch_size] if self.args.patch_size else [1, 2, 4]
        match = None
        for p in candidates:
            if lh % p or lw % p:
                continue
            n_img = (lh // p) * (lw // p)
            implied = L - n_img
            ok = implied == n_ctx
            print(f"[shapes] canvas {self.args.width}x{self.args.height}, vae_downscale {self.vae_downscale}, patch {p}: "
                  f"expected image tokens {n_img}; observed length {L}; implied text tokens {implied}; "
                  f"text context length {n_ctx} -> {'MATCH' if ok else 'no match'}")
            if ok and match is None:
                match = p
        if match is None:
            self.findings.problems.append(f"no patch size in {candidates} makes (sequence - image tokens) equal the "
                                          f"text context length ({L} vs context {n_ctx})")
        else:
            self.findings.patch_size = match
            self.n_text_seen = n_ctx
        return match

    def mode_order(self):
        print("\n=== order (FR-12d) ===")
        p = self.findings.patch_size or self.args.patch_size or self.mode_shapes()
        if p is None:
            print("[order] patch size unknown; cannot compute the image token count")
            return None
        rows, cols = self.grid(p)
        _, ra = self.generate(SHORT_PROMPT, 1, 1.0)
        _, rb = self.generate(LONG_PROMPT, 1, 1.0)
        ca, cb = ra.main_call(), rb.main_call()
        la, lb = ca["shape"][1], cb["shape"][1]
        xa, xb = ca["x"][0], cb["x"][0]
        na, nb = ra.forwards[0]["context"][1], rb.forwards[0]["context"][1]
        print(f"[order] short prompt: {na} text tokens, sequence {la}; long prompt: {nb} text tokens, sequence {lb}")
        pre, suf = identical_runs(xa, xb, atol=self.args.atol)
        print(f"[order] identical leading tokens {pre}, identical trailing tokens {suf} (image tokens {rows * cols})")
        pos, notes = decide_text_position(la, lb, na, nb, pre, suf, rows * cols)
        for n in notes:
            print(f"[order] {n}")
        if pos is None:
            self.findings.problems.append("text position: " + notes[-1])
        self.findings.text_position = pos
        return pos

    def probe_once(self, row, col, text_position, p):
        import torch
        from moss_regional_lora.geometry import canvas_to_centered, coverage_grid, raster_flatten
        from moss_regional_lora.layout import Layout
        a = self.args
        rows, cols = self.grid(p)
        cell = self.vae_downscale * p
        box = canvas_to_centered(col * cell, row * cell, (col + 1) * cell, (row + 1) * cell, a.width, a.height)
        img_mask = raster_flatten(coverage_grid(box, a.width, a.height, self.vae_downscale, p))
        assert sum(img_mask) == 1.0 and img_mask[row * cols + col] == 1.0
        layout = Layout(text_position, "raster", "none")
        gen = torch.Generator().manual_seed(1234)

        def probe(x):
            L, D = x.shape[1], x.shape[2]
            n_text = L - rows * cols
            seq = layout.sequence_mask(img_mask, L, n_text, 0)
            if seq is None:
                return x
            m = torch.tensor(seq, device=x.device, dtype=x.dtype)[None, :, None]
            v = torch.randn(D, generator=gen).to(x.device, x.dtype)
            v = v / v.norm() * x.float().norm(dim=-1).mean().to(x.dtype) * a.probe_strength
            return x + m * v

        base, _ = self.generate(a.prompt, a.steps, a.cfg)
        probed, _ = self.generate(a.prompt, a.steps, a.cfg, probe=probe)
        if self.vae is None:
            raise SystemExit("probe mode needs a VAE: pass --vae or use a checkpoint that contains one")
        ib, ip = self.vae.decode(base), self.vae.decode(probed)
        diff = (ip - ib).abs().mean(dim=-1)[0].float().cpu()
        (fr, fc), peak, median = strongest_cell(diff, cell)
        out = a.out
        os.makedirs(out, exist_ok=True)
        save_png(ib[0], os.path.join(out, "baseline.png"))
        save_png(ip[0], os.path.join(out, f"probe_r{row}_c{col}.png"))
        save_png(heat(diff, cell, (row, col)), os.path.join(out, f"diff_r{row}_c{col}.png"))
        print(f"[probe] perturbed image token {row * cols + col} (patch row {row}, col {col}; {cell}px cells, "
              f"grid {rows}x{cols}). Largest change: cell row {fr}, col {fc} (mean |diff| {peak:.4f}, "
              f"median cell {median:.4f}). Images in {os.path.abspath(out)}")
        if peak < 4 * max(median, 1e-6):
            print("[probe] WARNING: the strongest cell is not clearly above the rest; raise --probe-strength")
        ok, why = interpret_probe((fr, fc), (row, col), rows, cols)
        print(f"[probe] {why}")
        return ok, why

    def mode_probe(self, cells: Sequence[Tuple[int, int]]):
        print("\n=== probe (FR-12e) ===")
        p = self.findings.patch_size or self.args.patch_size or self.mode_shapes()
        pos = self.findings.text_position or self.args.text_position or self.mode_order()
        if p is None or pos is None:
            print("[probe] need the patch size and the text position first")
            return None
        results = [self.probe_once(r, c, pos, p) for r, c in cells]
        confirmed = all(ok for ok, _ in results) and any(r != c for r, c in cells)
        if confirmed:
            self.findings.image_order = "raster"
        else:
            if all(ok for ok, _ in results):
                print("[probe] raster order is consistent but not yet confirmed: probe an off-diagonal patch "
                      "(--probe-row 0 --probe-col 1) to rule out a transposed grid")
            else:
                self.findings.problems.append("image order: " + next(w for ok, w in results if not ok))
        return confirmed

    def mode_edit(self):
        print("\n=== edit (FR-12f) ===")
        import numpy as np
        import torch
        from PIL import Image
        a = self.args
        if not a.edit_image:
            print("[edit] no --edit-image given; skipped")
            return None
        if self.vae is None:
            raise SystemExit("edit mode needs a VAE: pass --vae or use a checkpoint that contains one")
        p = self.findings.patch_size or self.args.patch_size or self.mode_shapes()
        pos = self.findings.text_position or self.args.text_position
        img = Image.open(a.edit_image).convert("RGB").resize((a.width, a.height))
        t = torch.from_numpy(np.asarray(img).astype("float32") / 255.0)[None]
        ref = self.vae.encode(t)
        rows, cols = self.grid(p)
        expected_src = (ref.shape[-2] // p) * (ref.shape[-1] // p)
        _, r0 = self.generate(a.prompt, 1, 1.0)
        c0 = r0.main_call()
        results = []
        for method in (a.edit_method or [None]):
            _, r1 = self.generate(a.prompt, 1, 1.0, refs=[ref], method=method)
            c1, f1 = r1.main_call(), r1.forwards[0]
            l0, l1 = c0["shape"][1], c1["shape"][1]
            pre, suf = identical_runs(c0["x"][0], c1["x"][0], atol=a.atol)
            n_text = r0.forwards[0]["context"][1]
            print(f"[edit] reference_latents_method={method!r}: sequence {l0} -> {l1}; model received "
                  f"ref_latents {f1['refs']}, precomputed ref K/V: {f1['ref_kv']}; identical leading {pre}, trailing {suf}")
            found, notes = decide_source_position(l0, l1, expected_src, pre, suf, rows * cols, n_text, pos)
            for n in notes:
                print(f"[edit]   {n}")
            results.append(found)
        combined, why = combine_source_positions(results)
        print(f"[edit] SOURCE_TOKENS -> {combined!r} ({why})")
        if combined is None:
            self.findings.problems.append("source tokens: " + why)
        self.findings.source_tokens = combined
        self.findings.source_note = why
        return combined


def save_png(t, path):
    import numpy as np
    from PIL import Image
    arr = (t.float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def heat(diff, cell, probe_rc):
    """Amplified |diff| as an RGB image with the patch grid drawn and the probed patch outlined."""
    import torch
    d = diff / max(float(diff.max()), 1e-8)
    img = torch.stack([d, d * 0.3, d * 0.1], dim=-1)
    img[::cell, :, :] = 0.25
    img[:, ::cell, :] = 0.25
    r, c = probe_rc
    y0, x0 = r * cell, c * cell
    img[y0:y0 + cell, x0, 1] = img[y0:y0 + cell, min(x0 + cell - 1, img.shape[1] - 1), 1] = 1.0
    img[y0, x0:x0 + cell, 1] = img[min(y0 + cell - 1, img.shape[0] - 1), x0:x0 + cell, 1] = 1.0
    return img


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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-row", type=int, default=0)
    ap.add_argument("--probe-col", type=int, default=0)
    ap.add_argument("--probe-strength", type=float, default=8.0, help="perturbation size relative to the mean token norm")
    ap.add_argument("--edit-image", help="an image file for --mode edit")
    ap.add_argument("--edit-method", action="append",
                    help="reference_latents_method your edit workflow sets; repeat to test several (default: unset)")
    ap.add_argument("--atol", type=float, default=0.0, help="tolerance when comparing tokens across passes")
    ap.add_argument("--out", default="moss_layout_probe")
    args = ap.parse_args(argv)

    import torch
    with torch.inference_mode():
        r = Runner(args)
        if args.mode in ("shapes", "all"):
            r.mode_shapes()
        if args.mode in ("order", "all"):
            r.mode_order()
        if args.mode == "probe":
            r.mode_probe([(args.probe_row, args.probe_col)])
        if args.mode == "all":
            r.mode_probe([(0, 0), (0, 1)])
        if args.mode in ("edit", "all"):
            r.mode_edit()
    print("\n=== configuration (FR-12g) ===")
    if args.mode == "all" and r.findings.source_tokens is None and not args.edit_image:
        r.findings.source_note = ""
    print(render_config(r.findings))
    if args.mode == "all" and not args.edit_image:
        print("\n# SOURCE_TOKENS was not established because no --edit-image was given. For text-to-image only,\n"
              "# SOURCE_TOKENS = 'none' is exact (there are no source tokens). Before using an edit workflow,\n"
              "# re-run with --mode edit --edit-image <file> --edit-method <the method your edit node sets>.")


if __name__ == "__main__":
    main()
