"""Layout discovery (FR-12): measure where MOSS-Image puts text, image and source-image tokens.

Shared by the "MOSS-Image Layout Discovery" node (discovery_node.py) and the command-line tool
tools/discover_moss_layout.py. Everything here runs inside a ComfyUI process with a loaded MODEL,
CLIP and VAE; the analysis functions at the top are pure and unit-tested.

Passes, all with the same seed:
  shapes  first-block input length vs. the image token count implied by the canvas (FR-12b/c)
  order   short vs. long prompt: which end of the sequence stays identical is the image end (FR-12d)
  probe   nudge exactly one patch's token, run one step, locate the change in the prediction (FR-12e)
  edit    with vs. without a source image: are source tokens in the sequence, and where (FR-12f)
then the values to set are rendered (FR-12g).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple


class DiscoveryError(RuntimeError):
    pass


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


def find_block(seq, block, candidates: Sequence[int], atol: float = 0.0) -> Optional[int]:
    """First offset in ``candidates`` where ``block`` [N, D] occurs exactly (or within atol) in ``seq``."""
    import torch
    n = block.shape[0]
    for o in candidates:
        if 0 <= o and o + n <= seq.shape[0]:
            part = seq[o:o + n]
            if (torch.equal(part, block) if atol == 0 else torch.allclose(part, block, rtol=0, atol=atol)):
                return o
    return None


def decide_source_position(len_without: int, len_with: int, n_src: int, n_image: int, n_text: int,
                           image_offset: Optional[int], text_run: Optional[Tuple[int, int]]) -> Tuple[Optional[str], List[str]]:
    """Where an edit pipeline's source-image tokens sit.

    len_without / len_with: sequence length without / with a source image (same prompt).
    image_offset: where the output-image block (identical to the run without a source image) was found in
      the with-source sequence.
    text_run: (start, end) of the block that changes between two with-source runs that differ only in the
      prompt; image and source tokens do not depend on the prompt, so this is the text block. (Text tokens
      themselves may change when a source image is added, e.g. shifted position encodings, so they are
      located this way rather than by comparing with the run without a source image.)
    """
    delta = len_with - len_without
    if delta == 0:
        return "none", ["sequence length is unchanged with a source image: source tokens are not in the sequence"]
    notes = [f"a source image added {delta} tokens (patch-grid count of the source latent: {n_src})"]
    if delta != n_src:
        notes.append("the added count does not match the source latent's patch grid; this node cannot count "
                     "source tokens for this model. Send this output.")
        return None, notes
    if image_offset is None:
        notes.append("the output-image tokens were not found intact in the with-source sequence; inconclusive")
        return None, notes
    if text_run is None or text_run[1] - text_run[0] != n_text:
        notes.append(f"the prompt-dependent block {text_run} is not exactly the {n_text} text tokens; inconclusive")
        return None, notes
    occupied = sorted([(image_offset, image_offset + n_image), tuple(text_run)])
    if occupied[0][1] > occupied[1][0]:
        notes.append(f"image block {image_offset}+{n_image} overlaps text block {text_run}; inconclusive")
        return None, notes
    gaps, cursor = [], 0
    for lo, hi in occupied:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = hi
    if cursor < len_with:
        gaps.append((cursor, len_with))
    if len(gaps) != 1 or gaps[0][1] - gaps[0][0] != n_src:
        notes.append(f"the tokens outside the image and text blocks are {gaps}, not one block of {n_src}; inconclusive")
        return None, notes
    start = gaps[0][0]
    pos = "start" if start == 0 else "end" if gaps[0][1] == len_with else "between"
    notes.append(f"image tokens at {image_offset}, text at {text_run[0]}..{text_run[1]}, source tokens at "
                 f"{start}..{gaps[0][1]}: source_tokens = {pos!r}")
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
                          "diagonal, so a transposed (column-major) grid would land here too; the off-diagonal "
                          "probe (row 0, col 1) is what rules that out.")
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
    cells = diff.detach()[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).mean(dim=(1, 3))
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


HOW_TO = {
    "text_position": "needs the prompt-length (order) pass",
    "image_order": "both single-patch probes must confirm raster order",
    "source_tokens": "connect an edit image (or pass --edit-image) to run the edit pass",
    "vae_downscale": "needs the shapes pass",
    "patch_size": "needs the shapes pass",
}


def render_config(f: Findings, edit_ran: bool = True) -> str:
    why = dict(HOW_TO)
    if edit_ran:
        why["source_tokens"] = "the edit pass was inconclusive; see problems"

    def opt(name, v):
        return f"{name:<14}= {v}" if v is not None else f"{name:<14}= NOT ESTABLISHED ({why[name]})"

    def py(v, name):
        return repr(v) if v is not None else f"None  # NOT ESTABLISHED: {why[name]}"

    lines = [
        "---- set these on MOSS-Image Regional LoRA + Layout Prompt ----",
        "advanced options:",
        "  " + opt("text_position", f.text_position),
        "  " + opt("image_order", f.image_order),
        "  " + opt("source_tokens", f.source_tokens) + (f"   ({f.source_note})" if f.source_tokens and f.source_note else ""),
        "measured for reference (the node reads these from the latent; nothing to set):",
        "  " + opt("vae_downscale", f.vae_downscale),
        "  " + opt("patch_size", f.patch_size),
        "",
        "---- or, equivalently, in moss_regional_lora/layout_config.py ----",
        f"TEXT_POSITION = {py(f.text_position, 'text_position')}",
        f"IMAGE_ORDER = {py(f.image_order, 'image_order')}",
        f"SOURCE_TOKENS = {py(f.source_tokens, 'source_tokens')}",
    ]
    if not edit_ran and f.source_tokens is None:
        lines += ["", "source_tokens was not measured because no edit image was given. For text-to-image only,",
                  "'none' is exact (there are no source tokens). Run the edit pass before using an edit workflow",
                  "with LoRA boxes."]
    if f.problems:
        lines += ["", "---- problems (send this whole report back) ----"] + [f"- {p}" for p in f.problems]
    return "\n".join(lines)


def find_blocks(dm, override: Optional[str], log=print):
    """FR-12b: try `blocks`, then `double_blocks`, then report what the module tree contains."""
    import torch.nn as nn
    if override:
        m = dm
        try:
            for part in override.split("."):
                m = m[int(part)] if part.isdigit() else getattr(m, part)
            n = len(m)
            first = m[0]
        except (AttributeError, IndexError, KeyError, TypeError):
            lists = [k for k, v in dm.named_modules() if isinstance(v, nn.ModuleList) and len(v) > 0]
            raise DiscoveryError(f"[blocks] '{override}' is not a non-empty list of blocks on the diffusion model. "
                                 f"Block lists found: {', '.join(lists) or 'none'}") from None
        log(f"[blocks] using block list {override}: {n} x {type(first).__name__}")
        return override, m
    for name in ("blocks", "double_blocks"):
        m = getattr(dm, name, None)
        if isinstance(m, nn.ModuleList) and len(m) > 0:
            log(f"[blocks] found '{name}' ({len(m)} x {type(m[0]).__name__})")
            return name, m
    lists = [(n, m) for n, m in dm.named_modules() if isinstance(m, nn.ModuleList) and len(m) > 0]
    log("[blocks] no 'blocks' or 'double_blocks' attribute on the diffusion model. ModuleLists found:")
    for n, m in lists:
        log(f"           {n:<40} {len(m):>4} x {type(m[0]).__name__}")
    direct = [(n, m) for n, m in lists if "." not in n]
    if len(direct) == 1:
        n, m = direct[0]
        log(f"[blocks] using '{n}': it is the only ModuleList directly on the diffusion model "
              "(nested lists such as a text embedder's blocks see only part of the sequence). "
              "Override the block list if this is wrong.")
        return n, m
    raise DiscoveryError("[blocks] cannot choose a block list automatically; set the block list (--blocks / the node's blocks option) to one of the lists above")


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
            raise DiscoveryError("[hooks] the first block never ran inside a model call; check the block list")
        return max(calls, key=lambda c: c["shape"][1])

    def remove(self):
        self.h1.remove()
        self.h2.remove()


@dataclass
class Settings:
    width: int = 1024
    height: int = 1024
    patch_size: Optional[int] = None          # None: infer from the shapes pass
    text_position: Optional[str] = None       # only for running the probe without the order pass
    prompt: str = "a quiet mountain lake at dawn, pine trees on the shore, soft mist"
    sampler: str = "euler"        # every pass is a single step from pure noise, so these barely matter
    scheduler: str = "simple"
    seed: int = 42
    probe_strength: float = 8.0
    edit_methods: Sequence[Optional[str]] = (None,)   # reference_latents_method values; None = unset
    atol: float = 0.0
    blocks: Optional[str] = None              # dotted path of the block list; None = auto


class Discovery:
    """Runs the passes against an already loaded MODEL (ModelPatcher), CLIP and VAE."""

    def __init__(self, model, clip, vae, settings: Settings, log: Callable[[str], None] = print):
        self.model, self.clip, self.vae = model, clip, vae
        self.s = settings
        self.log = log
        self.images: List[Tuple[str, object]] = []        # (name, [H, W, 3] float tensor in 0..1)
        self.edit_ran = False
        self.dm = model.get_model_object("diffusion_model")
        lf = model.get_model_object("latent_format")
        self.vae_downscale = int(getattr(lf, "spacial_downscale_ratio", 8))
        self.latent_channels = int(getattr(lf, "latent_channels", 4))
        log(f"[model] {type(self.dm).__name__}; latent format {type(lf).__name__}: "
            f"{self.latent_channels} channels, {self.vae_downscale} px per latent cell")
        self.block_path, self.blocks = find_blocks(self.dm, settings.blocks, log)
        self.findings = Findings(vae_downscale=self.vae_downscale)
        self._probe_base = None

    # ------------------------------------------------------------------ plumbing
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

    def generate(self, prompt, steps, cfg, probe=None, refs=None, method=None):
        import comfy.sample
        import torch
        s = self.s
        lat = torch.zeros(1, self.latent_channels, s.height // self.vae_downscale, s.width // self.vae_downscale)
        lat = comfy.sample.fix_empty_latent_channels(self.model, lat)
        noise = comfy.sample.prepare_noise(lat, s.seed)
        pos = self.encode(prompt, refs, method)
        neg = self.encode("", refs, method)
        rec = Recorder(self.dm, self.blocks[0], probe)
        try:
            samples = comfy.sample.sample(self.model, noise, steps, cfg, s.sampler, s.scheduler,
                                          pos, neg, lat, denoise=1.0, seed=s.seed)
        finally:
            rec.remove()
        return samples, rec

    def grid(self, patch):
        cell = self.vae_downscale * patch
        return self.s.height // cell, self.s.width // cell

    def need_vae(self, what):
        if self.vae is None:
            raise DiscoveryError(f"the {what} pass needs a VAE")

    # ------------------------------------------------------------------ passes
    def mode_shapes(self):
        self.log("\n=== shapes (FR-12b, FR-12c) ===")
        _, rec = self.generate(SHORT_PROMPT, 1, 1.0)
        main, fwd = rec.main_call(), rec.forwards[0]
        L = main["shape"][1]
        n_ctx = fwd["context"][1] if fwd["context"] else 0
        lh, lw = fwd["latent"][-2:]
        self.log(f"[shapes] first block ({self.block_path}[0]) input shape {main['shape']}  -> sequence length {L}")
        self.log(f"[shapes] latent {fwd['latent']}, text context {fwd['context']}")
        candidates = [self.s.patch_size] if self.s.patch_size else [1, 2, 4]
        match = None
        for p in candidates:
            if lh % p or lw % p:
                continue
            n_img = (lh // p) * (lw // p)
            implied = L - n_img
            ok = implied == n_ctx
            self.log(f"[shapes] canvas {self.s.width}x{self.s.height}, vae_downscale {self.vae_downscale}, patch {p}: "
                     f"expected image tokens {n_img}; observed length {L}; implied text tokens {implied}; "
                     f"text context length {n_ctx} -> {'MATCH' if ok else 'no match'}")
            if ok and match is None:
                match = p
        if match is None:
            self.findings.problems.append(f"no patch size in {candidates} makes (sequence - image tokens) equal the "
                                          f"text context length ({L} vs context {n_ctx})")
        else:
            self.findings.patch_size = match
        return match

    def _patch(self):
        return self.findings.patch_size or self.s.patch_size or self.mode_shapes()

    def mode_order(self):
        self.log("\n=== order (FR-12d) ===")
        p = self._patch()
        if p is None:
            self.log("[order] patch size unknown; cannot compute the image token count")
            return None
        rows, cols = self.grid(p)
        _, ra = self.generate(SHORT_PROMPT, 1, 1.0)
        _, rb = self.generate(LONG_PROMPT, 1, 1.0)
        ca, cb = ra.main_call(), rb.main_call()
        la, lb = ca["shape"][1], cb["shape"][1]
        na, nb = ra.forwards[0]["context"][1], rb.forwards[0]["context"][1]
        self.log(f"[order] short prompt: {na} text tokens, sequence {la}; long prompt: {nb} text tokens, sequence {lb}")
        pre, suf = identical_runs(ca["x"][0], cb["x"][0], atol=self.s.atol)
        self.log(f"[order] identical leading tokens {pre}, identical trailing tokens {suf} (image tokens {rows * cols})")
        pos, notes = decide_text_position(la, lb, na, nb, pre, suf, rows * cols)
        for n in notes:
            self.log(f"[order] {n}")
        if pos is None:
            self.findings.problems.append("text position: " + notes[-1])
        self.findings.text_position = pos
        return pos

    def probe_once(self, row, col, text_position, p):
        """Nudge one image token at the first block and see which patch of the model's prediction moves.

        One denoising step at CFG 1, compared in latent space: the nudged token's own output changes
        most, before sampling has a chance to spread a small change across the whole picture (comparing
        full generations measured that spread, not the token position).
        """
        import torch
        from .geometry import canvas_to_centered, coverage_grid, raster_flatten
        from .layout import Layout
        s = self.s
        rows, cols = self.grid(p)
        cell = self.vae_downscale * p
        box = canvas_to_centered(col * cell, row * cell, (col + 1) * cell, (row + 1) * cell, s.width, s.height)
        img_mask = raster_flatten(coverage_grid(box, s.width, s.height, self.vae_downscale, p))
        assert sum(img_mask) == 1.0 and img_mask[row * cols + col] == 1.0
        layout = Layout(text_position, "raster", "none")
        gen = torch.Generator().manual_seed(1234)

        def probe(x):
            L, D = x.shape[1], x.shape[2]
            seq = layout.sequence_mask(img_mask, L, L - rows * cols, 0)
            if seq is None:
                return x
            m = torch.tensor(seq, device=x.device, dtype=x.dtype)[None, :, None]
            v = torch.randn(D, generator=gen).to(x.device, x.dtype)
            v = v / v.norm() * x.float().norm(dim=-1).mean().to(x.dtype) * s.probe_strength
            return x + m * v

        if self._probe_base is None:
            self._probe_base, _ = self.generate(s.prompt, 1, 1.0)
        probed, _ = self.generate(s.prompt, 1, 1.0, probe=probe)
        base = self._probe_base
        latent_diff = (probed.float() - base.float()).abs().mean(dim=1)[0].detach().cpu()     # [h, w]
        (fr, fc), peak, median = strongest_cell(latent_diff, p)
        at_target = float(latent_diff[row * p:(row + 1) * p, col * p:(col + 1) * p].mean())
        if self.vae is not None:
            if not any(name == "prediction" for name, _ in self.images):
                self.images.append(("prediction", self.vae.decode(base)[0].detach().float().cpu()))
            self.images.append((f"probe_r{row}_c{col}", self.vae.decode(probed)[0].detach().float().cpu()))
        px = latent_diff.repeat_interleave(self.vae_downscale, 0).repeat_interleave(self.vae_downscale, 1)
        self.images.append((f"diff_r{row}_c{col}", heat(px, cell, (row, col))))
        self.log(f"[probe] nudged image token {row * cols + col} (patch row {row}, col {col}; grid {rows}x{cols}). "
                 f"Largest change in the one-step prediction: patch row {fr}, col {fc} (mean |diff| {peak:.4f}; "
                 f"at patch ({row}, {col}): {at_target:.4f}; median patch {median:.4f})")
        if peak < 4 * max(median, 1e-6):
            self.log("[probe] WARNING: the strongest patch is not clearly above the rest; raise the probe strength")
        ok, why = interpret_probe((fr, fc), (row, col), rows, cols)
        self.log(f"[probe] {why}")
        return ok, why

    def mode_probe(self, cells: Sequence[Tuple[int, int]] = ((0, 0), (0, 1))):
        self.log("\n=== probe (FR-12e) ===")
        p = self._patch()
        pos = self.findings.text_position or self.s.text_position or self.mode_order()
        if p is None or pos is None:
            self.log("[probe] need the patch size and the text position first")
            return None
        results = [self.probe_once(r, c, pos, p) for r, c in cells]
        confirmed = all(ok for ok, _ in results) and any(r != c for r, c in cells)
        if confirmed:
            self.findings.image_order = "raster"
        elif all(ok for ok, _ in results):
            self.log("[probe] raster order is consistent but not yet confirmed: probe an off-diagonal patch "
                     "(row 0, col 1) to rule out a transposed grid")
        else:
            self.findings.problems.append("image order: " + next(w for ok, w in results if not ok))
        return confirmed

    def mode_edit(self, edit_image):
        """edit_image: [1, H, W, 3] float tensor in 0..1 (ComfyUI IMAGE), or None to skip."""
        self.log("\n=== edit (FR-12f) ===")
        if edit_image is None:
            self.log("[edit] no edit image given; skipped")
            return None
        import comfy.utils
        self.need_vae("edit")
        s = self.s
        p = self._patch()
        pos = self.findings.text_position or s.text_position
        if p is None or pos is None:
            self.log("[edit] need the patch size and the text position first")
            self.findings.problems.append("source tokens: patch size or text position unknown")
            return None
        img = edit_image[:1, :, :, :3]
        if img.shape[1] != s.height or img.shape[2] != s.width:
            img = comfy.utils.common_upscale(img.movedim(-1, 1), s.width, s.height, "bilinear", "disabled").movedim(1, -1)
        ref = self.vae.encode(img)
        rows, cols = self.grid(p)
        n_img = rows * cols
        n_src = (ref.shape[-2] // p) * (ref.shape[-1] // p)
        _, r0 = self.generate(SHORT_PROMPT, 1, 1.0)
        c0 = r0.main_call()
        n_text = r0.forwards[0]["context"][1]
        l0 = c0["shape"][1]
        img_start = 0 if pos == "suffix" else n_text
        image_block = c0["x"][0][img_start:img_start + n_img]
        results = []
        for method in (s.edit_methods or [None]):
            _, ra = self.generate(SHORT_PROMPT, 1, 1.0, refs=[ref], method=method)
            ca, fa = ra.main_call(), ra.forwards[0]
            la = ca["shape"][1]
            self.log(f"[edit] reference_latents_method={method!r}: sequence {l0} -> {la}; model received "
                     f"ref_latents {fa['refs']}, precomputed ref K/V: {fa['ref_kv']}")
            image_offset = text_run = None
            if la != l0:
                xa = ca["x"][0]
                image_offset = find_block(xa, image_block, [0, n_text, n_src, n_text + n_src, n_img, n_img + n_src],
                                          atol=s.atol)
                _, rb = self.generate(LONG_PROMPT, 1, 1.0, refs=[ref], method=method)
                xb = rb.main_call()["x"][0]
                pre, suf = identical_runs(xa, xb, atol=s.atol)
                text_run = (pre, la - suf)
                self.log(f"[edit]   output-image tokens found at {image_offset}; with a different prompt the "
                         f"sequence is identical in its first {pre} and last {suf} tokens")
            found, notes = decide_source_position(l0, la, n_src, n_img, n_text, image_offset, text_run)
            for n in notes:
                self.log(f"[edit]   {n}")
            results.append(found)
        combined, why = combine_source_positions(results)
        self.log(f"[edit] source_tokens -> {combined!r} ({why})")
        if combined is None:
            self.findings.problems.append("source tokens: " + why)
        self.findings.source_tokens = combined
        self.findings.source_note = why
        self.edit_ran = True
        return combined

    def run_all(self, edit_image=None):
        self.mode_shapes()
        self.mode_order()
        self.mode_probe(((0, 0), (0, 1)))
        self.mode_edit(edit_image)
        self.log("\n=== configuration (FR-12g) ===")
        self.log(render_config(self.findings, edit_ran=self.edit_ran))
        return self.findings


def parse_edit_methods(text: str) -> List[Optional[str]]:
    """Comma-separated reference_latents_method values; "default" (or empty) means unset."""
    out: List[Optional[str]] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(None if part.lower() in ("default", "none", "unset") else part)
    return out or [None]


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


