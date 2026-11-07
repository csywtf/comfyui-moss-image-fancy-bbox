"""Forward-time gated LoRA deltas (FR-3, FR-5, FR-6, FR-7, FR-10, NFR-1, NFR-2, NFR-3).

Mechanism
---------
Each target linear module gets one ``register_forward_hook``. The hook leaves the
module's own computation alone (whatever ComfyUI does to compute it: weight casting,
CPU offload, a merged LoraLoader patch, quantised kernels) and post-processes its
output::

    y = base + sum_r  mask_r * (scale_r * B_r(A_r x))     where mask_r != 0
    y = base                                              everywhere else

The second line is a ``torch.where`` select, so masked-out positions are the base
module's output tensor values, bit for bit, even if the delta overflowed or is NaN.

No parameter of the model is ever written (NFR-1). The LoRA matrices live on the
controller, are copied to the compute device/dtype of the call, and dropped on eject.

Per-call state (which rows are conditional, the step's ramp factor, the full-sequence
masks) is computed outside the transformer by an ``APPLY_MODEL`` wrapper and handed to
the hooks as tensors only. Nothing step-dependent is a Python scalar inside the traced
region, so ``torch.compile`` sees the same graph every step (NFR-3).
"""

from __future__ import annotations

import logging
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .geometry import coverage_grid, patch_grid_size, raster_flatten
from .layout import DISCOVERY_TOOL, Layout
from .lora_file import LoraTarget
from .regions import RegionSpec

log = logging.getLogger("moss_regional_lora")

HOOK_ATTR = "_moss_regional_lora_hook"
COND, UNCOND = 0, 1


class RegionalLoraError(RuntimeError):
    pass


def _compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except AttributeError:  # pragma: no cover - torch < 2.1
        return False


def ramp_factor(step: int, steps: int, ramp_start: float, ramp_end: float) -> float:
    """0 at ``ramp_start`` of the run, rising linearly to 1 at ``ramp_end`` (FR-7).

    Progress is ``step / (steps - 1)``, so step 0 is 0.0 and the final step is 1.0.
    """
    progress = step / max(steps - 1, 1)
    if progress < ramp_start:
        return 0.0
    if ramp_end <= ramp_start or progress >= ramp_end:
        return 1.0
    return (progress - ramp_start) / (ramp_end - ramp_start)


def step_index(sigma: float, sample_sigmas: Sequence[float]) -> int:
    """Index of the sampler step whose interval contains ``sigma``.

    Exact schedule sigmas map to their index; intermediate evaluations of multi-stage
    samplers (Heun, DPM++ 2S, ...) map to the step they belong to.
    """
    tol = 1e-4 * max(abs(sigma), 1e-6)
    count = sum(1 for s in sample_sigmas if s >= sigma - tol)
    return max(0, min(count - 1, len(sample_sigmas) - 2))


class _Adapter:
    """One region's LoRA on one module."""

    __slots__ = ("region", "scale", "down_master", "up_master", "down", "up", "out_offset", "out_size")

    def __init__(self, region: int, target: LoraTarget, strength: float):
        self.region = region
        self.scale = float(strength) * float(target.alpha_scale)
        self.down_master = target.down
        self.up_master = target.up
        self.down = None
        self.up = None
        self.out_offset = target.out_offset
        self.out_size = target.out_size

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """``scale * B(A x)`` for this adapter's output slice."""
        down, up = self.down, self.up
        if down is None or up is None or down.device != x.device:  # not prepared for this device (direct use / tests)
            down, up = self.down_master.to(x.device), self.up_master.to(x.device)
        if down.dtype != x.dtype:
            down, up = down.to(x.dtype), up.to(x.dtype)
        return F.linear(F.linear(x, down), up) * self.scale


class _State:
    """Per-controller call state read by the hooks. Persistent object; fields are swapped per call."""

    def __init__(self):
        self.active = False
        self.coefs: Dict[Tuple[int, int], List[torch.Tensor]] = {}   # (batch, seq_len) -> [region] -> [B, L, 1]
        self.hits: Optional[torch.Tensor] = None                       # joint-sequence hook calls this call
        self.unmatched: set = set()
        self.fired: set = set()


class _ModuleGate:
    """Forward hook for one linear module."""

    def __init__(self, controller: "RegionalLoraController", path: str, module: torch.nn.Module):
        self._controller = weakref.ref(controller)
        self.path = path
        self.module_ref = weakref.ref(module)
        self.out_features = linear_features(module)[1]
        self.adapters: List[_Adapter] = []
        self.handle = None
        self.saved_force_forward = None

    def __call__(self, module, args, output):
        controller = self._controller()
        if controller is None:
            return None
        st = controller.state
        if not st.active:
            return None
        x = args[0] if args else None
        if not torch.is_tensor(x) or not torch.is_tensor(output) or x.dim() != 3:
            if not _compiling():
                st.unmatched.add((self.path, "input has no [batch, tokens, features] layout"))
            return None
        coefs = st.coefs.get((x.shape[0], x.shape[1]))
        if coefs is None:
            if not _compiling():
                st.unmatched.add((self.path, f"sequence length {x.shape[1]} is not the joint text+image sequence"))
            return None
        st.hits.add_(1)
        if not _compiling():
            st.fired.add(self.path)
        total = None
        gate = None
        for ad in self.adapters:
            c = coefs[ad.region]
            d = ad.delta(x)
            if ad.out_size is not None:
                d = F.pad(d, (ad.out_offset, self.out_features - ad.out_offset - ad.out_size))
            c = c.to(d.dtype)
            term = c * d
            total = term if total is None else total + term
            g = c != 0
            gate = g if gate is None else gate | g
        return torch.where(gate, output + total.to(output.dtype), output)

    def installed_on(self, m: torch.nn.Module) -> bool:
        return self.handle is not None and getattr(m, HOOK_ATTR, None) is self

    def remove_from(self, m: torch.nn.Module):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.saved_force_forward is None:
            m.__dict__.pop("comfy_force_forward", None)
        else:
            m.comfy_force_forward = self.saved_force_forward
        self.saved_force_forward = None
        if getattr(m, HOOK_ATTR, None) is self:
            delattr(m, HOOK_ATTR)


def linear_features(m: torch.nn.Module) -> Optional[Tuple[int, int]]:
    """(in_features, out_features) if ``m`` is a linear layer, else None.

    ComfyUI's quantised ``MixedPrecisionOps.Linear`` is a plain ``nn.Module`` (not an ``nn.Linear``
    subclass) whose weight may be packed, so linearity and shape come from the ``in_features`` /
    ``out_features`` attributes every ComfyUI linear op carries, not from ``weight.shape``.
    """
    fi, fo = getattr(m, "in_features", None), getattr(m, "out_features", None)
    if isinstance(m, torch.nn.Linear) or (isinstance(fi, int) and isinstance(fo, int)):
        return int(fi), int(fo)
    return None


def iter_linear_modules(root: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Module]]:
    for name, m in root.named_modules():
        if linear_features(m) is not None:
            yield name, m


class RegionalLoraController:
    """Owns the hooks for one node execution. Created by the node, injected by ModelPatcher."""

    def __init__(self, regions: Sequence[RegionSpec], targets_per_region: Sequence[Sequence[LoraTarget]],
                 layout: Layout, width: int, height: int, vae_downscale: int, patch_size: int,
                 ramp_start: float, ramp_end: float, steps: int, strict: bool):
        if len(regions) != len(targets_per_region):
            raise ValueError("one target list per region")
        self.regions = list(regions)
        self.targets_per_region = [list(t) for t in targets_per_region]
        self.layout = layout
        self.width, self.height = int(width), int(height)
        self.vae_downscale, self.patch_size = int(vae_downscale), int(patch_size)
        self.rows, self.cols = patch_grid_size(self.width, self.height, self.vae_downscale, self.patch_size)
        self.ramp_start, self.ramp_end = float(ramp_start), float(ramp_end)
        self.steps = int(steps)
        self.strict = bool(strict)
        self.image_masks = [raster_flatten(coverage_grid(r.bbox, self.width, self.height,
                                                         self.vae_downscale, self.patch_size))
                            for r in self.regions]
        self.state = _State()
        self.gates: Dict[str, _ModuleGate] = {}
        self.skipped: List[Tuple[int, str, str]] = []   # (region index, LoRA key, reason)
        self.injected = False
        self._seq_cache: Dict[tuple, torch.Tensor] = {}
        self._fallback_step = (None, -1)  # (last sigma, step) when the sampler gives no schedule
        self._warned: set = set()
        self._calls = 0

    # ------------------------------------------------------------------ binding
    def bind(self, diffusion_model: torch.nn.Module) -> "RegionalLoraController":
        """Resolve LoRA targets to live linear modules. Unresolvable ones are skipped and counted (FR-9)."""
        modules = dict(diffusion_model.named_modules())
        for r, targets in zip(self.regions, self.targets_per_region):
            for t in targets:
                m = modules.get(t.module_path)
                if m is None:
                    self.skipped.append((r.index, t.source_prefix, f"no module '{t.module_path}' in the diffusion model"))
                    continue
                feats = linear_features(m)
                if feats is None:
                    self.skipped.append((r.index, t.source_prefix, f"'{t.module_path}' is {type(m).__name__}, not a linear module"))
                    continue
                in_f, out_f = feats
                size = t.out_size if t.out_size is not None else out_f
                if t.down.shape[1] != in_f or t.up.shape[0] != size or t.out_offset + size > out_f:
                    self.skipped.append((r.index, t.source_prefix, f"shape mismatch on '{t.module_path}': module {out_f}x{in_f}, "
                                                          f"LoRA up {tuple(t.up.shape)} down {tuple(t.down.shape)} "
                                                          f"slice {t.out_offset}+{size}"))
                    continue
                gate = self.gates.get(t.module_path)
                if gate is None:
                    gate = self.gates[t.module_path] = _ModuleGate(self, t.module_path, m)
                gate.adapters.append(_Adapter(r.index, t, r.weight))
        return self

    @property
    def applied_count(self) -> int:
        return sum(len(g.adapters) for g in self.gates.values())

    # ---------------------------------------------------------- inject / eject
    def inject(self, *_):
        """Install one hook per module (idempotent; replaces any stale hook from an earlier run)."""
        for gate in self.gates.values():
            m = gate.module_ref()
            if m is None:
                continue
            if gate.installed_on(m):
                continue
            stale = getattr(m, HOOK_ATTR, None)
            if stale is not None:
                stale.remove_from(m)
            gate.handle = m.register_forward_hook(gate)
            gate.saved_force_forward = m.__dict__.get("comfy_force_forward", None)
            # ComfyUI's fused paths (comfy.ops.linear_input_act) skip module.__call__ unless this is set.
            m.comfy_force_forward = True
            setattr(m, HOOK_ATTR, gate)
        self.injected = True

    def hooks_installed(self) -> bool:
        return all(g.module_ref() is None or g.installed_on(g.module_ref()) for g in self.gates.values())

    def eject(self, *_):
        for gate in self.gates.values():
            m = gate.module_ref()
            if m is not None and gate.installed_on(m):
                gate.remove_from(m)
        self._release_device_copies()
        self.injected = False

    def _release_device_copies(self):
        for gate in self.gates.values():
            for ad in gate.adapters:
                ad.down = ad.up = None
        self._seq_cache.clear()

    def _prepare_device(self, device, dtype):
        copies: Dict[tuple, torch.Tensor] = {}

        def to(t):
            key = (id(t), device, dtype)
            if key not in copies:
                copies[key] = t.to(device=device, dtype=dtype)
            return copies[key]

        for gate in self.gates.values():
            for ad in gate.adapters:
                if ad.down is None or ad.down.device != device or ad.down.dtype != dtype:
                    ad.down, ad.up = to(ad.down_master), to(ad.up_master)

    # ------------------------------------------------------------- per call
    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            log.warning("[MOSS Regional LoRA] %s", msg)

    def _fail(self, msg):
        if self.strict:
            raise RegionalLoraError(msg)
        self._warn_once(msg, msg + " (strict is off: regional LoRA skipped for this call)")

    def _resolve_step(self, transformer_options) -> int:
        sigmas = transformer_options.get("sigmas")
        sample_sigmas = transformer_options.get("sample_sigmas")
        sigma = float(sigmas.flatten()[0]) if torch.is_tensor(sigmas) else None
        if sigma is None:
            self._warn_once("nosigma", "no current sigma in transformer_options; treating every call as the final step")
            return max(self.steps - 1, 0)
        if sample_sigmas is not None:
            ss = [float(s) for s in (sample_sigmas.flatten().tolist() if torch.is_tensor(sample_sigmas) else sample_sigmas)]
            if len(ss) - 1 != self.steps:
                self._warn_once(("steps", len(ss)), f"node 'steps' is {self.steps} but the sampler is running "
                                                    f"{len(ss) - 1} steps; the ramp uses the node's value")
            return step_index(sigma, ss)
        # No schedule available: count distinct decreasing sigmas; a rise means a new run.
        last, step = self._fallback_step
        if last is None or sigma > last:
            step = 0
        elif sigma < last:
            step += 1
        self._fallback_step = (sigma, step)
        return step

    def _sequence_masks(self, seq_len, n_text, n_src, device, dtype) -> Optional[torch.Tensor]:
        key = (seq_len, n_text, n_src, device, dtype)
        t = self._seq_cache.get(key)
        if t is None:
            rows = [self.layout.sequence_mask(m, seq_len, n_text, n_src) for m in self.image_masks]
            if any(r is None for r in rows):
                return None
            t = torch.tensor(rows, dtype=torch.float32).to(device=device, dtype=dtype)  # [R, L]
            self._seq_cache[key] = t
        return t

    def begin_call(self, x: torch.Tensor, transformer_options: dict, c_crossattn=None, ref_latents=None,
                   dtype: Optional[torch.dtype] = None) -> bool:
        """Compute this call's masks. Returns False (and leaves hooks inert) if gating can't apply."""
        st = self.state
        st.active = False
        st.coefs = {}
        if not self.gates:
            return False
        if not self.hooks_installed():
            self.inject()
        if x.dim() != 4:
            self._fail(f"expected a 4-D image latent [B, C, H, W], got shape {tuple(x.shape)}")
            return False
        B, h, w = x.shape[0], x.shape[-2], x.shape[-1]
        if h * self.vae_downscale != self.height or w * self.vae_downscale != self.width:
            self._fail(f"the latent being sampled is {w}x{h}, which at vae_downscale={self.vae_downscale} is "
                       f"{w * self.vae_downscale}x{h * self.vae_downscale} px, but the node's canvas is "
                       f"{self.width}x{self.height}. Set width/height to the generated image size (area-cropped "
                       "conditioning is not supported).")
            return False
        n_image = self.rows * self.cols
        if (h // self.patch_size) * (w // self.patch_size) != n_image or h % self.patch_size or w % self.patch_size:
            self._fail(f"latent {w}x{h} does not tile into the {self.cols}x{self.rows} patch grid at patch_size={self.patch_size}")
            return False
        n_text = int(c_crossattn.shape[1]) if torch.is_tensor(c_crossattn) and c_crossattn.dim() >= 2 else 0
        n_src = 0
        for ref in (ref_latents or []):
            rh, rw = ref.shape[-2], ref.shape[-1]
            n_src += (rh // self.patch_size) * (rw // self.patch_size)
        # Source tokens may or may not ride in the sequence for a given edit setup (MOSS-Image keeps
        # them out when their keys/values are precomputed), so both joint lengths are accepted. Each
        # gets its own mask; a hook picks the one matching the length it actually sees.
        src_variants = [0] + ([n_src] if n_src and self.layout.source_tokens != "none" else [])

        cond_or_uncond = transformer_options.get("cond_or_uncond")
        if not cond_or_uncond:
            self._warn_once("nocou", "no cond_or_uncond in transformer_options; treating every row as conditional")
            cond_or_uncond = [COND]
        if B % len(cond_or_uncond):
            self._fail(f"batch {B} does not split into {len(cond_or_uncond)} cond/uncond chunks")
            return False
        per = B // len(cond_or_uncond)
        row_is_cond = [cond_or_uncond[b // per] == COND for b in range(B)]

        step = self._resolve_step(transformer_options)
        ramp = ramp_factor(step, self.steps, self.ramp_start, self.ramp_end)

        device = x.device
        dtype = dtype or x.dtype
        self._prepare_device(device, dtype)
        row_scale = torch.tensor([ramp if c else 0.0 for c in row_is_cond], dtype=torch.float32).to(device=device, dtype=dtype)
        coefs = {}
        for src in src_variants:
            seq_len = self.layout.sequence_length(n_image, n_text, src)
            seq = self._sequence_masks(seq_len, n_text, src, device, dtype)
            coef = seq[:, None, :, None] * row_scale[None, :, None, None]       # [R, B, L, 1]
            coefs[(B, seq_len)] = list(coef.unbind(0))
        st.coefs = coefs
        st.hits = torch.zeros((), dtype=torch.int64, device=device)
        st.unmatched = set()
        st.fired = set()
        st.active = True
        self._expected = (sorted(L for _, L in coefs), n_image, n_text, n_src)
        return True

    def end_call(self):
        st = self.state
        if not st.active:
            return
        st.active = False
        self._calls += 1
        hits = int(st.hits.item())
        for path, reason in sorted(st.unmatched):
            self._warn_once(("unmatched", path), f"'{path}': {reason}; its regional delta is zero")
        if hits == 0:
            lengths, n_image, n_text, n_src = self._expected
            seen = sorted({r for _, r in st.unmatched})
            self._fail(f"no LoRA-targeted linear module saw the expected joint sequence (length "
                       f"{' or '.join(map(str, lengths))}: {n_image} image + {n_text} text tokens"
                       f"{f' (+ {n_src} source)' if len(lengths) > 1 else ''}). Hooks reported: {seen[:4]}. "
                       f"Either vae_downscale/patch_size are wrong for this model or the layout constants in "
                       f"layout_config.py do not match it; re-run {DISCOVERY_TOOL}.")
        elif self._calls == 1 and not _compiling() and st.fired:
            silent = [p for p in self.gates if p not in st.fired and p not in {u[0] for u in st.unmatched}]
            if silent:
                self._warn_once("silent", f"{len(silent)} hooked module(s) were never called through forward, "
                                          f"so their delta is not applied, e.g. {silent[:3]}")

    def apply_model_wrapper(self, executor, x, t, c_concat=None, c_crossattn=None, control=None,
                            transformer_options={}, **kwargs):
        """ComfyUI ``WrappersMP.APPLY_MODEL`` wrapper: brackets each model call with begin/end."""
        dtype = None
        get_dtype = getattr(getattr(executor, "class_obj", None), "get_dtype_inference", None)
        if callable(get_dtype):
            dtype = get_dtype()
        self.begin_call(x, transformer_options, c_crossattn=c_crossattn,
                        ref_latents=kwargs.get("ref_latents"), dtype=dtype)
        try:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        finally:
            self.end_call()

