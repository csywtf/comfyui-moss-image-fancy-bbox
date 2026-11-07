# MOSS-Image Regional LoRA + Layout Prompt (ComfyUI custom node)

One node to draw the boxes of a MOSS-Image generation. Each box can carry:

- a **layout-prompt item**: a label (object, character or text), a caption, text to render and a
  colour palette, written into the MOSS-Image layout prompt exactly as ComfyUI's *Create Bounding
  Boxes* + *MOSS-Image Layout Prompt* would;
- a **LoRA** applied only inside the box, so two character LoRAs in two boxes stay two different
  people and a character's look stays out of the background;
- or **both**.

The node outputs the patched MODEL and the finished layout prompt STRING.

> **LoRA boxes need the layout discovery below first; until then the node refuses to apply them.**
> It needs three facts about MOSS-Image's transformer sequence that must be measured on your machine.
> A wrong guess would produce a node that *appears* to work (the LoRA still changes the picture) while
> the mask sits on the wrong tokens. Prompt-only boxes don't touch the model and work right away.

## Install

Clone or copy this folder into ComfyUI's `custom_nodes` directory and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
```

```bash
git clone https://github.com/csywtf/comfyui-moss-image-fancy-bbox.git
```

There are no dependencies beyond ComfyUI itself. The region editor in `web/` is registered
automatically through the node's `WEB_DIRECTORY`; there is nothing to install in the browser.

## Where the node goes

`model_patches/moss_image` → **MOSS-Image Regional LoRA + Layout Prompt**

```
Load Diffusion Model ──MODEL──▶ MOSS-Image Regional LoRA ──model───▶ KSampler (model)
                                 + Layout Prompt          ──prompt──▶ CLIP Text Encode (text) ──▶ KSampler (positive)
                                 (draw boxes here)
Load CLIP (type moss_image) ──CLIP──▶ CLIP Text Encode
Load VAE ──VAE──▶ VAE Decode
```

It replaces *Create Bounding Boxes* → *MOSS-Image Layout Prompt*. Place it on the MODEL line between
the loader (and any ordinary `LoraLoader`s) and the sampler, and connect its `prompt` output to your
positive text encode (convert that node's text to an input if needed). CLIP and the VAE stay wired as
usual; the node never touches them, and a LoRA's text-encoder keys are skipped (put trigger words in
a caption instead).
Set `width`/`height` to the size of the image you are generating and `steps` to the sampler's step
count. It is built to sit alongside normal `LoraLoader` nodes, ComfyUI's low-VRAM weight casting and
offload, and `torch.compile`, without turning any of them off (the hooks only read each layer's input
and output, and compile without graph breaks).

Inputs:

| input | meaning |
|---|---|
| `width`, `height` | output canvas in pixels; must match the latent being sampled (checked at run time) |
| `regions` | the region list as JSON, edited with the canvas below it (see *Region editor*) |
| `vae_downscale`, `patch_size` | pixels per latent cell and latent cells per token; the discovery tool prints both (ComfyUI's MOSS-Image latent format is 16 px per cell) |
| `ramp_start`, `ramp_end` | the LoRA fades in from 0 at `ramp_start` of the steps to full at `ramp_end` (default 0.15 → 1.0), so it fills the layout instead of fighting it |
| `steps` | total sampling steps, for the ramp |
| `strict` | on: refuse to apply LoRAs while the layout is unverified and raise on any alignment problem. Off: pass the model through untouched (with a warning) instead. Prompt-only boxes work either way |
| `caption` | description of the whole image: the layout prompt's summary caption. With no captioned boxes, `prompt` is just this text |

## Why the weights are never merged

`LoraLoader` adds `strength · B·A` into each weight matrix. A merged weight is the same for every
token, so there is nothing left to restrict to a region. This node never touches a model parameter.
It puts a forward hook on each linear layer the LoRA targets and, at run time, adds the LoRA's own
output only on the tokens inside a box:

```
y = W·x + b  +  Σ_regions  mask_r ⊙ scale_r · B_r(A_r · x)      (mask_r ≠ 0)
y = W·x + b                                                     (everywhere else)
```

Outside a box, the layer's output is selected unchanged, bit for bit. The mask is fractional on
patches that straddle a box edge, so silhouettes are not staircased. Text tokens, source-image
tokens and the unconditional (negative) pass always get a zero mask. `scale_r` follows ComfyUI's
LoRA convention exactly (`alpha / rank` when the file has an alpha, else 1), so a box covering the
whole canvas at weight 1.0 matches `LoraLoader` at strength 1.0 on the image tokens.

The hooks are installed through ComfyUI's own `ModelPatcher` injection mechanism, so they come and
go with the model being loaded, and running the node again replaces its hooks instead of stacking
them. Because the change is invisible to ComfyUI's cache, the node always re-executes.

kohya (`lora_down`/`lora_up`/`alpha`) and peft/diffusers (`lora_A`/`lora_B`) LoRAs are supported.
Keys that do not resolve to a linear layer (text-encoder keys, DoRA, LoHa, ...) are logged and
skipped, and the count is shown under the editor after a run.

## Region editor

The canvas under the `regions` text box shows the output canvas with a grid (a third of the canvas
by default) and the centre lines. The `regions` text is the source of truth: the canvas only draws
it and writes it back after each edit. You can also edit or paste the JSON by hand and the canvas
follows. If the text stops being valid JSON, the editor turns red and becomes read-only until it is
fixed; it never overwrites your text.

Boxes are stored as `[y_min, x_min, y_max, x_max]`, integer pixels, measured from the canvas centre
with y pointing down. This is the same convention as ComfyUI's *MOSS-Image Layout Prompt* node. The
selected box's stored value is shown above the canvas.

**Layer order:** the top row of the layer list is drawn first (underneath); the bottom row is drawn
last (on top). LoRAs and layout-prompt items are emitted in exactly this order, minus hidden rows, so
put the background first and the foreground last, as with *Create Bounding Boxes*. Groups only
organise rows; they change nothing about the output.

**What a box does** is decided by its fields. It is a layout-prompt item when it has a caption (or,
for a `text` box, text to render); it applies a LoRA when one is selected with a non-zero weight. A
box with only a LoRA leaves the prompt alone; a box with only a caption leaves the model alone.

| action | how |
|---|---|
| new region | drag on empty canvas |
| move | drag the region body |
| resize | drag one of the 8 handles of the selected region |
| square | hold **Shift** while creating or resizing |
| resize from centre | hold **Alt** while creating or resizing |
| bypass snapping | hold **Ctrl** (**Cmd** on macOS) while dragging, with Snap on |
| nudge 1 px / 10 px | **Arrow keys** / **Shift + Arrow keys** |
| delete selection | **Delete** or **Backspace** (locked regions are kept) |
| undo / redo | **Ctrl/Cmd + Z** / **Ctrl/Cmd + Shift + Z** (or **Ctrl + Y**) |
| multi-select | **Shift**- or **Ctrl/Cmd**-click regions or rows |
| reorder | ▲ / ▼ buttons, or drag a row by its ⠿ grip |
| group | **+ Group** (puts the selected regions in a new group), or pick a group in a row |

The keyboard shortcuts work while the canvas has focus (click it first). A row's first line has a
visibility toggle, colour, name, box readout, LoRA, weight, lock, group, move and delete controls; its
second line has the layout label, caption, text to render (`text` boxes) and up to five palette
colours. The canvas label shows what each box does (`LoRA`, its label, or both). ⚠ marks a
box that extends past the canvas (allowed, but usually a mistake). Locked regions cannot be moved
or resized on the canvas. Group headers collapse, rename, show all, hide all, move and delete all.
An optional reference image from ComfyUI's `input` folder can be shown under the boxes with an
opacity slider; it is a drawing aid only.

## Layout discovery (required once per checkpoint)

The node needs to know, for your checkpoint: which positions of the transformer sequence are image
tokens, whether the text comes before or after them, whether image tokens are in plain row-by-row
order, and where an edit pipeline's source-image tokens go. `tools/discover_moss_layout.py`
measures all of this with ComfyUI's own loaders and prints the values to set.

1. Make sure the MOSS-Image checkpoint generates normally in ComfyUI, and close ComfyUI so the GPU is
   free.
2. Open a terminal in the ComfyUI folder and activate the Python environment ComfyUI uses.
3. Pick any image to act as an edit source, for example one from `ComfyUI/input/`.
4. Run the full discovery. With an all-in-one checkpoint:

   ```bash
   python custom_nodes/comfyui-moss-image-fancy-bbox/tools/discover_moss_layout.py --mode all --checkpoint YOUR_MOSS_CHECKPOINT.safetensors --edit-image input/YOUR_IMAGE.png --out moss_probe
   ```

   With separate files, replace `--checkpoint ...` with
   `--unet MOSS_DIFFUSION_MODEL.safetensors --clip MOSS_TEXT_ENCODER.safetensors --vae MOSS_VAE.safetensors`.
   Names are looked up in ComfyUI's model folders; full paths also work (use full paths for models in
   `extra_model_paths.yaml` locations). If you use an edit workflow, also add one
   `--edit-method NAME` per `reference_latents_method` your edit node sets (for ComfyUI's
   *Text Encode MOSS-Image Edit* node these are listed in `EDIT_FORMATS` in
   `comfy_extras/nodes_moss_image.py`).
5. Read the `[blocks]` lines. If it says it cannot choose a block list, re-run with
   `--blocks <name>` using the list it printed that holds the main transformer blocks.
6. Read the `[shapes]` lines: one patch size must say **MATCH** (sequence length minus image tokens
   equals the text length).
7. Read the `[order]` lines: they say whether text is a PREFIX or a SUFFIX of the sequence.
8. Open the images in `moss_probe/`. `diff_r0_c0.png` and `diff_r0_c1.png` show where perturbing one
   image token changed the picture; the green square marks where raster order predicts it. The
   `[probe]` lines say whether the change landed there. Both probes must confirm.
9. Read the `[edit]` lines: they say whether the source image added tokens to the sequence and where.
10. Copy the block printed under `=== configuration ===` into
    `custom_nodes/comfyui-moss-image-fancy-bbox/moss_regional_lora/layout_config.py`, and set the
    node's `vae_downscale` and `patch_size` inputs to the printed values.
11. If any value printed as `None  # NOT ESTABLISHED`, or a `problems` section appears, do not
    guess: open an issue with the whole output and the `moss_probe/` images.

Single steps can be re-run with `--mode shapes`, `--mode order`, `--mode probe` (with
`--probe-row`, `--probe-col`, `--text-position`) or `--mode edit`. Run with `--help` for every option.

## Tests

No GPU and no MOSS weights needed; a toy transformer stands in for the model.

```bash
pip install torch pytest
```

```bash
python -m pytest
```

```bash
node --test tests/web/*.test.mjs
```

`node tests/web/editor_dom_smoke.mjs` additionally drives the editor in headless Chromium against a
stub of ComfyUI's frontend API (needs Playwright). With `COMFYUI_PATH=/path/to/ComfyUI`, `pytest`
also checks LoRA equivalence against ComfyUI's own `comfy.lora` code.
