# MOSS-Image Regional LoRA + Layout Prompt (ComfyUI custom node)

One node to draw the boxes of a MOSS-Image generation. Each box can carry:

- a **layout-prompt item**: a label (object, character or text), a caption, text to render and a
  colour palette, written into the MOSS-Image layout prompt exactly as ComfyUI's *Create Bounding
  Boxes* + *MOSS-Image Layout Prompt* would;
- one or more **LoRAs** applied only inside the box, each with its own strength and strength-over-time
  curve, so two character LoRAs in two boxes stay two different people and a character's look stays
  out of the background;
- or **both**.

The node outputs the patched MODEL and the finished layout prompt STRING.

> **Where the LoRA lands depends on three facts about MOSS-Image's transformer sequence.** They were
> measured with the included **MOSS-Image Layout Discovery** node on ComfyUI's MOSS-Image
> implementation and are the main node's defaults (`text_position = suffix`, `image_order = raster`,
> `source_tokens = end`). They come from ComfyUI's code, not the weights, so they hold for every
> MOSS-Image checkpoint. **After a ComfyUI update, re-run the discovery node** (see *Layout discovery*
> below): a wrong value would produce a node that *appears* to work (the LoRA still changes the
> picture) while the mask sits on the wrong tokens.

## Install

Clone or copy this folder into ComfyUI's `custom_nodes` directory and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
```

```bash
git clone https://github.com/csywtf/comfyui-moss-image-fancy-bbox.git
```

There are no dependencies beyond ComfyUI itself. The editor's icons are a small bundled subset of
[Lucide](https://lucide.dev) (ISC license, `web/LICENSE-lucide.txt`); nothing is downloaded at run time. The region editor in `web/` is registered
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
| `vae_downscale`, `patch_size` | pixels per latent cell and latent cells per token; the *MOSS-Image Layout Discovery* report gives both (ComfyUI's MOSS-Image latent format is 16 px per cell) |
| `ramp_start`, `ramp_end` | advanced, legacy: a linear fade-in used only by boxes **without** their own strength curve (workflows saved before curves). New boxes get their own curve (see *Strength curves*) |
| `steps` | advanced, fallback: the curves follow the sampler's own step count; this is used only if a sampler doesn't report one |
| `strict` | on: refuse to apply LoRAs while the layout is unverified and raise on any alignment problem. Off: pass the model through untouched (with a warning) instead. Prompt-only boxes work either way |
| `caption` | description of the whole image: the layout prompt's summary caption. With no captioned boxes, `prompt` is just this text |
| `text_position`, `image_order`, `source_tokens` | **advanced options**: the layout values reported by the *MOSS-Image Layout Discovery* node. Their defaults (`suffix`, `raster`, `end`) were measured on a real MOSS-Image run. "not set" falls back to `moss_regional_lora/layout_config.py` (unset by default) |

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

The editor under the `regions` text box has three panels, with a floating toolbar on the canvas:

```
[layers] [ [tools]       canvas        ] [properties]
```

The `regions` text is the source of truth: the editor only draws it and writes it back after each
edit. You can also edit or paste the JSON by hand and the editor follows. If the text stops being
valid JSON, the editor turns red and becomes read-only until it is fixed; it never overwrites your
text.

- **Tools** (floating at the top left of the canvas): **Select** (V) clicks, moves and resizes boxes; clicking the background deselects.
  **Box** (B) draws a new box, then switches back to Select with the new box selected.
- **Layers**: one row per box, with visibility, colour, name, badges (`2L` = two LoRAs, `✎` = in the
  layout prompt, ⚠ = extends past the canvas), lock and ▲ / ▼. Click a row to select the box.
  **+ Group** makes a group (with the selected boxes in it); group headers collapse, rename, show all,
  hide all, move and delete all.
- **Canvas**: the output canvas with a grid (a third of the canvas by default) and the centre lines.
- **Properties** (right): for the selected box, its name, colour, visibility, lock, box coordinates
  (editable), group, layout-prompt fields, and its **LoRA list**. With nothing selected it shows the
  canvas settings: grid, snap, and an optional reference image from ComfyUI's `input` folder with an
  opacity slider (a drawing aid only).

Boxes are stored as `[y_min, x_min, y_max, x_max]`, integer pixels, measured from the canvas centre
with y pointing down. This is the same convention as ComfyUI's *MOSS-Image Layout Prompt* node. The
selected box's stored value is shown above the canvas.

**Layer order:** the top row of the layer list is drawn first (underneath); the bottom row is drawn
last (on top). LoRAs and layout-prompt items are emitted in exactly this order, minus hidden rows, so
put the background first and the foreground last, as with *Create Bounding Boxes*. Groups only
organise rows; they change nothing about the output.

**What a box does** is decided by its fields. It is a layout-prompt item when it has a caption (or,
for a `text` box, text to render); each LoRA in its list with a file and a non-zero strength is
applied inside it. A box with only LoRAs leaves the prompt alone; a box with only a caption leaves the
model alone.

| action | how |
|---|---|
| new box | **Box** tool (B), drag on the canvas |
| select | **Select** tool (V), click a box or its layer row; click the background to deselect |
| move | drag the box body (Select tool) |
| resize | drag one of the 8 handles of the selected box |
| square | hold **Shift** while drawing or resizing |
| resize from centre | hold **Alt** while drawing or resizing |
| bypass snapping | hold **Ctrl** (**Cmd** on macOS) while dragging, with Snap on |
| nudge 1 px / 10 px | **Arrow keys** / **Shift + Arrow keys** |
| delete selection | **Delete** or **Backspace** (locked boxes are kept), or **Delete box** in properties |
| undo / redo | **Ctrl/Cmd + Z** / **Ctrl/Cmd + Shift + Z** (or **Ctrl + Y**) |
| multi-select | **Shift**- or **Ctrl/Cmd**-click boxes or rows |
| deselect / cancel | **Esc** |
| reorder | ▲ / ▼ buttons, or drag a row by its ⠿ grip |

The keyboard shortcuts work while the canvas has focus (click it first). Locked boxes cannot be moved
or resized on the canvas.

### LoRAs on a box

Each box has a list of LoRAs. A row shows the LoRA's file name, its strength and a small graph of its
strength over the run. Click the name to pick a file from the full list of paths (type to filter).
**+ Add LoRA** adds a row; ✕ removes it. Each LoRA is applied inside the box as its own gated delta,
with its own strength and curve. Workflows saved with one LoRA per box load as a one-row list.

### Strength curves

Click a LoRA's small graph to open its curve. Left to right is 0–100% of the steps, bottom to top is
0–100% of that LoRA's strength (a strength of −2 at 50% applies −1). New LoRAs start at full strength
for the whole run.

| action | how |
|---|---|
| add a point | click an empty spot in the graph (snaps to 5% steps when Snap is on; **Ctrl/Cmd** bypasses) |
| move a point | drag it (it stays between its neighbours) |
| remove a point | double-click or right-click it, or select it and press **Delete** with the graph focused |
| bend a segment | drag its ◆ handle up or down (ease in / ease out) |
| hard step | select a segment's ◆ handle and press **Hold**: its start value holds until the next point, then jumps |
| straighten | select a segment and press **Straight** |
| presets | **Full**, **Fade in**, **Fade out**, **Stop at 50%** |

Shape-type LoRAs (a body or weight slider) usually need strength in the early steps, where shape is
decided; character LoRAs that fight the layout can fade in later. Curve edits are undoable like any
other edit. LoRAs saved before curves existed show a dashed graph: they keep using the node's
`ramp_start`/`ramp_end` until you edit their curve.

## Layout discovery (after ComfyUI updates)

The node needs to know which positions of the transformer sequence are image tokens, whether the
prompt comes before or after them, whether image tokens are in plain row-by-row order, and where an
edit pipeline's source-image tokens go. The **MOSS-Image Layout Discovery** node measures this on your
loaded model and reports the values to set. It changes nothing; it only measures.

The values come from ComfyUI's MOSS-Image code, not from the weights, so one run covers every
MOSS-Image checkpoint. The main node's defaults are the values measured on ComfyUI's 0.44-era
implementation; re-run the discovery after a ComfyUI update, and if it reports different values, set
them in the main node's advanced options.

1. Add **MOSS-Image Layout Discovery** (`model_patches/moss_image`). Connect your MOSS-Image MODEL
   (straight from the loader, without other patches), CLIP and VAE.
2. Connect any image (for example a *Load Image* node) to `edit_image`, so the report also covers
   edit workflows.
3. Queue it on its own. It runs about ten single-step passes at the canvas size (no full
   generations), so it takes roughly as long as one or two normal generations.
4. Read the report on the node. It ends with the values to set:

   ```
   ---- set these on MOSS-Image Regional LoRA + Layout Prompt ----
   advanced options:
     text_position = ...
     image_order   = ...
     source_tokens = ...
   inputs:
     vae_downscale = ...
     patch_size    = ...
   ```

5. On **MOSS-Image Regional LoRA + Layout Prompt**, open the advanced options and set
   `text_position`, `image_order` and `source_tokens` to those values; set `vae_downscale` and
   `patch_size` to the reported numbers. They are saved with the workflow; no restart is needed.
6. Check the preview images: in each `diff_r0_c…` image, the bright spot (where nudging one image
   token changed the model's one-step prediction) should sit inside the green square (where
   row-by-row order puts it), in the top-left corner. The report says whether it did. Both probes
   must confirm. (`prediction` and `probe_r0_c…` are blurry one-step predictions; that's expected.)
7. If any value says `NOT ESTABLISHED`, or a `problems` section appears, don't guess: open an issue
   with the whole report and the images. If the change in the probe images isn't clear, raise the
   advanced `probe_strength`.

Advanced options on the discovery node: `edit_methods` (the `reference_latents_method` values tested
in the edit pass; the default covers both kinds of format of ComfyUI's *Text Encode MOSS-Image Edit*
node), `probe_strength`, `patch_size` (0 = infer) and `blocks` (the transformer block list, if it
can't be detected).

**Without the UI:** `tools/discover_moss_layout.py` runs the same passes from a terminal, using
ComfyUI's Python environment, and saves the images to a folder. Run it from the ComfyUI folder with
ComfyUI closed, for example with separate model files (full paths also work):

```bash
python custom_nodes/comfyui-moss-image-fancy-bbox/tools/discover_moss_layout.py --mode all --unet MOSS_MODEL.safetensors --clip MOSS_TEXT_ENCODER.safetensors --vae MOSS_VAE.safetensors --edit-image input/ANY_IMAGE.png --edit-method default --edit-method index_timestep_zero --out moss_probe
```

Use `--checkpoint FILE` for an all-in-one checkpoint and `--help` for every option.

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
