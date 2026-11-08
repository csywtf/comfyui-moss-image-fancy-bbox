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

**MOSS-Image Character Masks** follows the generation: it finds each box's character in the image with
SAM3 and masks its face, eyes, hands, hair and skin. **MOSS-Image Regional Detailer** then redraws each
of those parts with that character's own caption and LoRAs (see *Character masks* and *Regional
detailer*).

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
                                 (draw boxes here)        ──width───▶ Empty latent (width)  ──▶ KSampler (latent_image)
                                                          ──height──▶ Empty latent (height)
Load CLIP (type moss_image) ──CLIP──▶ CLIP Text Encode
Load VAE ──VAE──▶ VAE Decode
```

It replaces *Create Bounding Boxes* → *MOSS-Image Layout Prompt*. Place it on the MODEL line between
the loader (and any ordinary `LoraLoader`s) and the sampler, and connect its `prompt` output to your
positive text encode (convert that node's text to an input if needed). CLIP and the VAE stay wired as
usual; the node never touches them, and a LoRA's text-encoder keys are skipped (put trigger words in
a caption instead).
Set the image size in the editor's **Canvas** panel and connect the `width`/`height` outputs to your
empty latent node, so the latent always matches the canvas the boxes are drawn on (the sampler itself
takes no size; it samples the latent it is given). It is built to sit alongside normal `LoraLoader` nodes, ComfyUI's low-VRAM weight casting and
offload, and `torch.compile`, without turning any of them off (the hooks only read each layer's input
and output, and compile without graph breaks).

Outputs: `model`, `prompt`, `width`/`height` (the canvas size, for the empty latent node) and
`regions` (the boxes plus the canvas size, for *MOSS-Image Character Masks*).

Settings: all of them are in the editor's **Canvas** panel (shown when no box is selected). On the
node they are ordinary inputs that the editor hides, so they are saved with the workflow and sent
through the API under the names below. A setting whose input is wired to another node shows as
read-only in the panel, since the wired value is the one used.

| setting | meaning |
|---|---|
| `width`, `height` | output canvas in pixels; must match the latent being sampled (checked at run time). Rounded to whole 16 px latent cells. The token grid itself is read from the latent, so there is no VAE-downscale or patch-size setting |
| `caption` | description of the whole image: the layout prompt's summary caption. With no captioned boxes, `prompt` is just this text |
| `strict` | on: refuse to apply LoRAs while the layout is unverified and raise on any alignment problem. Off: pass the model through untouched (with a warning) instead. Prompt-only boxes work either way |
| `regions` | the region list as JSON; drawn and edited by the editor, viewable and pasteable with **Edit JSON** (see *Region editor*) |
| `ramp_start`, `ramp_end` | *Advanced*, legacy: a linear fade-in used only by LoRAs **without** their own strength curve (workflows saved before curves). New LoRAs get their own curve (see *Strength curves*) |
| `steps` | *Advanced*, fallback: the curves follow the sampler's own step count; this is used only if a sampler doesn't report one |
| `text_position`, `image_order`, `source_tokens` | *Advanced*: the layout values reported by the *MOSS-Image Layout Discovery* node. Their defaults (`suffix`, `raster`, `end`) were measured on a real MOSS-Image run. "not set" falls back to `moss_regional_lora/layout_config.py` (unset by default) |

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
them. The hooks live on the node's own copy of the model, so ComfyUI's cache works as usual: queueing
an unchanged workflow reuses the cached image, and the node re-runs (with everything after it) only
when one of its inputs, the upstream model, or one of the LoRA files it uses (by size and modification
time) changes.

kohya (`lora_down`/`lora_up`/`alpha`) and peft/diffusers (`lora_A`/`lora_B`) LoRAs are supported.
Keys that do not resolve to a linear layer (text-encoder keys, DoRA, LoHa, ...) are logged and
skipped, and the count is shown under the editor after a run.

## Region editor

The editor fills the node. It has three panels, with a floating toolbar on the canvas:

```
[layers] [ [tools]       canvas        ] [properties]
```

The `regions` JSON (a hidden input on the node) is the source of truth: the editor only draws it and
writes it back after each edit. **Edit JSON** (Canvas panel) shows it for copying, and applies pasted
or hand-edited JSON (undoable); JSON that doesn't parse is refused with the reason. If a workflow
arrives with invalid JSON, the editor turns red and becomes read-only (except the settings and Edit
JSON, to fix it); it never overwrites the text.

- **Tools** (floating at the top left of the canvas): **Select** (V) clicks, moves and resizes boxes; clicking the background deselects.
  **Box** (B) draws a new box, then switches back to Select with the new box selected.
- **Layers**: one row per box, with colour, name, badges (`2L` = two LoRAs, `✎` = in the
  layout prompt, ⚠ = extends past the canvas), then the visibility (eye) and lock toggles and
  move up / down. Click a row to select the box.
  **+ Group** makes a group (with the selected boxes in it); group headers collapse, rename, show all,
  hide all, move and delete all.
- **Canvas**: the output canvas with a grid (a third of the canvas by default) and the centre lines.
- **Properties** (right): for the selected box, its name, colour, box coordinates
  (editable), group, layout-prompt fields, and its **LoRA list**. With nothing selected it shows the
  **Canvas** panel: image size (with a swap button), caption, strict, a collapsed *Advanced* group,
  **Edit JSON**, and drawing aids: grid, snap, and an optional reference image from ComfyUI's `input`
  folder with an opacity slider (a drawing aid only).

**Caching:** queueing sends the backend only what affects the image (box positions and order, labels,
captions, text, palettes, LoRAs and curves; hidden boxes are left out). Names, colours, locks, groups
and the grid / reference-image settings stay in the workflow but aren't sent, so changing them doesn't
regenerate the image.

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

## Character masks

**MOSS-Image Character Masks** finds each box's character in the generated image and masks its parts,
for detailing each character with its own prompt and LoRAs.

```
Regional LoRA + Layout Prompt ──regions──▶ MOSS-Image Character Masks ──character_masks──▶ (detailer)
VAE Decode ───────────────────image─────▶      (editor)              ──preview / mask──▶ Preview Image
Load Checkpoint (SAM3) ───model, clip───▶
```

**SAM3:** ComfyUI's own SAM3 (0.44 or newer). Meta's original release is gated; use the copy ComfyUI's
*Image Segmentation (SAM3)* template uses instead: `sam3.1_multiplex_fp16.safetensors` from
`https://huggingface.co/Comfy-Org/sam3.1` (`checkpoints/` folder), saved in `ComfyUI/models/checkpoints`
and loaded with the normal **Load Checkpoint** node (its MODEL to `sam3_model`, its CLIP to `sam3_clip`).

**What it does** for each visible box with Detail on:

1. **Outline:** SAM3's detector looks for people around the box, and the one that best matches the box
   (mostly inside it, covering a fair part of it) is the character's silhouette. If none matches, SAM3
   segments the main object inside the box instead; if that finds nothing too, the box itself is used
   (the editor says so). Small holes are filled and stray specks removed.
2. **Overlaps:** a pixel inside two outlines belongs to the box higher in the layer list (on top),
   unless you've marked it with a point (below).
3. **Parts:** SAM3 looks for each ticked part by name (`face`, `eye`, `hand`, `hair`, `skin`) on a crop
   around the character, with every other character greyed out, so it can't pick a neighbour's face in
   a hug. Each part's mask stays inside the box SAM3's detector drew for it (plus a little), so a face
   stops at the chin instead of running down the neck. A part keeps only its own character's pixels,
   plus a few just outside the outline (stray hair, edges).

If the image isn't the canvas size (an upscaled image), the boxes are scaled to it.

**Points (whose arm is it?):** in a hug, an arm lying across the other character often ends up in the
wrong outline or only partly in either. Select the box, then **Shift-click** the preview on that
character's arm or hand: a **+** point. SAM3 traces the region at the point and adds it to her outline,
and it's hers even where another box is on top. **Alt-click** puts a **−** point on something that isn't
hers; that region is removed from her outline. Points are drawn in their box's colour (+ filled, −
ringed); with a box selected, the other boxes' points are faint. Click a point to remove it; **Clear**
removes them all.
A point needs no type: SAM3 works out the region from where you click (add a second point if it only
takes the sleeve). Each box's + points also count as − points for every other box. Points are stored as
fractions of the image, and skin and hands then follow the corrected outlines. Like any detail setting,
they re-run only this node, not the image.

**The editor** (all of the node's inputs are hidden and set here):

- **Boxes** (left): read live from the connected main node, with the same names and colours. The
  checkbox turns Detail on or off; after a run, a dot per ticked part shows whether it was found
  (dashed red: not found).
- **Preview** (middle): the last run's image with the masks tinted by part; the selected box is bright,
  the others faint. Click a character to select its box.
- **Detail** (right, for the selected box): Detail on/off; a detail prompt added to every part's prompt
  (a trigger word, "green eyes"); whether to use the box's own LoRAs; detail-only LoRAs; and per part a
  checkbox, a prompt (empty: "detailed face" and so on) and a denoise (empty: the detailer's default),
  with "found" / "not found" after a run. With no box selected: SAM3's threshold, box margin and
  refinement passes.

Characters start with face and eyes on; other labels start off. The settings are stored on this node,
not in the main node's regions, so changing them never regenerates the image: thanks to ComfyUI's
cache, only this node re-runs, and SAM3 results are reused when only a prompt or denoise changed.

Outputs: `character_masks` (each box's part masks with its caption, LoRAs and detail settings, for the
detailer), `preview` (the image with the parts tinted) and `mask` (every part as one mask).

## Regional detailer

**MOSS-Image Regional Detailer** redraws each masked part with its character's prompt and LoRAs.

```
Character Masks ──character_masks──▶ MOSS-Image Regional Detailer ──image──▶ Save Image
VAE Decode ──────────image────────▶   (seed, steps, cfg, sampler,
Load Diffusion Model ──model──────▶    denoise, crop size, ...)
CLIP / VAE / your negative encode ─▶
```

Connect the **base** model (from the loader, or after ordinary LoRA loaders), not the Regional LoRA
node's output: each box's LoRAs are applied here, to that box's crops only. `negative` is optional
(connect your negative text encode; empty if not).

For each image, characters are detailed back to front (layer order) and each character's parts coarse to
fine (skin, hair, face, hands, eyes), each as its own pass on the image so far:

1. **Crop** the part (its mask grown by `mask_grow` and feathered by `feather`) with `padding` around it.
   A crop smaller than `crop_size` is upscaled by a whole number (2x, 3x, ..., Lanczos) up to `crop_size`
   and at most `max_upscale` times (advanced); small parts (eyes) take more context instead. A larger crop
   (hair, skin) is sampled at its own size. Either way it is padded, not resized, to whole 16 px cells.
   A part in pieces far apart (two hands on either side of the body, skin on an arm and a midriff) is
   detailed as one pass per group of pieces whose crops would overlap, so each gets its own upscaled
   crop; two eyes stay one pass.
2. **Re-sample** it with a noise mask, so only the part changes. Prompt: the box's caption, the box's
   detail prompt and the part's prompt (empty: "detailed face" and so on), all set in the Character Masks
   editor. LoRAs: the box's own (unless *Box LoRAs* is off) plus its detail-only LoRAs, applied to the
   model only, as the main node does. Denoise: the part's own, or the node's `denoise` (0.35). Seed:
   `seed` plus the pass number.
3. **Scale back** by area averaging (exact for a whole-number upscale) and **restore detail**: the crop
   is also decoded without being redrawn, and what that round trip lost (resizing, VAE) is added back, so
   only the model's changes are kept on top of the original's fine detail. At denoise 0 the image comes
   back unchanged. `restore_detail` (advanced, default 1) scales this; lower it if a high denoise leaves
   faint traces of the old features.
4. **Blend** it back with the feathered mask. Pixels outside every part's feathered mask are untouched,
   bit for bit.

Use the same `cfg`, `sampler_name` and `scheduler` as your KSampler.

**Check the masks first:** tick `stop_before_detailing` and queue. Everything up to the masks runs (the
Character Masks editor shows them) and the detailer stops quietly, so nothing after it runs. Adjust the
detail settings or points (only the masks re-run), then untick it and queue again: the image and the
masks come from ComfyUI's cache and only the detailer runs.

## Layout discovery (after ComfyUI updates)

The node needs to know which positions of the transformer sequence are image tokens, whether the
prompt comes before or after them, whether image tokens are in plain row-by-row order, and where an
edit pipeline's source-image tokens go. The **MOSS-Image Layout Discovery** node measures this on your
loaded model and reports the values to set. It changes nothing; it only measures.

The values come from ComfyUI's MOSS-Image code, not from the weights, so one run covers every
MOSS-Image checkpoint. The main node's defaults are the values measured on ComfyUI's 0.44-era
implementation; re-run the discovery after a ComfyUI update, and if it reports different values, set
them in the *Advanced* group of the main node's Canvas panel.

1. Add **MOSS-Image Layout Discovery** (`model_patches/moss_image`). Connect your MOSS-Image MODEL
   (straight from the loader, without other patches), CLIP and VAE.
2. Connect any image (for example a *Load Image* node) to `edit_image`, so the report also covers
   edit workflows.
3. Queue it on its own. It runs about ten single-step passes at the canvas size (no full
   generations), so it takes roughly as long as one or two normal generations.
4. Read the report on the node. It ends with the values to set:

   ```
   ---- set these on MOSS-Image Regional LoRA + Layout Prompt ----
   Advanced (in the editor's Canvas panel):
     text_position = ...
     image_order   = ...
     source_tokens = ...
   measured for reference (the node reads these from the latent; nothing to set):
     vae_downscale = ...
     patch_size    = ...
   ```

5. On **MOSS-Image Regional LoRA + Layout Prompt**, deselect all boxes, open *Advanced* in the
   Canvas panel and set `text_position`, `image_order` and `source_tokens` to those values. They are saved with the
   workflow; no restart is needed.
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

`node tests/web/editor_dom_smoke.mjs` and `node tests/web/masks_dom_smoke.mjs` additionally drive the
two editors in headless Chromium against a stub of ComfyUI's frontend API (needs Playwright). SAM3 is
replaced by a stand-in in the tests. With `COMFYUI_PATH=/path/to/ComfyUI`, `pytest`
also checks LoRA equivalence against ComfyUI's own `comfy.lora` code.
