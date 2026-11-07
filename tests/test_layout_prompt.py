"""Layout-prompt output: same text as ComfyUI's MOSS-Image Layout Prompt node for the same boxes."""

import json
import math
import re

import pytest

from moss_regional_lora.layout_prompt import build_layout_prompt
from moss_regional_lora.regions import RegionsError, parse_document


def comfy_moss_layout_prompt(prompt, element, width, height, character_boxes=""):
    """Transcribed from comfy_extras/nodes_moss_image.py, MOSSImageLayoutPrompt.execute (ComfyUI 0.44.0).
    `element` is Create Bounding Boxes' output: bbox in 0-1000 [y0, x0, y1, x1]."""
    characters = {int(i) for i in re.findall(r"\d+", character_boxes)}
    items = []
    for i, el in enumerate(element if isinstance(element, list) else []):
        label = "text" if el.get("type") == "text" else ("character" if i + 1 in characters else "object")
        item = {"label": label, "caption": el.get("desc", "")}
        if item["label"] == "text":
            item["text"] = el.get("text", "")
        if el.get("color_palette"):
            item["palette"] = el["color_palette"]
        y0, x0, y1, x1 = el["bbox"]
        item["bbox"] = [math.floor(round(y0 * height / 1000, 9)) - height // 2, math.floor(round(x0 * width / 1000, 9)) - width // 2,
                        math.ceil(round(y1 * height / 1000, 9)) - height // 2, math.ceil(round(x1 * width / 1000, 9)) - width // 2]
        items.append(item)
    if not items:
        return prompt
    layout = {"summary": {"caption": prompt.strip()}, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def doc(*regions):
    return json.dumps({"version": 1, "regions": list(regions)})


def ours(caption, *regions):
    return build_layout_prompt(caption, parse_document(doc(*regions)).prompt_items)


def test_matches_comfy_layout_prompt_node():
    W, H = 1000, 800      # 0-1000 coordinates map to whole pixels at this size
    elements = [
        {"type": "obj", "bbox": [0, 0, 500, 1000], "desc": "a stormy sky, deep blue", "color_palette": ["#112233", "#AABBCC"]},
        {"type": "obj", "bbox": [250, 100, 1000, 450], "desc": "Alice, red coat"},
        {"type": "obj", "bbox": [250, 550, 1000, 900], "desc": "Bob — 帽子"},
        {"type": "text", "bbox": [50, 300, 150, 700], "text": "HELLO", "desc": "neon sign"},
    ]
    expected = comfy_moss_layout_prompt("  Two friends under a storm  ", elements, W, H, character_boxes="2, 3")

    def px(b):
        y0, x0, y1, x1 = b
        return [y0 * H // 1000 - H // 2, x0 * W // 1000 - W // 2, y1 * H // 1000 - H // 2, x1 * W // 1000 - W // 2]

    got = ours("  Two friends under a storm  ",
               {"name": "sky", "bbox": px(elements[0]["bbox"]), "caption": "a stormy sky, deep blue",
                "palette": ["#112233", "#aabbcc"]},
               {"name": "Alice", "bbox": px(elements[1]["bbox"]), "label": "character", "caption": "Alice, red coat",
                "lora": "alice.safetensors"},
               {"name": "Bob", "bbox": px(elements[2]["bbox"]), "label": "character", "caption": "Bob — 帽子"},
               {"name": "sign", "bbox": px(elements[3]["bbox"]), "label": "text", "text": "HELLO", "caption": "neon sign"})
    assert got == expected


def test_no_items_returns_caption_unchanged():
    assert ours("  just a caption ") == "  just a caption "
    assert ours("cap", {"bbox": [0, 0, 10, 10], "lora": "a.safetensors"}) == "cap"


def test_lora_only_prompt_only_and_both():
    d = parse_document(doc(
        {"name": "lora-only", "bbox": [0, 0, 10, 10], "lora": "a.safetensors"},
        {"name": "prompt-only", "bbox": [0, 0, 10, 10], "caption": "a cat"},
        {"name": "both", "bbox": [0, 0, 10, 10], "caption": "a dog", "lora": "b.safetensors", "weight": 0.8},
        {"name": "hidden", "bbox": [0, 0, 10, 10], "caption": "x", "lora": "c.safetensors", "visible": False},
        {"name": "nothing", "bbox": [0, 0, 10, 10]},
    ))
    assert [r.name for r in d.loras] == ["lora-only", "both"]
    assert [p.name for p in d.prompt_items] == ["prompt-only", "both"]
    assert any("nothing" in n for n in d.notes)


def test_text_label_and_object_fields():
    out = json.loads(ours("c",
                          {"bbox": [0, 0, 10, 10], "label": "text", "text": "SALE", "caption": ""},
                          {"bbox": [0, 0, 10, 10], "label": "object", "text": "ignored", "caption": "lamp"}))
    assert out["items"][0] == {"label": "text", "caption": "", "text": "SALE", "bbox": [0, 0, 10, 10]}
    assert out["items"][1] == {"label": "object", "caption": "lamp", "bbox": [0, 0, 10, 10]}


def test_palette_normalised_and_capped():
    out = json.loads(ours("c", {"bbox": [0, 0, 1, 1], "caption": "x",
                                "palette": ["#aa0000", "", "#bb0000", "#cc0000", "#dd0000", "#ee0000", "#ff0000"]}))
    assert out["items"][0]["palette"] == ["#AA0000", "#BB0000", "#CC0000", "#DD0000", "#EE0000"]


def test_non_integer_boxes_are_widened():
    out = json.loads(ours("c", {"bbox": [-10.5, 3.2, 20.1, 40.0], "caption": "x"}))
    assert out["items"][0]["bbox"] == [-11, 3, 21, 40]


@pytest.mark.parametrize("bad", [{"label": "person"}, {"caption": 5}, {"palette": "red"}])
def test_malformed_prompt_fields_raise(bad):
    with pytest.raises(RegionsError):
        parse_document(doc({"bbox": [0, 0, 1, 1], "caption": "x", **bad}))
