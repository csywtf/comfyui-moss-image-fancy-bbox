"""The prompt writer (Character Masks' optional prompt_llm input), with a stand-in for ComfyUI's CLIP."""

import json
import os
import sys
import types

import pytest
import torch

from moss_regional_lora import character_masks as cm
from moss_regional_lora import prompt_writer as pw

H, W = 100, 200


class FakeClip:
    """Records how it is called, like ComfyUI's CLIP (tokenize / generate / decode)."""

    def __init__(self, replies=None, vision=True, no_presence=False):
        self.replies = list(replies or [])
        self.vision, self.no_presence = vision, no_presence
        self.tokenized, self.generated = [], []

    def tokenize(self, text, **kw):
        if kw.get("images") and not self.vision:
            raise RuntimeError("no vision tower")
        self.tokenized.append((text, kw))
        return {"t": text}

    def generate(self, tokens, **kw):
        if self.no_presence and "presence_penalty" in kw:
            raise TypeError("unexpected keyword argument 'presence_penalty'")
        self.generated.append(kw)
        return self.replies.pop(0) if self.replies else "close-up of a detailed part"

    def decode(self, ids):
        return ids


def mask(x0, y0, x1, y1):
    m = torch.zeros(H, W, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def character(id, caption, parts, stored):
    info = cm.BoxInfo(id, id, (0, 0, 1, 1), "character", caption, "#fff", ())
    d = cm.resolve_detail(info, stored)
    return cm.CharacterMask(id=id, name=id, label="character", caption=caption, color="#fff", loras=(), detail=d,
                            rect=(0, 0, W, H), outline=torch.zeros(H, W, dtype=torch.bool), outline_found=True,
                            parts=parts, found={n: bool(m.any()) for n, m in parts.items()})


def scene():
    real = character("real", "a realistic woman, purple nails, black t-shirt",
                     {"face": mask(10, 10, 40, 40), "left_hand": mask(60, 60, 80, 80),
                      "eyes": torch.zeros(H, W, dtype=torch.bool)},                       # eyes: not found
                     {"parts": {"left_hand": {"on": True}, "face": {"full_prompt": "a freckled face"}}})
    off = character("off", "a lamp", {"face": mask(150, 10, 170, 30)}, {"enabled": False})
    return cm.CharacterMasks([cm.FrameMasks(W, H, [real, off]), cm.FrameMasks(W, H, [
        character("real", real.caption, real.parts, {"parts": {"left_hand": {"on": True},
                                                               "face": {"full_prompt": "a freckled face"}}}),
        off])])


def test_writes_only_missing_prompts_of_found_parts_and_stores_them_in_every_image():
    masks = scene()
    clip = FakeClip(["Prompt: \"close-up of a detailed realistic hand, short purple fingernails.\""])
    img = torch.rand(2, H, W, 3)
    written = pw.fill_missing(masks, img, clip, pw.WriterSettings(seed=7, temperature=0.6))
    assert written == {"real": {"left_hand": "close-up of a detailed realistic hand, short purple fingernails"}}
    assert len(clip.tokenized) == 1                                  # face has one, eyes not found, off: Detail off
    text, kw = clip.tokenized[0]
    assert "a realistic woman, purple nails" in text and "left side of the image" in text
    assert kw["thinking"] is False and kw["system_prompt"] == pw.default_system_prompt()
    assert kw["images"][0].shape[0] == 1 and kw["images"][0].shape[-1] == 3        # the part's crop, as a batch
    g = clip.generated[0]
    assert (g["seed"], g["temperature"], g["max_length"], g["do_sample"]) == (7, 0.6, 160, True)
    for f in masks.frames:
        assert f.boxes[0].detail.parts["left_hand"].full_prompt.startswith("close-up of a detailed realistic hand")
        assert f.boxes[0].detail.parts["face"].full_prompt == "a freckled face"       # untouched


def test_text_only_models_and_settings():
    masks = scene()
    clip = FakeClip(vision=False, no_presence=True)
    pw.fill_missing(masks, torch.rand(2, H, W, 3), clip, pw.WriterSettings(system_prompt="  Be brief.  "))
    text, kw = clip.tokenized[0]
    assert "images" not in kw and "image shows" not in text            # retried without the image
    assert kw["system_prompt"] == "Be brief."
    clip = FakeClip()
    pw.fill_missing(scene(), torch.rand(2, H, W, 3), clip, pw.WriterSettings(use_image=False, temperature=0))
    assert "images" not in clip.tokenized[0][1] and clip.generated[0]["do_sample"] is False


def test_a_clip_that_cannot_generate_is_refused_and_errors_are_reported():
    class EncoderOnly:
        def tokenize(self, *a, **k):
            return {}
    with pytest.raises(ValueError, match="can't generate"):
        pw.fill_missing(scene(), torch.rand(2, H, W, 3), EncoderOnly(), pw.WriterSettings())

    class Broken(FakeClip):
        def generate(self, tokens, **kw):
            raise RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        pw.fill_missing(scene(), torch.rand(2, H, W, 3), Broken(), pw.WriterSettings())


def test_clean_and_crop():
    assert pw.clean("<think>\nhmm\n</think>\n\nclose-up of an eye.") == "close-up of an eye"
    assert pw.clean("\n`Final prompt: “soft curly red hair”`\nextra line") == "soft curly red hair"
    assert pw.clean("   ") == ""
    img = torch.rand(H, W, 3)
    c = pw.part_crop(img, mask(60, 60, 80, 80), min_side=50)
    assert c.shape[-1] == 3 and min(c.shape[:2]) >= 32 and c.shape[0] <= H and c.shape[1] <= W
    big = torch.rand(2000, 1000, 3)
    m = torch.zeros(2000, 1000, dtype=torch.bool)
    m[100:1900, 100:900] = True
    assert max(pw.part_crop(big, m, max_side=768).shape[:2]) == 768


def test_default_system_prompt_is_the_editors():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "prompt_writer_system.txt")
    assert pw.default_system_prompt() == open(path, encoding="utf-8").read().strip()
    assert "one line" in pw.default_system_prompt()
