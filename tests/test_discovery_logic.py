"""Unit tests for the discovery tool's decision logic, on synthetic sequences.

These do not establish anything about MOSS-Image. They check that, *if* the model builds its
sequence in a given way, the tool's comparisons reach the right conclusion for every layout."""

import pytest
import torch
import torch.nn as nn

import toy
from toy import ToyDiT
from moss_regional_lora import discovery as disc
from moss_regional_lora.layout import Layout

D = 8
N_IMG = 16


def prompt_tokens(n, seed):
    return torch.randn(n, D, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("text_position", ["prefix", "suffix"])
def test_text_position_is_recovered(text_position):
    img = torch.randn(N_IMG, D, generator=torch.Generator().manual_seed(1))
    short, long = prompt_tokens(7, 2), prompt_tokens(300, 3)
    a = torch.cat([short, img] if text_position == "prefix" else [img, short])
    b = torch.cat([long, img] if text_position == "prefix" else [img, long])
    pre, suf = disc.identical_runs(a, b)
    pos, notes = disc.decide_text_position(a.shape[0], b.shape[0], 7, 300, pre, suf, N_IMG)
    assert pos == text_position, notes


def test_text_position_rejects_partial_text_entry():
    pos, notes = disc.decide_text_position(100, 150, 10, 300, 0, 50, N_IMG)
    assert pos is None and "one-for-one" in notes[-1]


def test_text_position_inconclusive_when_image_tokens_differ():
    pos, _ = disc.decide_text_position(23, 316, 7, 300, 0, 3, N_IMG)
    assert pos is None


def _sequences(text_position, src_pos, n_text_a=9, n_text_b=40, n_src=12, text_shift=0.25):
    """Block-0 inputs: without a source image (prompt A), and with one (prompts A and B).
    With a source image the text tokens change (``text_shift``), as MOSS-Image's shifted text position
    encodings do; image and source tokens never depend on the prompt."""
    img = torch.randn(N_IMG, D, generator=torch.Generator().manual_seed(1))
    src = torch.randn(n_src, D, generator=torch.Generator().manual_seed(4))
    txt_a, txt_b = prompt_tokens(n_text_a, 2), prompt_tokens(n_text_b, 3)

    def build(txt, with_src):
        first, second = (txt, img) if text_position == "prefix" else (img, txt)
        if not with_src:
            return torch.cat([first, second])
        txt2 = txt + text_shift
        first, second = (txt2, img) if text_position == "prefix" else (img, txt2)
        parts = ([src] if src_pos == "start" else []) + [first] + ([src] if src_pos == "between" else []) + \
                [second] + ([src] if src_pos == "end" else [])
        return torch.cat(parts)

    return build(txt_a, False), build(txt_a, True), build(txt_b, True), img, n_text_a, n_src


def _decide(text_position, without, with_a, with_b, img, n_text, n_src):
    img_start = 0 if text_position == "suffix" else n_text
    image_block = without[img_start:img_start + N_IMG]
    offset = disc.find_block(with_a, image_block, [0, n_text, n_src, n_text + n_src, N_IMG, N_IMG + n_src])
    pre, suf = disc.identical_runs(with_a, with_b)
    return disc.decide_source_position(without.shape[0], with_a.shape[0], n_src, N_IMG, n_text, offset,
                                       (pre, with_a.shape[0] - suf))


@pytest.mark.parametrize("text_position", ["prefix", "suffix"])
@pytest.mark.parametrize("src_pos", ["start", "between", "end"])
@pytest.mark.parametrize("text_shift", [0.0, 0.25], ids=["text-unchanged", "text-shifted"])
def test_source_position_is_recovered(text_position, src_pos, text_shift):
    without, with_a, with_b, img, n_text, n_src = _sequences(text_position, src_pos, text_shift=text_shift)
    pos, notes = _decide(text_position, without, with_a, with_b, img, n_text, n_src)
    assert pos == src_pos, notes
    # and the node's own layout agrees on where the image block is in the with-source sequence
    s = Layout(text_position, "raster", pos).image_start(n_text, n_src)
    assert torch.equal(with_a[s:s + N_IMG], img)


def test_regression_moss_edit_report():
    """The first real report: suffix layout, source appended after the text, text tokens changed by the
    source image. The old comparison (with vs. without source, same prompt) found only the image
    identical and gave up; the prompt-swap comparison resolves it."""
    without, with_a, with_b, img, n_text, n_src = _sequences("suffix", "end", text_shift=0.25)
    pre_old, suf_old = disc.identical_runs(without, with_a)
    assert (pre_old, suf_old) == (N_IMG, 0)          # what the report showed: 4096 leading, 0 trailing
    assert _decide("suffix", without, with_a, with_b, img, n_text, n_src)[0] == "end"


def test_source_none_and_count_mismatch():
    assert disc.decide_source_position(25, 25, 12, N_IMG, 9, None, None)[0] == "none"
    assert disc.decide_source_position(25, 40, 12, N_IMG, 9, 0, (16, 25))[0] is None      # 15 added, 12 expected
    assert disc.decide_source_position(25, 37, 12, N_IMG, 9, None, (16, 25))[0] is None   # image block not found


def test_combine_source_positions():
    assert disc.combine_source_positions(["none", "end"]) == ("end", disc.combine_source_positions(["end"])[1])
    assert disc.combine_source_positions(["none", "none"])[0] == "none"
    assert disc.combine_source_positions(["start", "end"])[0] is None
    assert disc.combine_source_positions([None])[0] is None


def test_interpret_probe():
    assert disc.interpret_probe((0, 1), (0, 1), 8, 8)[0] is True
    ok, why = disc.interpret_probe((0, 0), (0, 0), 8, 8)
    assert ok and "diagonal" in why
    assert "COLUMN-MAJOR" in disc.interpret_probe((1, 0), (0, 1), 8, 8)[1]
    assert "left-right" in disc.interpret_probe((0, 6), (0, 1), 8, 8)[1]
    assert "top-bottom" in disc.interpret_probe((7, 1), (0, 1), 8, 8)[1]
    assert disc.interpret_probe((3, 3), (0, 1), 8, 8)[0] is False


def test_strongest_cell():
    diff = torch.zeros(64, 96)
    diff[16:32, 48:64] = 1.0
    (r, c), peak, median = disc.strongest_cell(diff, 16)
    assert (r, c) == (1, 3) and peak == 1.0 and median == 0.0


def test_render_config_marks_unknowns():
    f = disc.Findings(vae_downscale=16, patch_size=1, text_position="suffix")
    out = disc.render_config(f, edit_ran=False)
    assert "text_position = suffix" in out
    assert "image_order   = NOT ESTABLISHED" in out
    assert "patch_size    = 1" in out
    assert "TEXT_POSITION = 'suffix'" in out and "IMAGE_ORDER = None  # NOT ESTABLISHED" in out
    assert "'none' is exact" in out          # edit pass skipped: explain what that means


def test_parse_edit_methods():
    assert disc.parse_edit_methods("default, index_timestep_zero") == [None, "index_timestep_zero"]
    assert disc.parse_edit_methods("") == [None]


class _TwoLists(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Module()
        self.text.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])


def test_find_blocks_prefers_named_then_single_direct_list(capsys):
    name, m = disc.find_blocks(ToyDiT(Layout("suffix", "raster", "none")), None)
    assert name == "blocks"
    name, m = disc.find_blocks(_TwoLists(), None)
    assert name == "layers" and len(m) == 3
    assert "text.blocks" in capsys.readouterr().out      # reported, not silently picked


def test_recorder_picks_main_pass_and_applies_probe():
    model = ToyDiT(Layout("suffix", "raster", "none")).double()
    calls = []

    def probe(x):
        calls.append(x.shape)
        return x + 1.0

    rec = disc.Recorder(model, model.blocks[0], probe)
    # a block call outside any model call (conditioning precompute) is ignored
    model.blocks[0](torch.zeros(1, 4, toy.DIM, dtype=torch.float64))
    lat = torch.zeros(1, toy.LATENT_C, 8, 8, dtype=torch.float64)
    ctx = torch.zeros(1, 5, toy.DIM, dtype=torch.float64)
    model(lat, context=ctx)
    rec.remove()
    main = rec.main_call(0)
    assert main["shape"] == (1, toy.N_IMG + 5, toy.DIM)
    assert rec.forwards[0]["context"] == (1, 5, toy.DIM)
    assert len(calls) == 2
