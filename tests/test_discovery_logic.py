"""Unit tests for the discovery tool's decision logic, on synthetic sequences.

These do not establish anything about MOSS-Image. They check that, *if* the model builds its
sequence in a given way, the tool's comparisons reach the right conclusion for every layout."""

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn

import toy
from toy import ToyDiT
from moss_regional_lora.layout import Layout

_spec = importlib.util.spec_from_file_location(
    "discover_moss_layout", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "discover_moss_layout.py"))
disc = importlib.util.module_from_spec(_spec)
sys.modules["discover_moss_layout"] = disc
_spec.loader.exec_module(disc)

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


@pytest.mark.parametrize("text_position", ["prefix", "suffix"])
@pytest.mark.parametrize("src_pos", ["start", "between", "end"])
def test_source_position_is_recovered(text_position, src_pos):
    n_text = 9
    img = torch.randn(N_IMG, D, generator=torch.Generator().manual_seed(1))
    txt = prompt_tokens(n_text, 2)
    src = torch.randn(12, D, generator=torch.Generator().manual_seed(4))
    first, second = (txt, img) if text_position == "prefix" else (img, txt)
    without = torch.cat([first, second])
    parts = ([src] if src_pos == "start" else []) + [first] + ([src] if src_pos == "between" else []) + [second] + \
            ([src] if src_pos == "end" else [])
    with_src = torch.cat(parts)
    pre, suf = disc.identical_runs(without, with_src)
    pos, notes = disc.decide_source_position(without.shape[0], with_src.shape[0], 12, pre, suf, N_IMG, n_text,
                                             text_position)
    assert pos == src_pos, notes
    # and the node's own layout agrees on where the image block is in the with-source sequence
    lay = Layout(text_position, "raster", pos)
    s = lay.image_start(n_text, 12)
    assert torch.equal(with_src[s:s + N_IMG], img)


def test_source_none_and_count_mismatch():
    assert disc.decide_source_position(25, 25, 12, 25, 25, N_IMG, 9, "prefix")[0] == "none"
    assert disc.decide_source_position(25, 40, 12, 25, 0, N_IMG, 9, "prefix")[0] is None


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
    out = disc.render_config(f)
    assert "TEXT_POSITION = 'suffix'" in out
    assert "IMAGE_ORDER = None  # NOT ESTABLISHED" in out
    assert "patch_size = 1" in out


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
