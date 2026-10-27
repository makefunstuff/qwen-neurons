import pytest
import torch

from qwen_neurons.steering import (Knob, SteeringHooks, active_knobs, chaos_level, chaos_zone, knob_stats,
                                   layer_pushes, median_resid_norm, parse_knobs, push_vector, row_pushes)


def test_parse_knobs():
    assert parse_knobs("14:8949:+1,13:7956:-1") == [Knob(14, 8949, 1.0), Knob(13, 7956, -1.0)]
    assert parse_knobs("") == []


def test_active_knobs_drops_off_and_zero():
    raw = [{"layer": 1, "neuron": 2, "dose": 0.5}, {"layer": 1, "neuron": 3, "dose": 0},
           {"layer": 2, "neuron": 4, "dose": 1, "enabled": False}]
    assert active_knobs(raw) == [Knob(1, 2, 0.5)]


def test_median_skips_token_zero():
    h = torch.zeros(5, 4)
    h[0] = 1000          # the attention sink
    h[1:, 0] = torch.tensor([1.0, 2.0, 3.0, 10.0])
    assert median_resid_norm(h) == pytest.approx(2.0)  # torch median: lower of the two middle values
    assert median_resid_norm(h.unsqueeze(0)) == pytest.approx(2.0)
    assert median_resid_norm(torch.full((1, 3), 2.0)) == pytest.approx(12 ** 0.5)  # a single token is used as is


def test_push_vector_has_the_recipe_size_and_direction():
    col = torch.tensor([3.0, 4.0])                     # |col| = 5
    v = push_vector(col, med=10.0, dose=0.5)
    assert v.norm().item() == pytest.approx(5.0)       # dose * med
    assert torch.allclose(v / v.norm(), col / 5)
    assert torch.allclose(push_vector(col, 10.0, -1.0), -2 * v)


def test_push_is_independent_of_column_scale():
    col = torch.randn(64)
    assert torch.allclose(push_vector(col, 7.0, 1.0), push_vector(col * 100, 7.0, 1.0), atol=1e-5)


def test_knob_stats():
    s = knob_stats(Knob(14, 8949, -0.5), torch.tensor([0.3, 0.4]), 10.0)
    assert s["col_norm"] == pytest.approx(0.5)
    assert s["push"] == pytest.approx(5.0)
    assert s["equiv_activation"] == pytest.approx(-10.0)


def test_layer_pushes_sum_knobs_on_one_layer():
    cols = {(3, 0): torch.tensor([1.0, 0.0]), (3, 1): torch.tensor([0.0, 2.0]), (5, 0): torch.tensor([1.0, 1.0])}
    p = layer_pushes([Knob(3, 0, 1.0), Knob(3, 1, -0.5), Knob(5, 0, 1.0)], cols, {3: 4.0, 5: 2.0})
    assert torch.allclose(p[3], torch.tensor([4.0, -2.0]))
    assert p[5].norm().item() == pytest.approx(2.0)


def test_row_pushes_zero_for_unsteered_rows():
    cols = {(3, 0): torch.tensor([1.0, 0.0])}
    p = row_pushes([[], [Knob(3, 0, 1.0)]], cols, {3: 4.0})
    assert p[3].shape == (2, 2)
    assert torch.allclose(p[3][0], torch.zeros(2))
    assert torch.allclose(p[3][1], torch.tensor([4.0, 0.0]))


def test_hooks_add_push_to_every_token_and_clean_up():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    x = torch.randn(2, 4, 3)                           # batch 2, 4 tokens
    base = lin(x)
    vec = torch.tensor([[0.0, 0.0], [1.0, -2.0]])      # row 0 unsteered
    with SteeringHooks({0: lin}, {0: vec}):
        out = lin(x)
    assert torch.allclose(out[0], base[0])
    assert torch.allclose(out[1], base[1] + torch.tensor([1.0, -2.0]))
    assert torch.allclose(lin(x), base)                # hooks removed
    with SteeringHooks({0: lin}, {0: torch.tensor([1.0, 1.0])}):
        assert torch.allclose(lin(x), base + 1)        # a plain [hidden] push goes to every row


def test_chaos_zones():
    assert chaos_level([Knob(1, 1, 1.0), Knob(2, 2, -1.0)]) == 2.0
    assert chaos_zone(0.8) == "ok" and chaos_zone(1.2) == "warm" and chaos_zone(2.0) == "loop"
