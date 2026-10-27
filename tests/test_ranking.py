import pytest
import torch

from qwen_neurons.ranking import contrast, rank_neurons, text_means


def test_text_means_skip_first_token():
    acts = torch.tensor([[100.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(text_means(acts), torch.tensor([2.0, 3.0]))


def sets():
    torch.manual_seed(1)
    a = torch.randn(8, 2, 50) * 0.1
    b = torch.randn(8, 2, 50) * 0.1
    a[:, 1, 7] += 2.0      # fires on set A, layer index 1
    b[:, 0, 3] += 1.0      # fires on set B, layer index 0
    return a, b


def test_contrast_finds_planted_neurons():
    a, b = sets()
    c = contrast(a, b)
    assert c["diff"][1, 7] == pytest.approx(2.0, abs=0.15)
    assert c["d"][1, 7] > 5 and c["d"][0, 3] < -5


def test_rank_neurons_top_and_dose_sign():
    a, b = sets()
    r = rank_neurons(a, b, layers=[13, 14], top_k=2, per_layer_k=1)
    top = {(c["layer"], c["neuron"]): c for c in r["top"]}
    assert set(top) == {(14, 7), (13, 3)}
    assert top[(14, 7)]["dose"] > 0 and top[(13, 3)]["dose"] < 0
    assert [p["layer"] for p in r["per_layer"]] == [13, 14]
    assert r["per_layer"][1]["top"][0]["neuron"] == 7


def test_resid_metric_weights_by_column_norm():
    a, b = sets()
    norms = torch.ones(2, 50)
    norms[0, 3] = 10.0
    r = rank_neurons(a, b, layers=[13, 14], metric="resid", top_k=1, col_norms=norms)
    assert (r["top"][0]["layer"], r["top"][0]["neuron"]) == (13, 3)
    with pytest.raises(ValueError):
        rank_neurons(a, b, layers=[13, 14], metric="resid")
    with pytest.raises(ValueError):
        rank_neurons(a, b, layers=[13, 14], metric="nope")
