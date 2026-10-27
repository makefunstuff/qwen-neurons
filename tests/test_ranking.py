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


def test_writer_scores_find_the_neuron_that_writes_the_difference():
    from qwen_neurons.ranking import rank_neurons, writer_scores
    torch.manual_seed(0)
    W = torch.randn(16, 40)
    diff = (-3 * W[:, 7]).unsqueeze(0)               # set A sits along -W[:, 7]
    cos = writer_scores([W], diff)
    assert cos.shape == (1, 40) and int(cos[0].abs().argmax()) == 7 and cos[0, 7] < -0.99
    a, b = torch.zeros(3, 1, 40), torch.zeros(3, 1, 40)
    res = rank_neurons(a, b, [5], metric="writer", top_k=1, writer=cos)
    assert res["top"][0]["neuron"] == 7 and res["top"][0]["dose"] < 0 and res["top"][0]["layer"] == 5


def test_writer_scores_ignore_outlier_dimensions_when_scaled():
    from qwen_neurons.ranking import writer_scores
    torch.manual_seed(1)
    W = torch.randn(16, 40)
    W[0, 3] = 50.0                                    # neuron 3 writes a huge outlier dimension
    diff = torch.zeros(1, 16); diff[0, 0] = 100.0     # the raw difference is all outlier...
    diff[0] += 2 * W[:, 9] / W[:, 9].norm()           # ...plus the real persona direction
    raw = writer_scores([W], diff)
    std = torch.ones(1, 16); std[0, 0] = 1000.0       # dimension 0 swings wildly on every token
    scaled = writer_scores([W], diff, std)
    assert int(raw[0].abs().argmax()) == 3
    assert int(scaled[0].abs().argmax()) == 9
