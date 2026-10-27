"""The weights-only explorer math: logit lens, the search index, virtual weights and head scores."""
import math

import torch

from qwen_neurons import lens


def test_lens_promotes_the_aligned_token():
    torch.manual_seed(0)
    W_U = torch.randn(50, 16)
    g = torch.ones(16)
    cols = torch.stack([W_U[7], -W_U[3]], dim=1)         # neuron 0 writes token 7, neuron 1 writes "anti token 3"
    ti, tv, bi, bv = lens.lens_topk(W_U, g, cols, k=3)
    assert ti[0, 0] == 7 and bi[1, 0] == 3
    assert tv[0, 0] > 2 and bv[1, 0] < -2                # z-scores
    z = lens.lens_logits(W_U, g, cols)
    assert torch.allclose(z.mean(0), torch.zeros(2), atol=1e-5)


def test_layer_index_chunks_agree():
    torch.manual_seed(1)
    W_U, g, W = torch.randn(40, 8), torch.rand(8) + 0.5, torch.randn(8, 10)
    idx = lens.build_layer_index(W_U, g, W, k=4, chunk=3)
    ti, tv, bi, bv = lens.lens_topk(W_U, g, W, k=4)
    assert idx["top_ids"].shape == (10, 4) and torch.equal(idx["top_ids"], ti) and torch.equal(idx["bot_ids"], bi)


def test_search_finds_both_sides():
    vocab = ["a", " cookie", "cookies", " Cookie", "b", "dog", " conspiracy", "z"]
    ids = torch.zeros(2, 3, 4, dtype=torch.int32)
    z = torch.ones(2, 3, 4, dtype=torch.float16)
    ids[:] = 4
    ids[1, 2] = torch.tensor([1, 2, 3, 0], dtype=torch.int32)    # L1 N2 promotes three cookie tokens
    ids[0, 1] = torch.tensor([0, 1, 4, 4], dtype=torch.int32)    # L0 N1 promotes one
    bot = torch.full((2, 3, 4), 4, dtype=torch.int32)
    bot[0, 0, 0] = 6                                             # L0 N0 suppresses " conspiracy"
    index = {"top_ids": ids, "top_z": z * 3, "bot_ids": bot, "bot_z": -z * 3}
    hits = lens.search_index(index, lens.matching_tokens(vocab, "COOKIE"))
    assert (hits[0]["layer"], hits[0]["neuron"], hits[0]["side"], hits[0]["hits"]) == (1, 2, "+", 3)
    assert {(h["layer"], h["neuron"]) for h in hits} == {(1, 2), (0, 1)}
    neg = lens.search_index(index, lens.matching_tokens(vocab, " conspir"))
    assert neg[0]["side"] == "-" and (neg[0]["layer"], neg[0]["neuron"]) == (0, 0) and neg[0]["best_rank"] == 1
    assert lens.search_index(index, []) == [] and lens.matching_tokens(vocab, "  ") == []


def rmsnorm(x, g):
    return x / x.pow(2).mean().sqrt() * g


def test_virtual_weight_matches_a_real_push():
    """The link weight is the change of the target's gate/up input per unit dose (small pushes, linear regime)."""
    torch.manual_seed(2)
    H = 64
    W_down = torch.randn(H, 5)
    W_gate, W_up = torch.randn(7, H), torch.randn(7, H)
    g = torch.rand(H) + 0.5
    c = W_down[:, 3] / W_down[:, 3].norm()
    x = torch.randn(H)
    x = x - (x @ c) * c                                   # residual orthogonal to the push: its norm stays put
    d = 1e-3
    push = d * x.norm() * c                               # the steering recipe with med = ||x||
    for n in range(7):
        before, after = rmsnorm(x, g), rmsnorm(x + push, g)
        dg = (W_gate[n] @ after - W_gate[n] @ before) / d
        du = (W_up[n] @ after - W_up[n] @ before) / d
        gate, up = lens.in_direction(W_gate, W_up, lens.norm_gain(g - 1), n)
        t = lens.upstream(W_down, gate, up)
        assert math.isclose(float(t["gate"][3]), float(dg), rel_tol=2e-2, abs_tol=1e-3)
        assert math.isclose(float(t["up"][3]), float(du), rel_tol=2e-2, abs_tol=1e-3)
        t2 = lens.downstream(W_gate, W_up, lens.norm_gain(g - 1), W_down[:, 3])
        assert math.isclose(float(t2["gate"][n]), float(dg), rel_tol=2e-2, abs_tol=1e-3)
    links = lens.top_links(t2, 3)
    assert len(links) == 3 and abs(links[0]["w"]) >= abs(links[1]["w"]) >= abs(links[2]["w"])
    assert all(x["neuron"] != 2 for x in lens.top_links(t2, 7, skip=2)[:6])


def test_head_scores_pick_the_aligned_head():
    torch.manual_seed(3)
    H, heads, dh = 32, 4, 8
    W_O = torch.randn(H, heads * dh) * 0.1
    r = torch.randn(H)
    W_O[:, 2 * dh] = r * 5                                # head 2 can write straight along the read direction
    w = lens.head_writes(W_O, heads, r)
    assert max(range(heads), key=lambda h: w[h]) == 2
    W_V = torch.randn(heads * dh, H) * 0.1
    col = torch.randn(H)
    W_V[dh] = col * 5                                     # head 1's values read the neuron's output
    rd = lens.head_reads(W_V, heads, torch.ones(H), col)
    assert max(range(heads), key=lambda h: rd[h]) == 1
