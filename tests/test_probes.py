"""Steer lens, ablation hooks and the fires top-k bookkeeping, on toy tensors."""
import gzip
import json

import torch

from qwen_neurons import corpus, probes


def test_token_shift_and_kl():
    lp0 = torch.log_softmax(torch.zeros(3, 10), -1)
    logits = torch.zeros(3, 10)
    logits[:, 4] = 3.0                                    # the edit makes token 4 likely
    logits[:, 7] = -3.0
    lp1 = torch.log_softmax(logits, -1)
    sh = probes.token_shift(lp0, lp1, k=3)
    assert sh["rising"][0]["id"] == 4 and sh["rising"][0]["p1"] > sh["rising"][0]["p0"]
    assert sh["falling"][0]["id"] == 7 and sh["falling"][0]["dlogp"] < 0
    assert all(t["dlogp"] > 0 for t in sh["rising"]) and sh["kl"] > 0
    assert probes.token_shift(lp0, lp0)["kl"] == 0 and probes.token_shift(lp0, lp0)["rising"] == []


def test_floor_ignores_tokens_that_stay_unlikely():
    lp0 = torch.log_softmax(torch.tensor([[10.0, 0, -30, 0]]), -1)
    lp1 = torch.log_softmax(torch.tensor([[10.0, 0, -20, 0]]), -1)   # token 2: a huge log-prob gain, still ~1e-13
    assert probes.token_shift(lp0, lp1)["rising"] == []


def test_merge_top_keeps_the_best_docs():
    k, N = 2, 3
    v = torch.zeros(0, N)
    d = torch.zeros(0, N, dtype=torch.long)
    p = torch.zeros(0, N, dtype=torch.long)
    batches = [torch.tensor([[1.0, 5, -1], [2, 0, -2]]), torch.tensor([[3.0, 1, -5]])]
    for b0, x in zip((0, 2), batches):
        doc = torch.arange(b0, b0 + x.shape[0])[:, None].expand_as(x)
        v, d, p = probes.merge_top(v, d, p, x, doc, torch.full_like(doc, 7), k)
    assert v[:, 0].tolist() == [3, 2] and d[:, 0].tolist() == [2, 1]
    assert v[:, 1].tolist() == [5, 1] and d[:, 1].tolist() == [0, 2]
    lo, ld, _ = probes.merge_top(torch.zeros(0, N), torch.zeros(0, N, dtype=torch.long), torch.zeros(0, N, dtype=torch.long),
                                 batches[0], torch.tensor([[0] * N, [1] * N]), torch.zeros(2, N, dtype=torch.long), k, largest=False)
    assert lo[0, 2] == -2 and ld[0, 2] == 1


def test_doc_extremes_respect_the_mask():
    acts = torch.tensor([[[9.0], [1.0], [-4.0], [100.0]]])          # [B=1, T=4, N=1]
    mask = torch.tensor([[False, True, True, False]])               # position 0 (sink) and a pad excluded
    hv, hi, lv, li = probes.doc_extremes(acts, mask)
    assert hv.item() == 1 and hi.item() == 1 and lv.item() == -4 and li.item() == 2


def test_row_edits_ablate_and_push_per_row():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 3, bias=False)
    x = torch.ones(2, 5, 4)
    plain = lin(x)
    push = torch.tensor([[0.0, 0, 0], [1.0, 2, 3]])
    with probes.RowEdits({0: lin}, pushes={0: push}, zero={0: [(0, 1)]}):
        out = lin(x)
    x0 = x[0].clone()
    x0[:, 1] = 0
    assert torch.allclose(out[0], lin(x0))                          # row 0: neuron 1 silenced, no push
    assert torch.allclose(out[1], plain[1] + push[1])                 # row 1: pushed, not silenced
    assert torch.allclose(lin(x), plain)                              # hooks removed


def test_corpus_reader(tmp_path):
    f = tmp_path / "c4.json.gz"
    with gzip.open(f, "wt") as g:
        for t in ["short", "x" * 400, "y" * 5000, "z" * 350]:
            g.write(json.dumps({"text": t, "url": "u"}) + "\n")
    docs = corpus.read_docs(str(f), 2)
    assert docs == ["x" * 400, "y" * 2000]
