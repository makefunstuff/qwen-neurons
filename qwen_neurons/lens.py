"""Weights-only views of one MLP neuron (plain torch, runs on CPU, unit-tested without the model).

- Logit lens: the tokens W_U · (g_final ⊙ W_down[:, n]) promotes and suppresses, as z-scores over the vocabulary.
  The search index keeps the top and bottom K tokens of every neuron, so neurons can be found by word.
- Web: virtual weights between layers. A push of dose d on neuron m (layer A) adds d · med · ĉ_m to the residual
  (ĉ = the unit column of W_down). A later neuron n reads RMSNorm(resid) ⊙ g, so its gate and up inputs move by
  about d · √hidden · (w ⊙ g) · ĉ_m when the push is as large as the residual. That number, per unit dose, is the
  link weight: positive means pushing m up raises n's inputs ("toward"), negative "away".
- Heads: how strongly each attention head can write into the neuron's input direction (its output projection),
  and how strongly each later head reads the neuron's output (its value projection).
"""
import math

import torch

K_INDEX = 20   # tokens per side kept per neuron in the search index


def norm_gain(w: torch.Tensor) -> torch.Tensor:
    """Qwen3.5's RMSNorm scales by (1 + weight)."""
    return 1.0 + w.float()


def lens_logits(W_U: torch.Tensor, g_final: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """W_U [vocab, hidden], g_final [hidden], cols [hidden, n] -> z-scored logits [vocab, n] (float32)."""
    lo = (W_U @ (cols * g_final.to(cols.dtype)[:, None]).to(W_U.dtype)).float()
    return (lo - lo.mean(0, keepdim=True)) / lo.std(0, keepdim=True).clamp_min(1e-8)


def lens_topk(W_U: torch.Tensor, g_final: torch.Tensor, cols: torch.Tensor, k: int = K_INDEX):
    """-> (top_ids, top_z, bot_ids, bot_z), each [n, k]; bot_z is negative (most suppressed first)."""
    z = lens_logits(W_U, g_final, cols)
    tv, ti = z.topk(k, dim=0)
    bv, bi = z.topk(k, dim=0, largest=False)
    return ti.T.int(), tv.T.half(), bi.T.int(), bv.T.half()


def build_layer_index(W_U: torch.Tensor, g_final: torch.Tensor, W_down: torch.Tensor, k: int = K_INDEX,
                      chunk: int = 512, progress=None) -> dict[str, torch.Tensor]:
    """The search index for one layer: W_down [hidden, neurons] -> {top_ids, top_z, bot_ids, bot_z} [neurons, k]."""
    parts = []
    N = W_down.shape[1]
    for c in range(0, N, chunk):
        parts.append(lens_topk(W_U, g_final, W_down[:, c:c + chunk], k))
        if progress:
            progress(min(N, c + chunk) / N)
    return {name: torch.cat([p[i] for p in parts]) for i, name in enumerate(("top_ids", "top_z", "bot_ids", "bot_z"))}


def matching_tokens(vocab: list[str], query: str) -> list[int]:
    """Token ids whose text contains the query (case-insensitive, surrounding spaces ignored)."""
    q = query.strip().lower()
    if not q:
        return []
    return [i for i, t in enumerate(vocab) if q in t.lower()]


def search_index(index: dict[str, torch.Tensor], token_ids: list[int], limit: int = 30,
                 layers: tuple[int, int] | None = None) -> list[dict]:
    """Neurons whose lens promotes (side +) or suppresses (side −) any of token_ids.

    index tensors are [layers, neurons, k]. Score = the summed |z| of the matching tokens in the neuron's list, so a
    neuron that promotes " cookie", "cookies" and " Cookie" beats one that has only one of them. A side − hit means a
    negative dose promotes the word. layers = (first, last) limits the search to those layers."""
    if not token_ids:
        return []
    want = torch.tensor(sorted(set(token_ids)), dtype=index["top_ids"].dtype)
    out = []
    for side, ids, z in (("+", index["top_ids"], index["top_z"]), ("-", index["bot_ids"], index["bot_z"])):
        hit = torch.isin(ids, want)
        score = (z.float().abs() * hit).sum(-1)               # [layers, neurons]
        if layers is not None:
            keep = torch.zeros(score.shape[0], 1)
            keep[max(0, layers[0]):layers[1] + 1] = 1
            score = score * keep
        flat = score.flatten()
        k = min(limit, int((flat > 0).sum()))
        if not k:
            continue
        vals, idx = flat.topk(k)
        N = score.shape[1]
        for v, i in zip(vals.tolist(), idx.tolist()):
            L, n = divmod(i, N)
            row = hit[L, n].nonzero().flatten().tolist()
            out.append({"layer": L, "neuron": n, "side": side, "score": round(v, 2), "hits": len(row),
                        "best_rank": row[0] + 1, "token_ids": [int(ids[L, n, j]) for j in row[:6]]})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def in_direction(W_gate: torch.Tensor, W_up: torch.Tensor, g: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The directions neuron n reads from the residual: (gate row ⊙ g, up row ⊙ g), float32 [hidden]."""
    g = g.float()
    return W_gate[n].float() * g, W_up[n].float() * g


def unit_cols(W_down: torch.Tensor) -> torch.Tensor:
    W = W_down.float()
    return W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)


def upstream(W_down_src: torch.Tensor, gate_dir: torch.Tensor, up_dir: torch.Tensor) -> dict[str, torch.Tensor]:
    """How each neuron of an earlier layer moves the target's inputs, per unit dose: {gate, up, w} [neurons]."""
    C = unit_cols(W_down_src)                                  # [hidden, N]
    s = math.sqrt(C.shape[0])
    dg, du = s * (gate_dir @ C), s * (up_dir @ C)
    return {"gate": dg, "up": du, "w": (dg + du) / 2}


def downstream(W_gate_dst: torch.Tensor, W_up_dst: torch.Tensor, g_dst: torch.Tensor, col: torch.Tensor) -> dict[str, torch.Tensor]:
    """How a unit-dose push along col moves every neuron of a later layer: {gate, up, w} [neurons]."""
    c = col.float() / col.float().norm().clamp_min(1e-8)
    s = math.sqrt(c.shape[0])
    v = c * g_dst.float()
    dg, du = s * (W_gate_dst.float() @ v), s * (W_up_dst.float() @ v)
    return {"gate": dg, "up": du, "w": (dg + du) / 2}


def top_links(t: dict[str, torch.Tensor], k: int, skip: int | None = None) -> list[dict]:
    w = t["w"].clone()
    if skip is not None:
        w[skip] = 0
    idx = w.abs().topk(min(k, w.numel())).indices.tolist()
    return [{"neuron": i, "w": round(float(t["w"][i]), 4), "gate": round(float(t["gate"][i]), 4),
             "up": round(float(t["up"][i]), 4)} for i in idx]


def head_writes(W_O: torch.Tensor, n_heads: int, read_dir: torch.Tensor) -> list[float]:
    """W_O [hidden, heads*dh]: for each head, the largest gain from its output space onto the unit read direction."""
    r = read_dir.float() / read_dir.float().norm().clamp_min(1e-8)
    proj = (r @ W_O.float()).view(n_heads, -1)                # [heads, dh]
    return proj.norm(dim=-1).tolist()


def head_reads(W_V: torch.Tensor, n_heads: int, g_in: torch.Tensor, col: torch.Tensor) -> list[float]:
    """W_V [heads*dh, hidden]: how strongly each head's values pick up the (normed) unit output direction col."""
    c = col.float() / col.float().norm().clamp_min(1e-8)
    v = W_V.float() @ (c * g_in.float())
    return v.view(n_heads, -1).norm(dim=-1).tolist()
