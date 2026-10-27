"""Find personality neurons: contrast the MLP activations (the input to down_proj) of two prompt sets.

Each text gives one sample per neuron: its mean activation over tokens 1:. Neurons are ranked per layer by the
effect size between the sets (Cohen's d), the raw mean difference, or the difference times ||W_down[:, n]||
(how far the neuron moves the residual stream). A positive score means the neuron fires more on set A, so a
positive dose pushes towards A.
"""
import torch

METRICS = ("d", "diff", "resid")


def text_means(acts: torch.Tensor) -> torch.Tensor:
    """acts [tokens, ...] -> mean over tokens 1: (all tokens if there is only one)."""
    a = acts[1:] if acts.shape[0] > 1 else acts
    return a.float().mean(dim=0)


def contrast(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """a [nA, L, N], b [nB, L, N] per-text means -> mean_a, mean_b, diff, d (each [L, N])."""
    ma, mb = a.mean(0), b.mean(0)
    va = a.var(0, unbiased=True) if a.shape[0] > 1 else torch.zeros_like(ma)
    vb = b.var(0, unbiased=True) if b.shape[0] > 1 else torch.zeros_like(mb)
    diff = ma - mb
    pooled = ((va + vb) / 2).sqrt()
    # a floor on the spread stops neurons that are ~constant in both sets from getting huge d out of noise
    floor = pooled.mean(dim=-1, keepdim=True) * 0.1 + eps
    d = diff / torch.maximum(pooled, floor)
    return {"mean_a": ma, "mean_b": mb, "diff": diff, "d": d}


def rank_neurons(a: torch.Tensor, b: torch.Tensor, layers: list[int], metric: str = "d", top_k: int = 30,
                 per_layer_k: int = 5, col_norms: torch.Tensor | None = None, dose: float = 0.5) -> dict:
    """Rank neurons by |score|. layers names the layer index of each row of the L axis.

    Returns {"top": [...], "per_layer": [{"layer", "max_score", "top": [...]}, ...]}; every candidate carries
    layer, neuron, score, d, diff, mean_a, mean_b and a suggested dose with the sign that pushes towards A."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    c = contrast(a, b)
    if metric == "resid":
        if col_norms is None:
            raise ValueError("metric 'resid' needs col_norms")
        score = c["diff"] * col_norms.float()
    else:
        score = c[metric]

    def cand(li: int, n: int) -> dict:
        s = float(score[li, n])
        return {"layer": layers[li], "neuron": n, "score": s, "d": float(c["d"][li, n]),
                "diff": float(c["diff"][li, n]), "mean_a": float(c["mean_a"][li, n]),
                "mean_b": float(c["mean_b"][li, n]), "dose": dose if s >= 0 else -dose}

    flat = score.abs().flatten()
    k = min(top_k, flat.numel())
    idx = flat.topk(k).indices.tolist()
    N = score.shape[1]
    top = [cand(i // N, i % N) for i in idx]
    per_layer = []
    for li, L in enumerate(layers):
        kk = min(per_layer_k, N)
        ids = score[li].abs().topk(kk).indices.tolist()
        per_layer.append({"layer": L, "max_score": float(score[li].abs().max()), "top": [cand(li, n) for n in ids]})
    return {"metric": metric, "top": top, "per_layer": per_layer}
