"""The forward-pass views of a neuron (the GPU worker runs them; the math here is plain torch and unit-tested).

- Steer lens: push the neuron ± on a fixed set of reference prompts and compare the next-token distributions with
  the plain model: which tokens rise and fall, and the KL.
- Zero-ablate: the same comparison with the neuron's activation set to 0.
- Fires: the corpus snippets where the neuron's activation (the input to down_proj) is largest, both signs.
"""
import torch

# The steer lens reads the next token at the end of each of these: chat turns (the reply's first token) and
# plain text cut mid-sentence (where the model is free to go anywhere).
REF_CHAT = [
    "Tell me about yourself.",
    "What should I do this weekend?",
    "Why is the sky blue?",
    "Give me a tip for cooking rice.",
    "What do you think about the news today?",
    "Write one sentence about your favorite animal.",
]
REF_TEXT = [
    "The most important thing to know about the city is that",
    "When I opened the door, I saw",
    "Scientists announced on Tuesday that",
    "My grandmother always told me that",
    "The real reason the meeting was cancelled is",
    "Here is my honest opinion:",
]

FLOOR = 1e-4   # a token has to reach this probability (on average) to count as rising or falling


def log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def kl(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) from log-probs [..., vocab] -> [...]."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def token_shift(lp0: torch.Tensor, lp1: torch.Tensor, k: int = 12, floor: float = FLOOR) -> dict:
    """lp0, lp1 [prompts, vocab]: the plain and the edited next-token log-probs.

    Rising: the biggest mean log-prob gain among tokens whose mean edited prob reaches floor; falling: the biggest
    loss among tokens whose mean plain prob reaches floor. Returns ids with mean log-prob change and both probs."""
    p0, p1 = lp0.exp().mean(0), lp1.exp().mean(0)
    d = (lp1 - lp0).mean(0)
    out = {}
    for name, mask, sign in (("rising", p1 >= floor, 1.0), ("falling", p0 >= floor, -1.0)):
        s = torch.where(mask, d * sign, torch.full_like(d, -float("inf")))
        n = min(k, int(mask.sum()))
        idx = s.topk(n).indices if n else torch.zeros(0, dtype=torch.long)
        out[name] = [{"id": int(i), "dlogp": round(float(d[i]), 3), "p0": float(p0[i]), "p1": float(p1[i])}
                     for i in idx if float(d[i]) * sign > 0]
    out["kl"] = float(kl(lp0, lp1).mean())
    return out


def merge_top(run_v, run_doc, run_pos, v, doc, pos, k: int, largest: bool = True):
    """Keep the k best of a running top list and a new batch, per neuron.

    run_* [k0, N] (k0 may be 0), new [B, N] -> [min(k, k0+B), N]."""
    V, D, P = torch.cat([run_v, v]), torch.cat([run_doc, doc]), torch.cat([run_pos, pos])
    kk = min(k, V.shape[0])
    top = V.topk(kk, dim=0, largest=largest)
    return top.values, D.gather(0, top.indices), P.gather(0, top.indices)


def doc_extremes(acts: torch.Tensor, mask: torch.Tensor):
    """acts [B, T, N], mask [B, T] (True = a real token, position 0 excluded by the caller) ->
    (max [B, N], argmax [B, N], min [B, N], argmin [B, N])."""
    m = mask[..., None]
    a = acts.float()
    hi = a.masked_fill(~m, -float("inf")).max(dim=1)
    lo = a.masked_fill(~m, float("inf")).min(dim=1)
    return hi.values, hi.indices, lo.values, lo.indices


class RowEdits:
    """Forward hooks on down_proj for a batch where every row has its own edit: a push added to the output
    (steering, like SteeringHooks) and/or neurons whose activation (the input) is set to 0 (ablation).

    pushes {layer: [batch, hidden]}; zero {layer: [(row, neuron), ...]}."""

    def __init__(self, modules: dict, pushes: dict | None = None, zero: dict | None = None):
        self.modules, self.pushes, self.zero = modules, pushes or {}, zero or {}
        self.handles = []

    def __enter__(self):
        for L, rows in self.zero.items():
            def pre(module, inputs, rows=rows):
                x = inputs[0].clone()
                for r, n in rows:
                    x[r, ..., n] = 0
                return (x,)
            self.handles.append(self.modules[L].register_forward_pre_hook(pre))
        for L, vec in self.pushes.items():
            def post(module, inputs, out, vec=vec):
                v = vec.to(device=out.device, dtype=out.dtype)
                return out + v.view(v.shape[0], *([1] * (out.dim() - 2)), v.shape[1])
            self.handles.append(self.modules[L].register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
