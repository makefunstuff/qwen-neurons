"""The steering recipe (after @_lyraaaa_, 2026-10-24).

For a knob (layer L, neuron n, dose d) the push added to the output of layers[L].mlp.down_proj, at every token, is

    d * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]|| * W_down[:, n]

so dose 1 is a push as large as a typical residual-stream vector. Everything here is plain torch and runs on CPU,
so it is unit-tested without a GPU.
"""
from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Knob:
    layer: int
    neuron: int
    dose: float

    @classmethod
    def from_dict(cls, d: dict) -> "Knob":
        return cls(int(d["layer"]), int(d["neuron"]), float(d["dose"]))


def parse_knobs(spec: str) -> list[Knob]:
    """'14:8949:+1,13:7956:-1' -> knobs."""
    knobs = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        L, n, d = part.split(":")
        knobs.append(Knob(int(L), int(n), float(d)))
    return knobs


def active_knobs(raw: list[dict]) -> list[Knob]:
    """Knobs from the API, dropping the ones switched off or at dose 0."""
    return [Knob.from_dict(k) for k in raw if k.get("enabled", True) and float(k.get("dose", 0)) != 0]


def median_resid_norm(hidden: torch.Tensor) -> float:
    """Median L2 norm of the residual stream over tokens 1: (token 0 is an attention sink with an outsized norm).

    hidden: [tokens, hidden] or [1, tokens, hidden]."""
    if hidden.dim() == 3:
        hidden = hidden[0]
    h = hidden[1:] if hidden.shape[0] > 1 else hidden
    return float(h.float().norm(dim=-1).median())


def push_vector(col: torch.Tensor, med: float, dose: float) -> torch.Tensor:
    """dose * med / ||col|| * col, in float32."""
    col = col.float()
    return dose * med / col.norm() * col


def knob_stats(knob: Knob, col: torch.Tensor, med: float) -> dict:
    col_norm = float(col.float().norm())
    return {"layer": knob.layer, "neuron": knob.neuron, "dose": knob.dose, "median_resid": med,
            "col_norm": col_norm, "push": abs(knob.dose) * med,
            # the push measured in units of the neuron's own output: what activation would give the same push
            "equiv_activation": knob.dose * med / col_norm if col_norm else 0.0}


def layer_pushes(knobs: list[Knob], columns: dict[tuple[int, int], torch.Tensor],
                 medians: dict[int, float]) -> dict[int, torch.Tensor]:
    """Sum the push vectors of all knobs on the same layer. columns[(L, n)] = W_down[:, n] of layer L."""
    out: dict[int, torch.Tensor] = {}
    for k in knobs:
        v = push_vector(columns[(k.layer, k.neuron)], medians[k.layer], k.dose)
        out[k.layer] = out[k.layer] + v if k.layer in out else v
    return out


def row_pushes(rows: list[list[Knob]], columns: dict[tuple[int, int], torch.Tensor],
               medians: dict[int, float]) -> dict[int, torch.Tensor]:
    """One knob list per batch row (an empty list = an unsteered row) -> {layer: [batch, hidden] pushes}."""
    per_row = [layer_pushes(knobs, columns, medians) for knobs in rows]
    layers = sorted({L for p in per_row for L in p})
    out = {}
    for L in layers:
        ref = next(p[L] for p in per_row if L in p)
        out[L] = torch.stack([p.get(L, torch.zeros_like(ref)) for p in per_row])
    return out


class SteeringHooks:
    """Context manager: adds each layer's push to its down_proj output, at every token.

    pushes[L] is [hidden] (the same push for every row) or [batch, hidden] (one push per row, so a single batch
    can hold an unsteered and several steered copies of the same prompt)."""

    def __init__(self, modules: dict[int, torch.nn.Module], pushes: dict[int, torch.Tensor]):
        self.modules, self.pushes = modules, pushes
        self.handles = []

    @staticmethod
    def _hook(vec: torch.Tensor):
        def hook(module, inputs, out):
            v = vec.to(device=out.device, dtype=out.dtype)
            if v.dim() == 2:  # [batch, hidden] -> broadcast over tokens
                v = v.view(v.shape[0], *([1] * (out.dim() - 2)), v.shape[1])
            return out + v
        return hook

    def __enter__(self):
        for L, vec in self.pushes.items():
            self.handles.append(self.modules[L].register_forward_hook(self._hook(vec)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def chaos_level(knobs: list[Knob]) -> float:
    """Total push in residual-stream units (sum of |dose|): the chaos meter. Around 1.5 and up the model tends to
    loop (dose 1 alone is a push as large as the whole residual stream)."""
    return sum(abs(k.dose) for k in knobs)


LOOP_ZONE = 1.5   # summed |dose| where replies start to loop
WARM_ZONE = 1.0   # where it gets wobbly


def chaos_zone(total: float) -> str:
    return "loop" if total >= LOOP_ZONE else "warm" if total >= WARM_ZONE else "ok"


def needed_layers(knobs: list[Knob]) -> list[int]:
    return sorted({k.layer for k in knobs})


def group_by_layer(knobs: list[Knob]) -> dict[int, list[Knob]]:
    g = defaultdict(list)
    for k in knobs:
        g[k.layer].append(k)
    return dict(g)
