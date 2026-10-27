"""The debugger's bookkeeping, kept free of the model so it is unit-tested on the CPU: the wiring (steering knobs,
clamped neurons, pins, breakpoints), the registers report and token sampling."""
from dataclasses import dataclass, field

import torch

from .steering import Knob, layer_pushes


@dataclass
class DebugState:
    knobs: list[Knob] = field(default_factory=list)
    clamps: list[dict] = field(default_factory=list)      # [{layer, neuron, value}]
    pins: list[tuple[int, int]] = field(default_factory=list)
    bp_tokens: set[int] = field(default_factory=set)
    bp_text: str = ""
    meds: dict[int, float] = field(default_factory=dict)
    pushes: dict = field(default_factory=dict)

    @classmethod
    def from_job(cls, job: dict) -> "DebugState":
        st = cls()
        st.update(job)
        bp = job.get("breakpoints") or {}
        st.bp_tokens = {int(i) for i in bp.get("tokens", [])}
        st.bp_text = str(bp.get("text") or "")
        return st

    def update(self, d: dict) -> None:
        if "knobs" in d:
            self.knobs = [Knob.from_dict(k) for k in d["knobs"] if float(k.get("dose", 0)) != 0]
        if "clamps" in d:
            self.clamps = [{"layer": int(c["layer"]), "neuron": int(c["neuron"]), "value": float(c["value"])}
                           for c in d["clamps"]]
        if "pins" in d:
            self.pins = [(int(p["layer"]), int(p["neuron"])) for p in d["pins"]]
        if "breakpoints" in d:
            bp = d["breakpoints"] or {}
            self.bp_tokens = {int(i) for i in bp.get("tokens", [])}
            self.bp_text = str(bp.get("text") or "")

    @property
    def clamps_by_layer(self) -> dict[int, list[tuple[int, float]]]:
        out: dict[int, list[tuple[int, float]]] = {}
        for c in self.clamps:
            out.setdefault(c["layer"], []).append((c["neuron"], c["value"]))
        return out

    def build_pushes(self, column) -> dict[int, torch.Tensor]:
        """column(L, n) -> W_down[:, n] of layer L. The steering pushes, with the prompt's medians."""
        cols = {(k.layer, k.neuron): column(k.layer, k.neuron) for k in self.knobs}
        return layer_pushes(self.knobs, cols, self.meds)


def top_registers(acts: dict[int, torch.Tensor], k: int, pins: list[tuple[int, int]]) -> dict:
    """acts[L] = the MLP activations (down_proj input) at the current position -> each layer's k strongest
    neurons by |activation| (neuron, value), the layer's mean |activation|, and the pinned neurons' values."""
    layers = []
    for L in sorted(acts):
        a = acts[L].float()
        vals, idx = a.abs().topk(min(k, a.numel()))
        layers.append({"layer": L, "top": [[int(n), round(float(a[n]), 3)] for n in idx.tolist()],
                       "mean_abs": round(float(a.abs().mean()), 4)})
    pinned = [[L, n, round(float(acts[L][n]), 3)] for L, n in pins if L in acts and 0 <= n < acts[L].numel()]
    return {"layers": layers, "pins": pinned}


def top_candidates(tok, logits: torch.Tensor, k: int = 5) -> list[list]:
    """The k most likely next tokens: [[text, probability], ...]."""
    p = torch.softmax(logits.float(), -1)
    vals, idx = p.topk(k)
    return [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(vals, idx)]


def sample_token(torch_mod, logits: torch.Tensor, temperature: float, top_k: int = 40, top_p: float = 0.95):
    """Sample like generate(): temperature, top-k, top-p (greedy at 0). Returns (token id, its probability)."""
    probs = torch_mod.softmax(logits.float(), -1)
    if temperature <= 0:
        t = int(probs.argmax())
        return t, round(float(probs[t]), 4)
    vals, idx = (logits.float() / temperature).topk(min(top_k, logits.numel()))
    p = torch_mod.softmax(vals, -1)
    keep = (p.cumsum(-1) - p) < top_p
    p = p * keep
    p = p / p.sum()
    t = int(idx[torch_mod.multinomial(p, 1)])
    return t, round(float(probs[t]), 4)
