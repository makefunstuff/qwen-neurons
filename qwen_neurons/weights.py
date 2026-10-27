"""The weights-only half of the neuron explorer: a CPU process that reads the safetensors files directly.

The logit lens, the word search, the web (virtual weights) and the map need no forward pass, so they never wake the
GPU: a separate process (started on demand, exits after the idle timeout, like the model process) opens the
weights with safetensors and does the math with torch on the CPU. The search index (the lens of all 32 × 9216
neurons) is built once, in about ten minutes, and cached in <data_dir>/explorer/.

The server talks to it through two queues; nothing here imports torch at module level.
"""
import asyncio
import base64
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import queue
import re
import threading
import time
from pathlib import Path

LAYERS, NEURONS = 32, 9216


def cache_dir(data_dir: Path, model: str) -> Path:
    """Where the explorer's caches for this model live (the lens index, the fires index, the column norms)."""
    name = re.sub(r"[^A-Za-z0-9._-]", "-", Path(model.rstrip("/")).name or "model")[:40]
    return Path(data_dir) / "explorer" / f"{name}-{hashlib.sha1(model.encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------------------------------- the process
def main(cfg: dict, req, resp) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU only: this process never touches a GPU
    try:
        os.nice(5)
    except OSError:
        pass
    box = Box(cfg)
    lock = threading.Lock()                               # one heavy op at a time; quick ones share the CPU

    def run(msg):
        rid, op, kw = msg["id"], msg["op"], msg.get("kw") or {}

        def progress(frac, text):
            resp.put({"id": rid, "t": "progress", "frac": round(frac, 4), "msg": text})
        try:
            fn = getattr(box, "op_" + op, None)
            if fn is None:
                raise ValueError(f"unknown op {op}")
            if op in Box.HEAVY:
                with lock:
                    out = fn(progress=progress, **kw)
            else:
                out = fn(**kw)
            resp.put({"id": rid, "t": "result", "data": out})
        except Exception as e:  # noqa: BLE001 - everything goes back to the caller
            resp.put({"id": rid, "t": "error", "msg": f"{type(e).__name__}: {e}"})
    while True:
        msg = req.get()
        if msg is None:
            return
        threading.Thread(target=run, args=(msg,), daemon=True).start()


class Box:
    """The weights, the tokenizer and the cached indexes, all loaded lazily."""

    HEAVY = {"build_index"}

    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
        self.cfg = cfg
        self.dir = Path(cfg["cache_dir"])
        self.mu = threading.RLock()
        self._path = self._map = self._files = self._tok = self._vocab = None
        self._index = self._fires = self._norms = None
        self._mats: dict[str, object] = {}                # a small LRU of weight matrices (bf16)

    # ---- files ----------------------------------------------------------------------------------------------
    def path(self) -> str:
        with self.mu:
            if self._path is None:
                from .worker import _resolve_model
                self._path = _resolve_model(self.cfg["model"], _NullEvents())
            return self._path

    def weight_map(self) -> dict:
        with self.mu:
            if self._map is None:
                p = Path(self.path())
                idx = p / "model.safetensors.index.json"
                if idx.is_file():
                    wm = json.loads(idx.read_text())["weight_map"]
                else:
                    from safetensors import safe_open
                    f = next(p.glob("*.safetensors"))
                    with safe_open(str(f), "pt") as h:
                        wm = {k: f.name for k in h.keys()}
                down = next(k for k in wm if k.endswith("layers.0.mlp.down_proj.weight") and "visual" not in k
                            and not k.startswith("mtp"))
                self.prefix = down[:-len("layers.0.mlp.down_proj.weight")]
                cfgj = json.loads((p / "config.json").read_text())
                self.text_cfg = cfgj.get("text_config", cfgj)
                self.tied = cfgj.get("tie_word_embeddings", self.text_cfg.get("tie_word_embeddings", False))
                self._map, self._files = wm, {}
            return self._map

    def has(self, name: str) -> bool:
        return (self.prefix_name(name)) in self.weight_map()

    def prefix_name(self, name: str) -> str:
        self.weight_map()
        return name if name.startswith("lm_head") else self.prefix + name

    def get(self, name: str, keep: int = 12):
        """A tensor by its name under the text model (e.g. 'layers.3.mlp.down_proj.weight'), bf16 on the CPU."""
        key = self.prefix_name(name)
        with self.mu:
            if key in self._mats:
                t = self._mats.pop(key)
                self._mats[key] = t
                return t
            fname = self.weight_map()[key]
            if fname not in self._files:
                from safetensors import safe_open
                self._files[fname] = safe_open(str(Path(self.path()) / fname), "pt")
            t = self._files[fname].get_tensor(key)
            self._mats[key] = t
            big = [k for k in self._mats if not k.endswith(("norm.weight", "embed_tokens.weight", "lm_head.weight"))]
            while len(big) > keep:
                self._mats.pop(big.pop(0))
            return t

    def W_U(self):
        self.weight_map()
        return self.get("lm_head.weight" if not self.tied and self.has("lm_head.weight") else "embed_tokens.weight")

    def g_final(self):
        from .lens import norm_gain
        return norm_gain(self.get("norm.weight"))

    def layer_types(self) -> list[str]:
        self.weight_map()
        return self.text_cfg.get("layer_types") or ["full_attention"] * self.text_cfg.get("num_hidden_layers", LAYERS)

    def tok(self):
        with self.mu:
            if self._tok is None:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self.path())
            return self._tok

    def vocab(self) -> list[str]:
        with self.mu:
            if self._vocab is None:
                tok, V = self.tok(), int(self.W_U().shape[0])
                n = min(V, len(tok))
                self._vocab = tok.batch_decode([[i] for i in range(n)]) + [""] * (V - n)
            return self._vocab

    def show(self, ids) -> list[str]:
        v = self.vocab()
        return [v[int(i)] for i in ids]

    # ---- caches ---------------------------------------------------------------------------------------------
    def index(self):
        with self.mu:
            if self._index is None and (self.dir / "lens-index.pt").is_file():
                self._index = self.torch.load(self.dir / "lens-index.pt")
            return self._index

    def fires_data(self):
        with self.mu:
            f = self.dir / "fires.pt"
            if f.is_file() and (self._fires is None or self._fires[0] != f.stat().st_mtime):
                self._fires = (f.stat().st_mtime, self.torch.load(f))
            return self._fires[1] if self._fires else None

    def col_norms(self):
        """||W_down[:, n]|| of every neuron, [layers, neurons] (cached on disk)."""
        with self.mu:
            if self._norms is None:
                f = self.dir / "col-norms.pt"
                if f.is_file():
                    self._norms = self.torch.load(f)
                else:
                    self._norms = self.torch.stack([self.get(f"layers.{L}.mlp.down_proj.weight").float().norm(dim=0)
                                                    for L in range(self.n_layers())])
                    self.dir.mkdir(parents=True, exist_ok=True)
                    self.torch.save(self._norms, f)
            return self._norms

    def n_layers(self) -> int:
        self.weight_map()
        return int(self.text_cfg.get("num_hidden_layers", LAYERS))

    # ---- ops ------------------------------------------------------------------------------------------------
    def op_status(self) -> dict:
        idx = self.dir / "lens-index.pt"
        parts = sorted(self.dir.glob("lens-L*.pt")) if self.dir.is_dir() else []
        fires = self.fires_data()
        return {"index": idx.is_file(), "index_layers": LAYERS if idx.is_file() else len(parts),
                "fires": fires is not None, "fires_docs": int(fires["docs"].shape[0]) if fires else 0,
                "fires_source": fires.get("source") if fires else None}

    def op_lens(self, layer: int, neuron: int, k: int = 16) -> dict:
        from .lens import lens_topk
        col = self.get(f"layers.{layer}.mlp.down_proj.weight")[:, neuron:neuron + 1]
        ti, tv, bi, bv = lens_topk(self.W_U(), self.g_final(), col, k)
        return {"layer": layer, "neuron": neuron, "col_norm": float(col.float().norm()),
                "promotes": [{"tok": t, "z": round(float(z), 2)} for t, z in zip(self.show(ti[0]), tv[0])],
                "suppresses": [{"tok": t, "z": round(float(z), 2)} for t, z in zip(self.show(bi[0]), bv[0])]}

    def op_build_index(self, progress=None) -> dict:
        """The lens of every neuron, top and bottom K tokens, one layer file at a time (resumes after a stop)."""
        from .lens import K_INDEX, build_layer_index
        torch = self.torch
        if (self.dir / "lens-index.pt").is_file():
            return self.op_status()
        self.dir.mkdir(parents=True, exist_ok=True)
        W_U, g = self.W_U(), self.g_final()
        nL, t0 = self.n_layers(), time.time()
        for L in range(nL):
            f = self.dir / f"lens-L{L:02d}.pt"
            if f.is_file():
                continue
            W = self.get(f"layers.{L}.mlp.down_proj.weight", keep=2)

            def prog(x, L=L):
                done = (L + x) / nL
                left = (time.time() - t0) / max(done, 1e-3) * (1 - done) if done > 0.02 else None
                progress(done, f"lens of layer {L}: {x * 100:.0f}%" + (f" · about {left / 60:.0f} min left" if left else ""))
            part = build_layer_index(W_U, g, W, K_INDEX, progress=prog if progress else None)
            tmp = f.with_suffix(".tmp")
            torch.save(part, tmp)
            tmp.replace(f)
        parts = [torch.load(self.dir / f"lens-L{L:02d}.pt") for L in range(nL)]
        index = {k: torch.stack([p[k] for p in parts]) for k in parts[0]}
        tmp = self.dir / "lens-index.tmp"
        torch.save(index, tmp)
        tmp.replace(self.dir / "lens-index.pt")
        for L in range(nL):
            (self.dir / f"lens-L{L:02d}.pt").unlink(missing_ok=True)
        with self.mu:
            self._index = index
        return self.op_status()

    def op_search(self, query: str, limit: int = 30, lo: int = 0, hi: int = LAYERS - 1) -> dict:
        from .lens import matching_tokens, search_index
        index = self.index()
        if index is None:
            raise RuntimeError("the lens index isn't built yet (🧭 Explorer → build the index)")
        ids = matching_tokens(self.vocab(), query)
        res = search_index(index, ids, limit, (lo, hi))
        for r in res:
            r["tokens"] = self.show(r.pop("token_ids"))
        return {"query": query, "matched_tokens": len(ids), "examples": self.show(ids[:12]), "layers": [lo, hi], "results": res}

    def op_card(self, layer: int, neuron: int) -> dict:
        """A small summary for the map's hover card: a few lens tokens, the column norm and the act max."""
        index = self.index()
        out = {"layer": layer, "neuron": neuron, "col_norm": float(self.col_norms()[layer, neuron])}
        if index is not None:
            out["promotes"] = self.show(index["top_ids"][layer, neuron, :6])
            out["suppresses"] = self.show(index["bot_ids"][layer, neuron, :4])
        else:
            lens = self.op_lens(layer, neuron, 6)
            out["promotes"] = [t["tok"] for t in lens["promotes"]]
            out["suppresses"] = [t["tok"] for t in lens["suppresses"][:4]]
        fires = self.fires_data()
        if fires is not None:
            out["act_max"] = float(fires["top_v"][layer, neuron, 0])
            out["act_min"] = float(fires["bot_v"][layer, neuron, 0])
        return out

    def op_map(self, color: str = "norm") -> dict:
        """One value per neuron for the map's dot colors, quantized to bytes: norm (||W_down[:, n]||) or act (act max)."""
        torch = self.torch
        if color == "act":
            fires = self.fires_data()
            if fires is None:
                raise RuntimeError("no fires index yet")
            v = torch.maximum(fires["top_v"][:, :, 0].float(), -fires["bot_v"][:, :, 0].float())
        else:
            v = self.col_norms()
        v = v.float()
        lo, hi = (float(x) for x in v.flatten().quantile(torch.tensor([0.01, 0.999])))
        q = ((v - lo) / max(hi - lo, 1e-8)).clamp(0, 1).mul(255).round().to(torch.uint8)
        return {"color": color, "layers": int(v.shape[0]), "neurons": int(v.shape[1]), "lo": lo, "hi": hi,
                "values": base64.b64encode(q.numpy().tobytes()).decode()}

    def op_web(self, layer: int, neuron: int, per_layer: int = 4, heads: int = 3) -> dict:
        """Upstream neurons that feed this one and downstream neurons it pushes, ranked per layer, plus heads."""
        from .lens import (downstream, head_reads, head_writes, in_direction, norm_gain, top_links, upstream)
        self.weight_map()
        tc = self.text_cfg
        nL = self.n_layers()
        gate, up = in_direction(self.get(f"layers.{layer}.mlp.gate_proj.weight"),
                                self.get(f"layers.{layer}.mlp.up_proj.weight"),
                                norm_gain(self.get(f"layers.{layer}.post_attention_layernorm.weight")), neuron)
        col = self.get(f"layers.{layer}.mlp.down_proj.weight")[:, neuron].float()
        ups, downs = [], []
        for L in range(layer):
            t = upstream(self.get(f"layers.{L}.mlp.down_proj.weight"), gate, up)
            ups.append({"layer": L, "top": top_links(t, per_layer)})
        for L in range(layer + 1, nL):
            t = downstream(self.get(f"layers.{L}.mlp.gate_proj.weight"), self.get(f"layers.{L}.mlp.up_proj.weight"),
                           norm_gain(self.get(f"layers.{L}.post_attention_layernorm.weight")), col)
            downs.append({"layer": L, "top": top_links(t, per_layer)})
        writers, readers, types = [], [], self.layer_types()
        read = (gate + up) / 2
        for L in range(nL):
            kind = types[L] if L < len(types) else "full_attention"
            if kind == "full_attention":
                n_heads, dh = tc.get("num_attention_heads", 16), tc.get("head_dim", 256)
                n_kv = tc.get("num_key_value_heads", n_heads)
                o, v = f"layers.{L}.self_attn.o_proj.weight", f"layers.{L}.self_attn.v_proj.weight"
                if L <= layer and self.has(o):
                    W = self.get(o)
                    writers += [{"layer": L, "head": h, "kind": "attn", "s": s}
                                for h, s in enumerate(head_writes(W, W.shape[1] // dh, read))]
                if L > layer and self.has(v):
                    readers += [{"layer": L, "head": h, "kind": "attn kv", "s": s}
                                for h, s in enumerate(head_reads(self.get(v), n_kv, norm_gain(self.get(f"layers.{L}.input_layernorm.weight")), col))]
            else:                                          # Gated DeltaNet (linear attention): out_proj, value part of in_proj_qkv
                nv, dv = tc.get("linear_num_value_heads", 32), tc.get("linear_value_head_dim", 128)
                nk, dk = tc.get("linear_num_key_heads", 16), tc.get("linear_key_head_dim", 128)
                o, qkv = f"layers.{L}.linear_attn.out_proj.weight", f"layers.{L}.linear_attn.in_proj_qkv.weight"
                if L <= layer and self.has(o):
                    writers += [{"layer": L, "head": h, "kind": "linear", "s": s} for h, s in enumerate(head_writes(self.get(o), nv, read))]
                if L > layer and self.has(qkv):
                    Wv = self.get(qkv)[2 * nk * dk:2 * nk * dk + nv * dv]
                    readers += [{"layer": L, "head": h, "kind": "linear", "s": s}
                                for h, s in enumerate(head_reads(Wv, nv, norm_gain(self.get(f"layers.{L}.input_layernorm.weight")), col))]
        for lst in (writers, readers):                    # comparable within a kind: as multiples of that kind's median
            for kind in {h["kind"] for h in lst}:
                vals = sorted(h["s"] for h in lst if h["kind"] == kind)
                med = vals[len(vals) // 2] or 1.0
                for h in lst:
                    if h["kind"] == kind:
                        h["s"] = round(h["s"] / med, 2)
        writers.sort(key=lambda h: -h["s"])
        readers.sort(key=lambda h: -h["s"])
        return {"layer": layer, "neuron": neuron, "upstream": ups, "downstream": downs,
                "head_writers": writers[:heads * 3], "head_readers": readers[:heads * 3]}

    def op_fires(self, layer: int, neuron: int, before: int = 20, after: int = 6) -> dict:
        """The top activating corpus snippets for a neuron (both signs), from the cached fires index."""
        f = self.fires_data()
        if f is None:
            raise RuntimeError("no fires index yet (🧭 Explorer → build fires; it runs the corpus through Qwen once)")
        tok = self.tok()

        def snip(d, p, v):
            ids = f["docs"][d, :int(f["lens"][d])].tolist()
            return {"doc": int(d), "act": round(float(v), 3), "before": tok.decode(ids[max(0, p - before):p]),
                    "tok": tok.decode(ids[p:p + 1]), "after": tok.decode(ids[p + 1:p + 1 + after])}
        top = [snip(d, int(p), v) for d, p, v in zip(f["top_doc"][layer, neuron].tolist(), f["top_pos"][layer, neuron].tolist(),
                                                     f["top_v"][layer, neuron].float().tolist())]
        bot = [snip(d, int(p), v) for d, p, v in zip(f["bot_doc"][layer, neuron].tolist(), f["bot_pos"][layer, neuron].tolist(),
                                                     f["bot_v"][layer, neuron].float().tolist())]
        return {"layer": layer, "neuron": neuron, "act_max": top[0]["act"] if top else 0.0,
                "act_min": bot[0]["act"] if bot else 0.0, "mean_abs": float(f["mean_abs"][layer, neuron]),
                "top": top, "bottom": bot, "source": f.get("source", "")}


class _NullEvents:
    def put(self, ev):
        pass


# ---------------------------------------------------------------------------------------------------- the manager
class WeightLab:
    """Server side: starts the CPU process on the first request and stops it after the idle timeout."""

    def __init__(self, model: str, data_dir: Path, idle_timeout_s: int = 600):
        self.model, self.dir, self.idle_timeout_s = model, cache_dir(data_dir, model), idle_timeout_s
        self.ctx = mp.get_context("spawn")
        self.proc = self.req = self.resp = None
        self.pending: dict[int, asyncio.Queue] = {}
        self.ids = itertools.count(1)
        self.last_used = time.time()
        self.reader = None
        self.building = None             # the latest index build progress, for the status

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    async def _ensure(self) -> None:
        if self.alive():
            return
        self.req, self.resp = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=main, args=({"model": self.model, "cache_dir": str(self.dir)}, self.req, self.resp),
                                     daemon=True, name="qwen-neurons-weights")
        self.proc.start()
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        resp, proc = self.resp, self.proc
        while proc is self.proc:
            try:
                ev = await asyncio.to_thread(resp.get, True, 0.5)
            except queue.Empty:
                if not proc.is_alive():
                    for q in self.pending.values():
                        await q.put({"t": "error", "msg": "the weights process exited"})
                    self.pending.clear()
                    return
                continue
            q = self.pending.get(ev["id"])
            if q is not None:
                await q.put(ev)

    async def stream(self, op: str, **kw):
        """Events of one op: progress..., then result (or error)."""
        await self._ensure()
        rid = next(self.ids)
        q: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = q
        self.last_used = time.time()
        self.req.put({"id": rid, "op": op, "kw": kw})
        try:
            while True:
                ev = await q.get()
                self.last_used = time.time()
                yield ev
                if ev["t"] in ("result", "error"):
                    return
        finally:
            self.pending.pop(rid, None)

    async def call(self, op: str, **kw):
        async for ev in self.stream(op, **kw):
            if ev["t"] == "result":
                return ev["data"]
            if ev["t"] == "error":
                raise RuntimeError(ev["msg"])

    def status(self) -> dict:
        return {"alive": self.alive(), "building": self.building, "cache": str(self.dir)}

    async def stop(self) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            try:
                self.req.put(None)
            except (OSError, ValueError):
                pass
            await asyncio.to_thread(p.join, 3)
            if p.is_alive():
                p.kill()

    async def idle_watch(self) -> None:
        while True:
            await asyncio.sleep(10)
            if self.alive() and not self.pending and time.time() - self.last_used > self.idle_timeout_s:
                await self.stop()
