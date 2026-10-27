"""The model process. The server starts it on demand and kills it to free VRAM: only killing the process frees the
CUDA context too. It reads jobs from one queue and writes events to another.

Nothing heavy is imported at module level: CUDA_VISIBLE_DEVICES must be set before torch loads.
"""
import os
import time
import traceback

EXPECTED_BYTES = 9.3e9  # Qwen3.5-4B bf16 safetensors


def main(cfg: dict, jobs, events, cancel) -> None:
    if cfg["device"].startswith("cuda:"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi numbering
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["device"].split(":")[1]
        cfg = {**cfg, "device": "cuda:0"}
    try:
        w = Worker(cfg, events)
    except Exception as e:  # noqa: BLE001 - everything goes back to the UI
        events.put({"t": "load_error", "msg": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
        return
    events.put({"t": "ready", "device": cfg["device"], "load_s": w.load_s})
    while True:
        job = jobs.get()
        if job is None:
            return
        cancel.clear()
        try:
            w.run(job, cancel)
        except Exception as e:  # noqa: BLE001
            events.put({"t": "error", "job": job.get("id"), "msg": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-2000:]})
        events.put({"t": "done", "job": job.get("id")})


def _resolve_model(path: str, events) -> str:
    """A local directory, or a Hugging Face repo id downloaded to the HF cache (with progress events)."""
    if os.path.isdir(path):
        return path
    import threading
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    blobs = os.path.join(HF_HUB_CACHE, "models--" + path.replace("/", "--"), "blobs")
    stop = threading.Event()

    def watch():
        while not stop.wait(1.0):
            try:
                size = sum(e.stat().st_size for e in os.scandir(blobs))
            except OSError:
                size = 0
            events.put({"t": "progress", "stage": "download", "frac": min(0.99, size / EXPECTED_BYTES),
                        "msg": f"Downloading {path} ({size / 1e9:.1f} / ~{EXPECTED_BYTES / 1e9:.1f} GB, first run only)"})
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        # official repo only: weights, config and tokenizer files, never code
        return snapshot_download(path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE"])
    finally:
        stop.set()


class Worker:
    def __init__(self, cfg: dict, events):
        self.events = events
        events.put({"t": "progress", "stage": "import", "frac": 0.02, "msg": "Waking up the neurons (importing torch)"})
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        t0 = time.time()
        path = _resolve_model(cfg["model"], events)
        events.put({"t": "progress", "stage": "load", "frac": 0.3, "msg": "Shoveling 9 GB of weights onto the GPU"})
        self.tok = AutoTokenizer.from_pretrained(path)
        dtype = torch.bfloat16 if cfg["device"] != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map=cfg["device"]).eval()
        self.layers = next(m for name, m in self.model.named_modules()
                           if name.endswith("language_model.layers") or name.endswith("model.layers"))
        self.down = {i: layer.mlp.down_proj for i, layer in enumerate(self.layers)}
        self.device = cfg["device"]
        events.put({"t": "progress", "stage": "warmup", "frac": 0.95, "msg": "Stretching (warm-up pass)"})
        with torch.no_grad():
            ids = self.tok("Hello", return_tensors="pt").to(self.device)
            self.model(**ids)
        self.eos = {self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        self.load_s = round(time.time() - t0, 1)

    def run(self, job: dict, cancel) -> None:
        kinds = {"generate": self.generate, "discover": self.discover, "evaluate": self.evaluate, "screen": self.screen}
        if job["kind"] not in kinds:
            raise ValueError(f"unknown job {job['kind']}")
        kinds[job["kind"]](job, cancel)

    def emit(self, job: dict, ev: dict) -> None:
        self.events.put({**ev, "job": job["id"]})

    def prompt_ids(self, messages: list[dict], think: bool):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, enable_thinking=think).to(self.device)

    def steer_setup(self, ids, rows):
        """One unsteered pass over the prompt for the residual scale per layer -> (per-row knob stats, pushes)."""
        from .steering import knob_stats, median_resid_norm, row_pushes
        torch = self.torch
        steer_layers = sorted({k.layer for r in rows for k in r})
        if not steer_layers:
            return None, {}
        with torch.no_grad():
            hs = self.model(**ids, output_hidden_states=True).hidden_states  # hs[L+1] leaves layer L
        meds = {L: median_resid_norm(hs[L + 1]) for L in steer_layers}
        del hs
        cols = {(k.layer, k.neuron): self.down[k.layer].weight[:, k.neuron].detach().float().cpu()
                for r in rows for k in r}
        stats = [[knob_stats(k, cols[(k.layer, k.neuron)], meds[k.layer]) for k in r] for r in rows]
        return stats, row_pushes(rows, cols, meds)

    def gen_kwargs(self, job: dict, cancel) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch = self.torch

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

        temp = float(job.get("temperature", 0.7))
        gen = dict(max_new_tokens=int(job.get("max_new_tokens", 600)), stopping_criteria=StoppingCriteriaList([Cancel()]),
                   pad_token_id=self.tok.pad_token_id, do_sample=temp > 0)
        if temp > 0:
            gen.update(temperature=temp, top_p=0.95, top_k=40)
        return gen

    def generate(self, job: dict, cancel) -> None:
        """job: messages, rows (one knob list per row; [] = unsteered), max_new_tokens, temperature, think."""
        from .steering import Knob, SteeringHooks
        torch = self.torch
        rows = [[Knob.from_dict(k) for k in r] for r in job["rows"]]
        think = bool(job.get("think", True))
        ids = self.prompt_ids(job["messages"], think)
        stats, pushes = self.steer_setup(ids, rows)
        if stats:
            self.emit(job, {"t": "stats", "rows": stats})
        B = len(rows)
        batch = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in ids.items()}
        streamer = RowStreamer(self.tok, B, self.eos, think, lambda ev: self.emit(job, ev))
        if job.get("seed") is not None:
            torch.manual_seed(int(job["seed"]))
        t0 = time.time()
        with SteeringHooks(self.down, pushes), torch.no_grad():
            self.model.generate(**batch, streamer=streamer, **self.gen_kwargs(job, cancel))
        streamer.finish("stopped" if cancel.is_set() else "length", time.time() - t0)

    def gen_texts(self, messages: list[dict], rows, job: dict, cancel) -> list[str]:
        """Non-streaming: one reply per row (no thinking), for the evaluation."""
        from .steering import SteeringHooks
        torch = self.torch
        ids = self.prompt_ids(messages, False)
        _, pushes = self.steer_setup(ids, rows)
        B, n0 = len(rows), ids["input_ids"].shape[1]
        batch = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in ids.items()}
        with SteeringHooks(self.down, pushes), torch.no_grad():
            out = self.model.generate(**batch, **self.gen_kwargs(job, cancel))
        texts = []
        for r in range(B):
            toks = out[r, n0:].tolist()
            cut = next((i for i, t in enumerate(toks) if t in self.eos), len(toks))
            texts.append(self.tok.decode(toks[:cut], skip_special_tokens=True).strip())
        return texts

    # ---- discovery ------------------------------------------------------------------------------------------
    def col_norms(self):
        """||W_down[:, n]|| for every layer and neuron: [layers, neurons] on the CPU (computed once)."""
        if getattr(self, "_col_norms", None) is None:
            self._col_norms = self.torch.stack([self.down[L].weight.detach().float().norm(dim=0).cpu()
                                                for L in range(len(self.layers))])
        return self._col_norms

    def discover(self, job: dict, cancel) -> None:
        """job: a, b (lists of replies), questions, layers ([] = all), metric, top_k, per_layer_k, dose.

        Each reply is run as the assistant's answer to a neutral question; the input of every down_proj (the MLP
        neuron activations) is averaged over the reply's tokens, and the two sets are contrasted per layer."""
        from .ranking import rank_neurons, writer_scores
        torch = self.torch
        layers = job.get("layers") or list(range(len(self.layers)))
        qs = job["questions"]
        cur: dict = {}

        def capture(L):
            def pre(module, inputs):
                cur[L] = inputs[0][0, cur["start"]:].float().mean(0)
            return pre
        handles = [self.down[L].register_forward_pre_hook(capture(L)) for L in layers]
        sets, med_sum, n_done = {"a": [], "b": []}, torch.zeros(len(layers)), 0
        resid = {"a": 0, "b": 0}  # summed mean residual leaving each layer, per set
        tok_sum, tok_sq, n_tok = 0, 0, 0  # per-dimension spread of the residual over all reply tokens
        total = len(job["a"]) + len(job["b"])
        t0 = time.time()
        try:
            for name in ("a", "b"):
                for i, text in enumerate(job[name]):
                    if cancel.is_set():
                        self.emit(job, {"t": "stopped"})
                        return
                    prefix = self.tok.apply_chat_template([{"role": "user", "content": qs[i % len(qs)]}],
                                                          add_generation_prompt=True, tokenize=False,
                                                          enable_thinking=False)
                    start = len(self.tok(prefix, add_special_tokens=False).input_ids)
                    ids = self.tok(prefix + text, add_special_tokens=False, return_tensors="pt").to(self.device)
                    cur["start"] = min(start, ids["input_ids"].shape[1] - 1)
                    with torch.no_grad():
                        hs = self.model(**ids, output_hidden_states=True).hidden_states
                    sets[name].append(torch.stack([cur[L] for L in layers]).cpu())
                    reply = [hs[L + 1][0, cur["start"]:].float() for L in layers]
                    med_sum += torch.tensor([float(h.norm(dim=-1).median()) for h in reply])
                    resid[name] = resid[name] + torch.stack([h.mean(0) for h in reply]).cpu()
                    tok_sum = tok_sum + torch.stack([h.sum(0) for h in reply]).cpu()
                    tok_sq = tok_sq + torch.stack([(h * h).sum(0) for h in reply]).cpu()
                    n_tok += reply[0].shape[0]
                    del hs
                    n_done += 1
                    self.emit(job, {"t": "step", "i": n_done, "n": total, "set": name,
                                    "msg": f"reading {'set A' if name == 'a' else 'set B'} reply {i + 1}"})
        finally:
            for h in handles:
                h.remove()
        meds = (med_sum / n_done).tolist()
        norms = self.col_norms()[layers]
        dose = float(job.get("dose", 0.5))
        diff = resid["a"] / len(sets["a"]) - resid["b"] / len(sets["b"])
        with torch.no_grad():
            std = (tok_sq / n_tok - (tok_sum / n_tok) ** 2).clamp_min(0).sqrt()
            writer = writer_scores([self.down[L].weight for L in layers], diff, std)
        res = rank_neurons(torch.stack(sets["a"]), torch.stack(sets["b"]), layers, job.get("metric", "d"),
                           int(job.get("top_k", 30)), int(job.get("per_layer_k", 5)), norms, dose, writer)
        li = {L: i for i, L in enumerate(layers)}
        for c in res["top"] + [c for pl in res["per_layer"] for c in pl["top"]]:
            i = li[c["layer"]]
            c["median_resid"] = meds[i]
            c["col_norm"] = float(norms[i, c["neuron"]])
            c["push"] = abs(c["dose"]) * meds[i]
            c["equiv_activation"] = c["dose"] * meds[i] / c["col_norm"] if c["col_norm"] else 0.0
        res["median_resid"] = dict(zip(layers, meds))
        self.emit(job, {"t": "result", "kind": "discover", "secs": round(time.time() - t0, 1), **res})

    # ---- evaluation -----------------------------------------------------------------------------------------
    def judge_ids(self):
        if getattr(self, "_judge_ids", None) is None:
            def first(words):
                return sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in words})
            self._judge_ids = (first(["YES", "Yes", "yes", " YES", " Yes"]), first(["NO", "No", "no", " NO", " No"]))
        return self._judge_ids

    def judge(self, judge: dict, trait: str, question: str, reply: str) -> bool:
        from .scoring import caps_hit, judge_prompt, keyword_hit
        if judge["kind"] == "keywords":
            return keyword_hit(reply, judge.get("keywords") or [])
        if judge["kind"] == "caps":
            return caps_hit(reply)
        yes, no = self.judge_ids()
        ids = self.prompt_ids([{"role": "user", "content": judge_prompt(trait, question, reply)}], False)
        with self.torch.no_grad():  # the plain model, no hooks: a blind judge
            logits = self.model(**ids).logits[0, -1].float()
        return bool(logits[yes].max() > logits[no].max())

    def evaluate(self, job: dict, cancel) -> None:
        """job: genes, trait, judge {kind, keywords?}, questions, controls, max_new_tokens, temperature, seed.

        Per question one batch: normal, steered and the random controls; each reply is judged blind."""
        import random
        from .scoring import control_genes, verdict
        from .steering import Knob
        genes = job["genes"]
        rng = random.Random(int(job.get("seed") or 0))
        controls = [control_genes(genes, rng) for _ in range(int(job.get("controls", 4)))]
        rows = [[]] + [[Knob.from_dict(g) for g in gs] for gs in [genes] + controls]
        hits, done = [0] * len(rows), 0
        t0 = time.time()
        self.emit(job, {"t": "eval_start", "n": len(job["questions"]), "controls": controls})
        for qi, q in enumerate(job["questions"]):
            if cancel.is_set():
                break
            self.torch.manual_seed(int(job.get("seed") or 0) * 1000 + qi)
            texts = self.gen_texts([{"role": "user", "content": q}], rows, job, cancel)
            if cancel.is_set():
                break
            judged = [self.judge(job["judge"], job["trait"], q, t) for t in texts]
            hits = [h + j for h, j in zip(hits, judged)]
            done += 1
            self.emit(job, {"t": "eval_q", "i": done, "n": len(job["questions"]), "q": q, "normal": texts[0],
                            "steered": texts[1], "hits": judged, "tally": hits})
        if not done:
            self.emit(job, {"t": "stopped"})
            return
        v = verdict(hits[1], hits[0], hits[2:], done)
        self.emit(job, {"t": "result", "kind": "evaluate", **v, "control_genes": controls, "judge": job["judge"]["kind"],
                        "stopped": done < len(job["questions"]), "secs": round(time.time() - t0, 1)})


    def screen(self, job: dict, cancel) -> None:
        """job: candidates (gene lists), trait, judge, questions, max_new_tokens, temperature, seed.

        A quick look before a full evaluation: plain + every candidate on a few questions, in one batch per
        question, each reply judged blind. No controls and no p-values: it only says which ones are worth it."""
        from .steering import Knob
        cands = job["candidates"]
        rows = [[]] + [[Knob.from_dict(g) for g in c] for c in cands]
        hits, samples, done = [0] * len(rows), [""] * len(rows), 0
        for qi, q in enumerate(job["questions"]):
            if cancel.is_set():
                break
            self.torch.manual_seed(int(job.get("seed") or 0) * 1000 + qi)
            texts = self.gen_texts([{"role": "user", "content": q}], rows, job, cancel)
            if cancel.is_set():
                break
            judged = [self.judge(job["judge"], job["trait"], q, t) for t in texts]
            for r, (t, j) in enumerate(zip(texts, judged)):
                if j and not hits[r] or not samples[r]:  # show the first reply that hit, else the first reply
                    samples[r] = t
                hits[r] += j
            done += 1
            self.emit(job, {"t": "screen_q", "i": done, "n": len(job["questions"]), "q": q, "hits": judged,
                            "texts": texts, "tally": hits})
        self.emit(job, {"t": "result", "kind": "screen", "n": done, "plain": hits[0], "hits": hits[1:],
                        "samples": samples[1:], "plain_sample": samples[0]})


class RowStreamer:
    """A generate() streamer for a batch: decodes each row and emits think/text deltas per row."""

    TAG = "</think>"

    def __init__(self, tok, n_rows: int, eos: set[int], think: bool, emit):
        self.tok, self.eos, self.think, self.emit = tok, eos, think, emit
        self.ids = [[] for _ in range(n_rows)]
        self.done = [False] * n_rows
        self.sent = [("", "") for _ in range(n_rows)]
        self.first = True

    def put(self, value):
        if self.first:  # the prompt
            self.first = False
            return
        toks = value.view(-1).tolist()
        for r, t in enumerate(toks):
            if self.done[r]:
                continue
            if t in self.eos:
                self._flush(r, final=True)
                self.done[r] = True
                self.emit({"t": "end", "row": r, "reason": "eos", "tokens": len(self.ids[r])})
                continue
            self.ids[r].append(t)
            self._flush(r, final=False)

    def end(self):
        pass

    def split(self, text: str, final: bool) -> tuple[str, str]:
        if not self.think:
            return "", text
        i = text.find(self.TAG)
        if i >= 0:
            return text[:i].strip("\n"), text[i + len(self.TAG):].lstrip("\n")
        if not final:  # hold back a partial closing tag
            for k in range(len(self.TAG) - 1, 0, -1):
                if text.endswith(self.TAG[:k]):
                    text = text[:-k]
                    break
        return text.strip("\n") if final else text.lstrip("\n"), ""

    def _flush(self, r: int, final: bool) -> None:
        text = self.tok.decode(self.ids[r], skip_special_tokens=True)
        if not final and text.endswith("�"):
            return  # half a multibyte character
        think, answer = self.split(text, final)
        st, sa = self.sent[r]
        dt = think[len(st):] if think.startswith(st) else None
        da = answer[len(sa):] if answer.startswith(sa) else None
        if dt is None or da is None:  # rare rewrite (stripping); send whole strings
            self.emit({"t": "tok", "row": r, "think_full": think, "text_full": answer})
        elif dt or da:
            self.emit({"t": "tok", "row": r, "think": dt, "text": da})
        self.sent[r] = (think, answer)

    def finish(self, reason: str, secs: float) -> None:
        for r in range(len(self.ids)):
            if not self.done[r]:
                self._flush(r, final=True)
                self.done[r] = True
                self.emit({"t": "end", "row": r, "reason": reason, "tokens": len(self.ids[r])})
        self.emit({"t": "finished", "secs": round(secs, 1),
                   "tok_s": round(max(len(i) for i in self.ids) / secs, 1) if secs > 0 else 0})
