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
        if job["kind"] == "generate":
            self.generate(job, cancel)
        else:
            raise ValueError(f"unknown job {job['kind']}")

    def prompt_ids(self, messages: list[dict], think: bool):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, enable_thinking=think).to(self.device)

    def generate(self, job: dict, cancel) -> None:
        """job: messages, rows (one knob list per row; [] = unsteered), max_new_tokens, temperature, think."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        from .steering import Knob, SteeringHooks, knob_stats, median_resid_norm, row_pushes
        torch, jid = self.torch, job["id"]
        rows = [[Knob.from_dict(k) for k in r] for r in job["rows"]]
        think = bool(job.get("think", True))
        ids = self.prompt_ids(job["messages"], think)
        steer_layers = sorted({k.layer for r in rows for k in r})
        pushes = {}
        if steer_layers:
            with torch.no_grad():  # one unsteered pass: the residual scale per layer, for this prompt
                hs = self.model(**ids, output_hidden_states=True).hidden_states  # hs[L+1] leaves layer L
            meds = {L: median_resid_norm(hs[L + 1]) for L in steer_layers}
            del hs
            cols = {(k.layer, k.neuron): self.down[k.layer].weight[:, k.neuron].detach().float().cpu()
                    for r in rows for k in r}
            self.events.put({"t": "stats", "job": jid, "rows": [
                [knob_stats(k, cols[(k.layer, k.neuron)], meds[k.layer]) for k in r] for r in rows]})
            pushes = row_pushes(rows, cols, meds)
        B = len(rows)
        batch = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in ids.items()}
        streamer = RowStreamer(self.tok, B, self.eos, think, lambda ev: self.events.put({**ev, "job": jid}))

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

        temp = float(job.get("temperature", 0.7))
        gen = dict(max_new_tokens=int(job.get("max_new_tokens", 600)), streamer=streamer,
                   stopping_criteria=StoppingCriteriaList([Cancel()]), pad_token_id=self.tok.pad_token_id,
                   do_sample=temp > 0)
        if temp > 0:
            gen.update(temperature=temp, top_p=0.95, top_k=40)
        if job.get("seed") is not None:
            torch.manual_seed(int(job["seed"]))
        t0 = time.time()
        with SteeringHooks(self.down, pushes), torch.no_grad():
            self.model.generate(**batch, **gen)
        streamer.finish("stopped" if cancel.is_set() else "length", time.time() - t0)


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
