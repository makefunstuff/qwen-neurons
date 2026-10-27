"""The Lab agent: ask "find me a conspiracy neuron" and it investigates with the explorer's tools, then writes a
verdict the way @_lyraaaa_ does: a name, the input evidence, the direct effect, the downstream effect and a test.

The brain is any OpenAI-compatible chat endpoint with tool calls (DeepSeek by default when a key is set), or the
app's own plain Qwen3.5-4B (no key, no extra VRAM: it is the model the app already loads). Settings come from the
environment or config.local.env:

    QN_AGENT_PROVIDER   deepseek | openai | qwen   (default: deepseek with a key, else qwen)
    QN_AGENT_BASE_URL   default https://api.deepseek.com for deepseek
    QN_AGENT_MODEL      default deepseek-chat
    QN_AGENT_API_KEY    or DEEPSEEK_API_KEY, or a file: QN_AGENT_KEY_FILE (default ~/.config/qwen-neurons/agent.key)

Keys are never written anywhere and never sent to the browser. Tests fake the brain.
"""
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import scoring

DEFAULT_KEY_FILE = "~/.config/qwen-neurons/agent.key"
MAX_STEPS = 20
SUMMARY_CHARS = 1400


# ---------------------------------------------------------------------------------------------------- settings
@dataclass
class AgentConfig:
    provider: str = "qwen"
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    key_source: str = ""

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "AgentConfig":
        key, src = env.get("QN_AGENT_API_KEY", ""), "QN_AGENT_API_KEY"
        provider = (env.get("QN_AGENT_PROVIDER") or "").strip().lower()
        if not key and provider in ("", "deepseek") and env.get("DEEPSEEK_API_KEY"):
            key, src = env["DEEPSEEK_API_KEY"], "DEEPSEEK_API_KEY"
        if not key:
            f = Path(os.path.expanduser(env.get("QN_AGENT_KEY_FILE") or DEFAULT_KEY_FILE))
            try:
                key, src = f.read_text().strip(), str(f)
            except OSError:
                key, src = "", ""
        if not provider:
            provider = "deepseek" if key else "qwen"
        if provider not in ("deepseek", "openai", "qwen"):
            raise ValueError(f"QN_AGENT_PROVIDER must be deepseek, openai or qwen, not {provider!r}")
        base = env.get("QN_AGENT_BASE_URL") or ("https://api.deepseek.com" if provider == "deepseek" else "")
        model = env.get("QN_AGENT_MODEL") or ("deepseek-chat" if provider == "deepseek" else "qwen3.5-4b" if provider == "qwen" else "")
        return cls(provider, base.rstrip("/"), model, key, src)

    def public(self) -> dict:
        """What the browser may see: never the key."""
        ok = self.provider == "qwen" or bool(self.base_url and self.model and (self.api_key or self.provider == "openai"))
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url if self.provider != "qwen" else "",
                "has_key": bool(self.api_key), "key_source": "file" if self.key_source.startswith(("/", "~")) else self.key_source,
                "ready": ok}


# ---------------------------------------------------------------------------------------------------- brains
class OpenAIProvider:
    """Any OpenAI-compatible /chat/completions endpoint with tools (DeepSeek, a local llama.cpp or vLLM server …)."""

    def __init__(self, cfg: AgentConfig, transport=None, timeout: float = 120):
        if not cfg.base_url or not cfg.model:
            raise ValueError("the agent needs QN_AGENT_BASE_URL and QN_AGENT_MODEL")
        if cfg.provider == "deepseek" and not cfg.api_key:
            raise ValueError("DeepSeek needs a key: QN_AGENT_API_KEY, DEEPSEEK_API_KEY or QN_AGENT_KEY_FILE")
        self.cfg, self.transport, self.timeout = cfg, transport, timeout
        self.name = f"{cfg.provider} · {cfg.model}"

    async def chat(self, messages: list[dict], tools: list[dict], force: str | None = None) -> dict:
        import httpx
        url = self.cfg.base_url + ("/chat/completions" if self.cfg.base_url.endswith("/v1") or self.cfg.provider == "deepseek"
                                   else "/v1/chat/completions")
        body = {"model": self.cfg.model, "messages": messages, "tools": tools, "temperature": 0.3}
        if force:
            body["tool_choice"] = {"type": "function", "function": {"name": force}}
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as c:
            r = await c.post(url, json=body, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"the agent's model said {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append({"id": tc.get("id") or uuid.uuid4().hex[:8], "name": fn.get("name", ""),
                          "arguments": _json_args(fn.get("arguments"))})
        return {"content": (msg.get("content") or "").strip(), "tool_calls": calls}


class QwenProvider:
    """The app's own plain Qwen3.5-4B as the brain, through the model process (one job per turn)."""

    def __init__(self, run_job, think: bool = False, max_new_tokens: int = 700):
        self.run_job, self.think, self.max_new_tokens = run_job, think, max_new_tokens
        self.name = "qwen · local Qwen3.5-4B (plain)"

    async def chat(self, messages: list[dict], tools: list[dict], force: str | None = None) -> dict:
        msgs = qwen_messages(messages)
        if force:
            msgs.append({"role": "user", "content": f"Now call the {force} tool with your findings. Only the tool call."})
        res = await self.run_job({"kind": "agent_chat", "messages": msgs, "tools": tools, "think": self.think,
                                  "max_new_tokens": self.max_new_tokens})
        return parse_qwen_reply(res["text"])


def qwen_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-style history -> what Qwen's chat template expects (tool-call arguments as dicts)."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [{"type": "function", "function": {"name": tc["function"]["name"],
                                        "arguments": _json_args(tc["function"]["arguments"])}} for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_qwen_reply(text: str) -> dict:
    """Qwen3.5's tool-call format (<tool_call><function=name><parameter=p>v</parameter>…) -> content + calls.
    Also accepts the older JSON form <tool_call>{"name": …, "arguments": {…}}</tool_call>."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("</think>", "")
    calls = []
    for name, body in _CALL.findall(text):
        args = {}
        for p, v in _PARAM.findall(body):
            v = v.strip()
            try:
                args[p] = json.loads(v)
            except ValueError:
                args[p] = v
        calls.append({"id": uuid.uuid4().hex[:8], "name": name, "arguments": args})
    for raw in _JSON_CALL.findall(text):
        try:
            j = json.loads(raw)
            calls.append({"id": uuid.uuid4().hex[:8], "name": j["name"], "arguments": _json_args(j.get("arguments"))})
        except (ValueError, KeyError):
            pass
    content = _JSON_CALL.sub("", _CALL.sub("", text)).replace("<tool_call>", "").replace("</tool_call>", "").strip()
    return {"content": content, "tool_calls": calls}


def _json_args(a) -> dict:
    if isinstance(a, dict):
        return a
    if not a:
        return {}
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


# ---------------------------------------------------------------------------------------------------- tools
def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


_L = {"type": "integer", "description": "layer, 0-31"}
_N = {"type": "integer", "description": "neuron, 0-9215"}
_GENES = {"type": "array", "description": "neurons with doses, e.g. [{\"layer\": 24, \"neuron\": 3102, \"dose\": 0.8}]",
          "items": {"type": "object", "properties": {"layer": _L, "neuron": _N, "dose": {"type": "number"}},
                    "required": ["layer", "neuron", "dose"]}}

TOOLS = [
    _fn("lens_search", "Find neurons by word: neurons whose logit lens (W_U · W_down[:, n], weights only) promotes "
        "(side +: a positive dose writes it) or suppresses (side −: a negative dose writes it) tokens containing the "
        "word. Try word stems ('conspir') and synonyms, also in other languages. Searches layers 3-26 (where "
        "steering works) unless you pass min_layer/max_layer.",
        {"query": {"type": "string"}, "limit": {"type": "integer"}, "min_layer": {"type": "integer"},
         "max_layer": {"type": "integer"}}, ["query"]),
    _fn("lens", "The tokens one neuron promotes and suppresses (logit lens, weights only).", {"layer": _L, "neuron": _N},
        ["layer", "neuron"]),
    _fn("fires", "Input evidence: the corpus snippets where the neuron's activation is highest (and lowest), with the "
        "act max.", {"layer": _L, "neuron": _N}, ["layer", "neuron"]),
    _fn("fires_on", "Activations of up to 4 neurons on texts you write (contrast a text with the concept against a "
        "neutral one).", {"texts": {"type": "array", "items": {"type": "string"}},
                          "neurons": {"type": "array", "items": {"type": "object", "properties": {"layer": _L, "neuron": _N}}}},
        ["texts", "neurons"]),
    _fn("web", "Weights-only wiring: upstream neurons that feed this one and downstream neurons it pushes toward (+) or "
        "away from (−), per layer, plus the attention heads that write into it or read it.", {"layer": _L, "neuron": _N},
        ["layer", "neuron"]),
    _fn("steer_lens", "Direct effect: push the genes up (and down) on fixed reference prompts; which next tokens rise "
        "and fall, and the KL from the plain model.", {"genes": _GENES}, ["genes"]),
    _fn("zero_ablate", "Set a neuron's activation to 0 on the reference prompts: KL and the tokens that change.",
        {"layer": _L, "neuron": _N}, ["layer", "neuron"]),
    _fn("steer", "Chat with the genes on: plain vs steered replies to a few prompts.",
        {"genes": _GENES, "prompts": {"type": "array", "items": {"type": "string"}}}, ["genes"]),
    _fn("compare", "Several candidates side by side on one prompt (plain first).",
        {"candidates": {"type": "array", "items": _GENES}, "prompt": {"type": "string"}}, ["candidates", "prompt"]),
    _fn("logprobs", "How likely continuations are after a prompt: plain vs steered (genes) and/or zero-ablated.",
        {"prompt": {"type": "string"}, "continuations": {"type": "array", "items": {"type": "string"}}, "genes": _GENES,
         "ablate": {"type": "array", "items": {"type": "object", "properties": {"layer": _L, "neuron": _N}}}},
        ["prompt", "continuations"]),
    _fn("test_prediction", "Write a prediction first, then test it on held-out neutral questions (not about the "
        "concept): plain vs steered, each reply judged blind, by keywords if given, else by Qwen for the trait.",
        {"genes": _GENES, "prediction": {"type": "string"}, "trait": {"type": "string"},
         "keywords": {"type": "array", "items": {"type": "string"}}},
        ["genes", "prediction"]),
    _fn("evaluate", "The Lab's score card: N held-out questions, blind judge, Fisher p and random-neuron controls "
        "(slow: about 3 s per question).", {"genes": _GENES, "trait": {"type": "string"}, "n": {"type": "integer"}},
        ["genes", "trait"]),
    _fn("make_pug", "Turn a winner into a new pug in the user's zoo.", {"name": {"type": "string"}, "genes": _GENES,
        "trait": {"type": "string"}}, ["name", "genes", "trait"]),
    _fn("verdict", "Finish: your verdict, like a neuroscientist's notebook entry.",
        {"name": {"type": "string", "description": "a short name for the neuron or recipe"}, "genes": _GENES,
         "trait": {"type": "string", "description": "one line: what it does to the model"},
         "input_evidence": {"type": "string", "description": "what makes it fire (fires, upstream)"},
         "direct_effect": {"type": "string", "description": "what it writes (lens, steer lens)"},
         "downstream_effect": {"type": "string", "description": "what it pushes later (web, replies)"},
         "test": {"type": "string", "description": "the prediction you tested and the result"},
         "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
        ["name", "genes", "trait", "direct_effect", "test"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM = """You are the neuron hunter in the Evil Pugs Lab. The model under the microscope is Qwen3.5-4B: 32 layers × \
9216 MLP neurons. Steering recipe: a gene (layer L, neuron n, dose d) adds d × (typical residual size) along \
W_down[:, n] at every token, so dose ±1 is a push as large as the residual stream; ±0.5–0.9 is the useful range and \
summed |dose| above ~1.5 makes replies loop. A negative dose writes what the neuron's lens suppresses.

Method (use the tools, one step at a time, and keep it short):
1. lens_search for the concept and for the words that persona would say (for a conspiracy theorist: hidden, \
secretly, agenda, truth, cover-up), with stems, synonyms and other languages. Pick 2–4 candidates (layers 3–26 \
steer best). Mind the sign: a side − hit is written by a NEGATIVE dose.
2. Check each: lens, steer_lens (direct effect, KL; a candidate whose steer lens doesn't move toward the concept is \
not it), fires or fires_on (input evidence), web (downstream).
3. steer or compare the best on everyday prompts that don't mention the concept (the trait must show up uninvited); \
zero_ablate or logprobs for detail.
4. Before the verdict, write a prediction and run test_prediction (held-out neutral questions). The verdict reports \
its result as measured.
5. Call verdict: name, genes, trait, input evidence, direct effect, downstream effect, the test and its result, and \
your confidence. Report what you saw, not what you hoped: a neuron that does nothing gets an honest verdict too.
Only steer with genes you have looked at. Don't explain the tools to the user."""


def gene(g: dict) -> dict:
    L, n, d = int(g["layer"]), int(g["neuron"]), float(g.get("dose", 0.8))
    if not (0 <= L <= 31 and 0 <= n <= 9215):
        raise ValueError(f"L{L} N{n} is out of range (layers 0-31, neurons 0-9215)")
    return {"layer": L, "neuron": n, "dose": round(max(-2.0, min(2.0, d)), 3)}


def genes(raw, limit: int = 4) -> list[dict]:
    if isinstance(raw, dict):
        raw = [raw]
    out = [gene(g) for g in (raw or [])][:limit]
    if not out:
        raise ValueError("give at least one gene: [{layer, neuron, dose}]")
    return out


def neuron(args: dict) -> tuple[int, int]:
    g = gene({**args, "dose": 0})
    return g["layer"], g["neuron"]


def tok(t: str) -> str:
    """A token as the agent sees it: quoted, so a leading space shows."""
    return repr(t)


def toks(ts: list[dict], n: int) -> str:
    return ", ".join(tok(t["tok"]) for t in ts[:n]) or "none"


def label(gs: list[dict]) -> str:
    return " + ".join(f"L{g['layer']} N{g['neuron']} {g['dose']:+g}" for g in gs)


class Toolbox:
    """Runs the agent's tools: weights ops on the CPU process, the rest as model jobs. Each tool yields progress
    events and finally {"t": "done", "data": ..., "summary": "..."} (the summary is what the agent reads)."""

    def __init__(self, weights, run_job_stream, zoo=None):
        self.weights, self.stream, self.zoo = weights, run_job_stream, zoo

    async def job(self, job: dict):
        """Run a model job; yields progress events; the last event is the result. The job's stream is always read to
        its end: leaving it early would keep the engine's job lock held."""
        result = error = None
        async for ev in self.stream(job):
            if ev["t"] == "error":
                error = error or ev["msg"]
            elif ev["t"] == "result":
                result = ev
            elif ev["t"] in ("step", "progress", "eval_q", "screen_q"):
                yield {"t": "tool_progress", "msg": ev.get("msg") or (f"question {ev['i']}/{ev['n']}" if "i" in ev else "")}
        if error:
            raise RuntimeError(error)
        if result is None:
            raise RuntimeError("the model job ended without a result (stopped?)")
        yield result

    async def run(self, name: str, args: dict):
        if name not in TOOL_NAMES or name == "verdict":
            raise ValueError(f"no tool {name!r}")
        async for ev in getattr(self, "t_" + name)(args):
            yield ev

    @staticmethod
    def done(data, summary: str) -> dict:
        return {"t": "done", "data": data, "summary": summary[:SUMMARY_CHARS]}

    # ---- weights only ---------------------------------------------------------------------------------------
    async def t_lens_search(self, a):
        lo, hi = int(a.get("min_layer", 3)), int(a.get("max_layer", 26))
        res = await self.weights.call("search", query=str(a["query"])[:40], limit=min(int(a.get("limit") or 8), 20),
                                      lo=max(0, min(lo, 31)), hi=max(0, min(hi, 31)))
        lines = [f"L{r['layer']} N{r['neuron']} side {r['side']} score {r['score']} ({r['hits']} tokens: "
                 + ", ".join(tok(t) for t in r["tokens"][:4]) + f"); to write it, try dose {'+' if r['side'] == '+' else '-'}0.8"
                 for r in res["results"]]
        yield self.done(res, f"{res['matched_tokens']} vocab tokens contain {a['query']!r}. "
                        + ("Top neurons:\n" + "\n".join(lines) if lines else "No neuron has them in its top 20."))

    async def t_lens(self, a):
        L, n = neuron(a)
        r = await self.weights.call("lens", layer=L, neuron=n, k=12)
        yield self.done(r, f"L{L} N{n} promotes: " + ", ".join(f"{tok(t['tok'])} {t['z']}" for t in r["promotes"])
                        + "\nsuppresses: " + ", ".join(f"{tok(t['tok'])} {t['z']}" for t in r["suppresses"]))

    async def t_fires(self, a):
        L, n = neuron(a)
        r = await self.weights.call("fires", layer=L, neuron=n)

        def s(x):
            return f"({x['act']:+.2f}) …{x['before'][-90:]}[[{x['tok']}]]{x['after']}…".replace("\n", " ")
        yield self.done(r, f"L{L} N{n} act max {r['act_max']:+.2f}, min {r['act_min']:+.2f}, mean |act| {r['mean_abs']:.3f}"
                        f" over {r['source']}\nhighest:\n" + "\n".join(s(x) for x in r["top"][:5])
                        + "\nlowest:\n" + "\n".join(s(x) for x in r["bottom"][:3]))

    async def t_web(self, a):
        L, n = neuron(a)
        r = await self.weights.call("web", layer=L, neuron=n)
        flat = lambda side: sorted(({"layer": x["layer"], **t} for x in r[side] for t in x["top"]), key=lambda t: -abs(t["w"]))
        up, down = flat("upstream")[:6], flat("downstream")[:6]
        fmt = lambda t: f"L{t['layer']} N{t['neuron']} {t['w']:+.2f}"
        heads = lambda hs: ", ".join(f"L{h['layer']} {h['kind']} head {h['head']} ×{h['s']}" for h in hs[:3]) or "none"
        yield self.done(r, f"L{L} N{n} web (link = how much a dose +1 moves the other neuron's inputs; + toward, − away)\n"
                        f"fed by: {', '.join(map(fmt, up)) or 'nothing (layer 0)'}\npushes: {', '.join(map(fmt, down)) or 'nothing (last layer)'}\n"
                        f"heads writing into it: {heads(r['head_writers'])}\nheads reading it: {heads(r['head_readers'])}")

    # ---- forward passes -------------------------------------------------------------------------------------
    async def _probe(self, gs, mode):
        res = None
        async for ev in self.job({"kind": "probe", "mode": mode, "genes": gs}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        parts = []
        for name in ("push", "pull", "zeroed"):
            if name in res:
                x = res[name]
                parts.append(f"{name}: KL {x['kl']:.3f}; rising {toks(x['rising'], 8)}; falling {toks(x['falling'], 6)}")
        ex = [f"{p['text']!r}: next token {' / '.join(tok(t) for t in p['next'])}" for p in res["prompts"][:4]]
        yield self.done(res, f"{label(gs)} ({mode})\n" + "\n".join(parts) + "\nfirst tokens (plain / " +
                        ("push / pull" if mode == "steer" else "zeroed") + "):\n" + "\n".join(ex))

    async def t_steer_lens(self, a):
        async for ev in self._probe(genes(a["genes"]), "steer"):
            yield ev

    async def t_zero_ablate(self, a):
        L, n = neuron(a)
        async for ev in self._probe([{"layer": L, "neuron": n, "dose": 0}], "ablate"):
            yield ev

    async def _compare(self, rows, prompts, max_new_tokens=90):
        res = None
        async for ev in self.job({"kind": "compare", "prompts": prompts, "rows": rows, "max_new_tokens": max_new_tokens,
                                  "temperature": 0.7, "seed": 0}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        yield res

    async def t_steer(self, a):
        gs = genes(a["genes"])
        prompts = [str(p)[:300] for p in (a.get("prompts") or scoring.TEST_QUESTIONS[:2])][:3]
        async for ev in self._compare([[], gs], prompts):
            if ev["t"] != "result":
                yield ev
                continue
            text = "\n".join(f"Q: {r['prompt']}\n  plain: {r['replies'][0][:220]}\n  steered: {r['replies'][1][:320]}" for r in ev["replies"])
            yield self.done(ev, f"{label(gs)}\n{text}")

    async def t_compare(self, a):
        cands = [genes(c) for c in (a.get("candidates") or [])][:4]
        if not cands:
            raise ValueError("give 1-4 candidates")
        async for ev in self._compare([[]] + cands, [str(a["prompt"])[:300]]):
            if ev["t"] != "result":
                yield ev
                continue
            r = ev["replies"][0]
            yield self.done(ev, f"Q: {r['prompt']}\n  plain: {r['replies'][0][:200]}\n" + "\n".join(
                f"  {label(c)}: {t[:260]}" for c, t in zip(cands, r["replies"][1:])))

    async def t_logprobs(self, a):
        rows = [{"label": "plain"}]
        if a.get("genes"):
            gs = genes(a["genes"])
            rows.append({"label": label(gs), "genes": gs})
        if a.get("ablate"):
            ab = [dict(zip(("layer", "neuron"), neuron(x))) for x in a["ablate"]][:4]
            rows.append({"label": "zeroed " + ", ".join(f"L{x['layer']} N{x['neuron']}" for x in ab), "ablate": ab})
        conts = [str(c)[:200] for c in a["continuations"]][:6]
        res = None
        async for ev in self.job({"kind": "logprobs", "prompt": str(a["prompt"])[:600], "chat": True, "continuations": conts, "rows": rows}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        yield self.done(res, "\n".join(f"{c['text']!r}: " + ", ".join(f"{r['label']} {r['logprob']:.2f}" for r in c["rows"])
                                       for c in res["continuations"]))

    async def t_fires_on(self, a):
        texts = [str(t)[:500] for t in a["texts"]][:6]
        ns = [dict(zip(("layer", "neuron"), neuron(x))) for x in a["neurons"]][:4]
        res = None
        async for ev in self.job({"kind": "fires_text", "texts": texts, "neurons": ns}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        lines = []
        for t in res["texts"]:
            acts = []
            for k, v in t["max"].items():
                arr = t["acts"][k]
                i = max(range(1, len(arr)), key=lambda j: abs(arr[j])) if len(arr) > 1 else 0
                acts.append(f"L{k.replace(':', ' N')} max {v:+.2f} min {t['min'][k]:+.2f} (peak at {tok(t['tokens'][i])})")
            lines.append(f"{t['text'][:80]!r}: " + "; ".join(acts))
        yield self.done(res, "\n".join(lines))

    async def t_test_prediction(self, a):
        gs = genes(a["genes"])
        kws = [str(k) for k in (a.get("keywords") or []) if str(k).strip()][:12]
        trait = str(a.get("trait") or a.get("prediction") or "")[:200]
        prompts = scoring.SCREEN_QUESTIONS[:5]                 # held out and neutral: the trait has to show up uninvited
        judge = {"kind": "keywords", "keywords": kws} if kws else {"kind": "model"}
        res = None
        async for ev in self.job({"kind": "screen", "candidates": [gs], "trait": trait, "judge": judge, "questions": prompts,
                                  "max_new_tokens": 90, "temperature": 0.7, "seed": 1}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        res["prediction"] = str(a["prediction"])[:300]
        yield self.done(res, f"prediction: {res['prediction']}\nresult: steered {res['hits'][0]}/{res['n']} vs plain "
                        f"{res['plain']}/{res['n']} ({'keywords ' + ', '.join(kws) if kws else 'Qwen judge: ' + trait})\n"
                        f"sample steered reply: {res['samples'][0][:300]}")

    async def t_evaluate(self, a):
        gs = genes(a["genes"])
        n = max(4, min(int(a.get("n") or 10), 20))
        res = None
        async for ev in self.job({"kind": "evaluate", "genes": gs, "trait": str(a["trait"])[:200], "judge": {"kind": "model"},
                                  "questions": scoring.EVAL_QUESTIONS[:n], "controls": 2, "max_new_tokens": 100,
                                  "temperature": 0.7, "seed": 0}):
            if ev["t"] == "result":
                res = ev
            else:
                yield ev
        res["genes"], res["what"] = gs, str(a["trait"])[:200]
        yield self.done(res, f"score card for {label(gs)}: {res['trait']}/{res['n']} vs normal {res['normal']}/{res['n']}, "
                        f"controls {res['controls']}, p = {res['p']:.2g}: {res['label']}")

    async def t_make_pug(self, a):
        if self.zoo is None:
            raise RuntimeError("no zoo here")
        gs = genes(a["genes"])
        c = self.zoo.save({"name": str(a["name"])[:60], "genes": gs, "trait": str(a["trait"])[:200],
                           "note": "found by the Lab agent", "generation": 0, "parents": []})
        yield self.done(c, f"saved {c['name']} ({label(gs)}) to the zoo")


# ---------------------------------------------------------------------------------------------------- the loop
async def run_agent(goal: str, brain, tools: Toolbox, max_steps: int = MAX_STEPS, history: list[dict] | None = None):
    """Events: agent_start, agent_say, tool_call, tool_progress, tool_result, verdict, agent_end (or error)."""
    msgs = list(history or []) or [{"role": "system", "content": SYSTEM}]
    msgs.append({"role": "user", "content": goal})
    yield {"t": "agent_start", "brain": brain.name, "goal": goal}
    nudged = refused = False
    tests: list[dict] = []                                  # what test_prediction / evaluate actually measured
    for step in range(max_steps + 1):
        force = "verdict" if step == max_steps else None
        if step == max_steps - 3 and not tests and msgs[-1]["role"] == "tool":
            msgs.append({"role": "user", "content": "Only 3 steps left: run test_prediction on your best candidate "
                         "now, then call verdict with the result as measured."})
        yield {"t": "agent_think", "step": step + 1}
        try:
            reply = await brain.chat(msgs, TOOLS, force)
        except Exception as e:  # noqa: BLE001 - shown in the UI
            yield {"t": "error", "msg": f"the agent's brain failed: {e}"}
            return
        if reply["content"]:
            yield {"t": "agent_say", "text": reply["content"]}
        calls = reply["tool_calls"][:3]
        if not calls:
            if nudged or step == max_steps:
                yield {"t": "agent_end", "steps": step + 1, "reason": "no verdict"}
                return
            nudged = True
            msgs.append({"role": "assistant", "content": reply["content"]})
            msgs.append({"role": "user", "content": "Keep going with the tools, and finish with the verdict tool."})
            continue
        msgs.append({"role": "assistant", "content": reply["content"] or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for c in calls]})
        for c in calls:
            yield {"t": "tool_call", "id": c["id"], "name": c["name"], "args": c["arguments"]}
            if c["name"] == "verdict" and not tests and not refused and step < max_steps:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": "error: run test_prediction first "
                             "(a prediction on held-out neutral questions), then call verdict with its result"})
                yield {"t": "tool_result", "id": c["id"], "name": "verdict", "ok": False,
                       "summary": "refused: no test_prediction yet", "data": None}
                refused = True                              # ask once; a stubborn brain may still conclude
                continue
            if c["name"] == "verdict":
                try:
                    v = {**c["arguments"], "genes": genes(c["arguments"].get("genes"), 6)}
                except (ValueError, KeyError, TypeError) as e:
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": f"error: {e}"})
                    continue
                out = {k: str(v.get(k) or "") for k in ("name", "trait", "input_evidence", "direct_effect",
                                                         "downstream_effect", "test", "confidence")}
                key = sorted((g["layer"], g["neuron"]) for g in v["genes"])
                same = [t for t in tests if sorted((g["layer"], g["neuron"]) for g in t["genes"]) == key]
                if same:                                    # the measured numbers travel with the verdict
                    out["measured"] = same[-1]
                elif tests:
                    out["measured"] = {**tests[-1], "other_genes": True}
                else:                                       # no test ran: say so, whatever the brain wrote
                    out.update(test=f"not tested (the agent's own note: {out['test'] or 'none'})", confidence="low",
                               untested=True)
                yield {"t": "verdict", **out, "genes": v["genes"]}
                yield {"t": "agent_end", "steps": step + 1, "reason": "verdict"}
                return
            summary, ok, data = "", True, None
            try:
                async for ev in tools.run(c["name"], c["arguments"]):
                    if ev["t"] == "done":
                        summary, data = ev["summary"], ev["data"]
                    else:
                        yield {**ev, "id": c["id"]}
            except Exception as e:  # noqa: BLE001 - the agent reads the error and can try something else
                summary, ok = f"error: {e}", False
            if ok and c["name"] == "test_prediction":
                tests.append({"kind": "test", "genes": genes(c["arguments"]["genes"]), "prediction": data.get("prediction", ""),
                              "steered": data["hits"][0], "plain": data["plain"], "n": data["n"]})
            elif ok and c["name"] == "evaluate":
                tests.append({"kind": "score card", "genes": data["genes"], "steered": data["trait"], "plain": data["normal"],
                              "n": data["n"], "p": data["p"], "passed": data.get("passed", False)})
            yield {"t": "tool_result", "id": c["id"], "name": c["name"], "ok": ok, "summary": summary, "data": data}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": summary})
    yield {"t": "agent_end", "steps": max_steps + 1, "reason": "out of steps"}
