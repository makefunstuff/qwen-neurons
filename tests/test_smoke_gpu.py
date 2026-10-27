"""Live smoke test on the real model. Skips unless the model is set (QN_MODEL, a local dir) and the GPU named by
QN_TEST_DEVICE (default cuda:1) has room; it never touches other GPU processes. With QN_TEST_BUSY_URL set to another
Evil Pugs' /api/status (e.g. a service sharing the GPU), it also skips unless that one is asleep."""
import asyncio
import json
import os
import urllib.request

import pytest

from qwen_neurons import gpu
from qwen_neurons.config import Settings
from qwen_neurons.engine import Engine


def _settings():
    s = Settings.load()
    s.device = os.environ.get("QN_TEST_DEVICE", "cuda:1")
    return s


def _skip_reason():
    s = _settings()
    if not os.path.isdir(s.model):
        return f"QN_MODEL is not a local model dir ({s.model})"
    url = os.environ.get("QN_TEST_BUSY_URL")
    if url:
        try:
            st = json.load(urllib.request.urlopen(url, timeout=3))
        except OSError as e:
            return f"GPU busy? could not read {url}: {e}"
        if st.get("state") != "asleep" or st.get("busy"):
            return f"GPU busy: the service at {url} is {st.get('state')}"
    pick = gpu.pick_device(s.device, s.need_mib)
    if not pick["ok"]:
        return "GPU busy: " + pick["msg"]
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_cookie_neuron_talks_about_cookies():
    async def go():
        e = Engine(_settings())
        out = ["", ""]
        try:
            async for ev in e.run({"kind": "generate", "messages": [{"role": "user", "content": "Who are you?"}],
                                   "rows": [[], [{"layer": 24, "neuron": 3102, "dose": 0.8}]], "think": False,
                                   "max_new_tokens": 60, "temperature": 0}):
                if ev["t"] == "tok":
                    out[ev["row"]] = ev.get("text_full", out[ev["row"]] + (ev.get("text") or ""))
                assert ev["t"] != "error", ev
        finally:
            await e.unload()
        return out
    normal, cookie = asyncio.run(go())
    assert normal and "cookie" not in normal.lower()
    assert "cookie" in cookie.lower()


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_discover_and_evaluate_live():
    from qwen_neurons import discovery, scoring

    async def go():
        e = Engine(_settings())
        out = {}
        try:
            async for ev in e.run({"kind": "discover", "a": discovery.PIRATE, "b": discovery.NORMAL,
                                   "questions": discovery.NEUTRAL_QUESTIONS, "layers": [8, 9, 10], "metric": "writer",
                                   "top_k": 5, "per_layer_k": 2, "dose": 0.8}):
                assert ev["t"] != "error", ev
                if ev["t"] == "result":
                    out["discover"] = ev
            async for ev in e.run({"kind": "evaluate", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}],
                                   "trait": "mentions cookies", "judge": {"kind": "keywords", "keywords": ["cookie"]},
                                   "questions": scoring.EVAL_QUESTIONS[:4], "controls": 1, "max_new_tokens": 60,
                                   "temperature": 0.7, "seed": 0}):
                assert ev["t"] != "error", ev
                if ev["t"] == "result":
                    out["evaluate"] = ev
        finally:
            await e.unload()
        return out
    out = asyncio.run(go())
    d = out["discover"]
    assert len(d["top"]) == 5 and {c["layer"] for c in d["top"]} <= {8, 9, 10}
    assert all(c["col_norm"] > 0 and c["median_resid"] > 0 for c in d["top"])
    v = out["evaluate"]
    assert v["n"] == 4 and v["trait"] >= 2 and v["normal"] == 0


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_party_two_pugs_take_turns_live():
    from qwen_neurons import party
    cookie = {"id": "cookie", "name": "Cookie", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}]}
    caps = {"id": "all-caps", "name": "ALL CAPS", "genes": [{"layer": 17, "neuron": 1991, "dose": 0.8}]}

    async def go():
        e = Engine(_settings())
        transcript, turns = [{"kind": "human", "text": "Hi both! What did you have for breakfast?"}], []
        try:
            for i in range(2):
                plan = party.plan_turn([cookie, caps], transcript, "a breakfast chat", seed=i, chaos=0.5)
                turn = {"who": plan["speaker"], "reactions": []}
                async for ev in e.run({"kind": "party", **plan, "max_new_tokens": 60, "policy": "steer", "seed": i}):
                    assert ev["t"] != "error", ev
                    if ev["t"] == "wearing":
                        turn["layers"] = ev["layers"]
                    if ev["t"] == "said":
                        turn["text"] = ev["text"]
                    if ev["t"] == "reaction":
                        turn["reactions"].append(ev)
                transcript.append({"kind": "pug", "who": plan["speaker"], "name": plan["name"], "text": turn["text"]})
                turns.append(turn)
        finally:
            await e.unload()
        return turns
    t1, t2 = asyncio.run(go())
    assert t1["who"] == "cookie" and t2["who"] == "all-caps"       # the steering swapped with the speaker
    assert t1["layers"] == [24] and t2["layers"] == [17]
    assert t1["text"] and t2["text"] and len(t1["reactions"]) == 1 and t1["reactions"][0]["who"] == "all-caps"
    assert "cookie" in t1["text"].lower()
    letters = [c for c in t2["text"] if c.isalpha()]
    assert sum(c.isupper() for c in letters) > 0.6 * len(letters)
