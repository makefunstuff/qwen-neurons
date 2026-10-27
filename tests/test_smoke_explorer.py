"""Live checks of the explorer on the real Qwen3.5-4B weights.

The weights-only views run on the CPU whenever QN_MODEL is a local model dir (no GPU needed); the word search also
needs the lens index in the data dir (🧭 Explorer → build the index). The forward-pass tools run the agent's real
toolbox with a scripted brain (no LLM) on the GPU named by QN_TEST_DEVICE (default cuda:1), and skip when it is busy."""
import asyncio
import os

import pytest

from qwen_neurons import agent
from qwen_neurons.config import Settings
from qwen_neurons.weights import Box, cache_dir

from .test_smoke_gpu import _settings, _skip_reason

S = Settings.load()
HAS_MODEL = os.path.isdir(S.model)


@pytest.fixture(scope="module")
def box():
    return Box({"model": S.model, "cache_dir": str(cache_dir(S.data_dir, S.model))})


@pytest.mark.skipif(not HAS_MODEL, reason=f"QN_MODEL is not a local model dir ({S.model})")
def test_cookie_neuron_writes_cookie(box):
    lens = box.op_lens(24, 3102, 8)
    assert "cookie" in lens["promotes"][0]["tok"].lower()


@pytest.mark.skipif(not HAS_MODEL, reason=f"QN_MODEL is not a local model dir ({S.model})")
def test_web_covers_every_layer(box):
    w = box.op_web(19, 7839)
    assert [x["layer"] for x in w["upstream"]] == list(range(19)) and [x["layer"] for x in w["downstream"]] == list(range(20, 32))
    assert w["head_writers"] and w["head_readers"]


@pytest.mark.skipif(not HAS_MODEL, reason="no local model dir")
def test_search_rediscovers_zoo_and_conspiracy_neurons(box):
    if box.index() is None:
        pytest.skip("the lens index isn't built in the data dir")
    rank = lambda q, L, n: next((i + 1 for i, r in enumerate(box.op_search(q, 30, 3, 26)["results"])
                                 if (r["layer"], r["neuron"]) == (L, n)), None)
    assert rank("cookie", 24, 3102) == 1                     # Cookie
    assert rank("destiny", 24, 6441) == 1                    # Fate & destiny (side −: its dose is −0.8)
    assert rank("clandest", 17, 9209) == 1                   # Conspiracy's + gene writes "clandestine"
    assert rank("agenda", 19, 7839) <= 10                    # Conspiracy's − gene suppresses "agenda"


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
def test_toolbox_on_the_real_model(box):
    """The agent's real tools, driven by a script instead of an LLM."""
    from qwen_neurons.engine import Engine

    class W:                                                  # the weights process, in-process
        async def call(self, op, **kw):
            return getattr(box, "op_" + op)(**kw)

    g = [{"layer": 24, "neuron": 3102, "dose": 0.8}]
    script = [{"content": "", "tool_calls": [{"id": "1", "name": "lens", "arguments": {"layer": 24, "neuron": 3102}}]},
              {"content": "", "tool_calls": [{"id": "2", "name": "steer_lens", "arguments": {"genes": g}}]},
              {"content": "", "tool_calls": [{"id": "3", "name": "fires_on", "arguments": {
                  "texts": ["I baked chocolate chip cookies.", "The train leaves at noon."], "neurons": [{"layer": 24, "neuron": 3102}]}}]},
              {"content": "", "tool_calls": [{"id": "4", "name": "test_prediction", "arguments": {
                  "genes": g, "prediction": "it brings up cookies", "keywords": ["cookie"]}}]},
              {"content": "", "tool_calls": [{"id": "5", "name": "verdict", "arguments": {
                  "name": "cookie", "genes": g, "trait": "cookies", "direct_effect": "-", "test": "-"}}]}]

    class Brain:
        name = "script"

        async def chat(self, messages, tools, force=None):
            return script.pop(0)

    async def go():
        e = Engine(_settings())
        try:
            return [ev async for ev in agent.run_agent("cookie", Brain(), agent.Toolbox(W(), e.run))]
        finally:
            await e.unload()
    res = {e["name"]: e for e in asyncio.run(go()) if e["t"] == "tool_result"}
    assert all(r["ok"] for r in res.values()), res
    assert "cookie" in res["steer_lens"]["summary"].lower() and res["steer_lens"]["data"]["push"]["kl"] > 0.01
    fired = res["fires_on"]["data"]["texts"]
    assert fired[0]["max"]["24:3102"] > fired[1]["max"]["24:3102"]
    assert res["test_prediction"]["data"]["hits"][0] > res["test_prediction"]["data"]["plain"]
