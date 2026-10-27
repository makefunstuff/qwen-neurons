"""Live smoke test on the real model. Skips unless the model is set (QN_MODEL, a local dir) and the GPU named by
QN_TEST_DEVICE (default cuda:1) has room; it never touches other GPU processes."""
import asyncio
import os

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
