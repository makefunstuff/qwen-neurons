"""The explorer and agent endpoints with a fake weights process, a fake engine and a fake brain."""
import json

import pytest
from fastapi.testclient import TestClient

from qwen_neurons.config import Settings
from qwen_neurons.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings.load({"QN_DATA_DIR": str(tmp_path), "QN_AGENT_API_KEY": "sk-fake-server", "QN_AGENT_PROVIDER": "openai",
                       "QN_AGENT_BASE_URL": "http://127.0.0.1:9/v1", "QN_AGENT_MODEL": "fake"})
    app = create_app(s)
    calls, jobs = [], []

    async def fake_call(op, **kw):
        calls.append((op, kw))
        if op == "search":
            return {"query": kw["query"], "matched_tokens": 3, "examples": [" cookie"], "layers": [kw["lo"], kw["hi"]],
                    "results": [{"layer": 24, "neuron": 3102, "side": "+", "score": 154.5, "hits": 9, "best_rank": 1, "tokens": [" cookie"]}]}
        if op == "fires":
            raise RuntimeError("no fires index yet")
        return {"op": op, **kw}

    async def fake_stream(op, **kw):
        yield {"t": "progress", "frac": 0.5, "msg": "half"}
        yield {"t": "result", "data": {}}

    async def fake_run(job):
        jobs.append(job)
        if job["kind"] == "probe":
            yield {"t": "result", "kind": "probe", "genes": job["genes"], "push": {"kl": 0.3, "rising": [], "falling": []},
                   "pull": {"kl": 0.1, "rising": [], "falling": []}, "prompts": []}
        else:
            yield {"t": "result", "kind": job["kind"]}
    monkeypatch.setattr(app.state.weights, "call", fake_call)
    monkeypatch.setattr(app.state.weights, "stream", fake_stream)
    monkeypatch.setattr(app.state.engine, "run", fake_run)
    monkeypatch.setattr("qwen_neurons.engine.gpu.gpus", lambda: [])
    c = TestClient(app)
    c.app, c.calls, c.jobs = app, calls, jobs
    return c


def events(r):
    return [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]


def test_status_names_the_brain_but_never_the_key(client):
    st = client.get("/api/x/status").json()
    assert st["index"] is False and st["fires"] is False and st["agent"]["provider"] == "openai"
    assert "sk-fake-server" not in json.dumps(st) and "sk-fake-server" not in client.get("/api/agent").text


def test_weights_endpoints_validate_and_forward(client):
    assert client.get("/api/x/lens/24/3102").json() == {"op": "lens", "layer": 24, "neuron": 3102, "k": 16}
    assert client.get("/api/x/web/3/7").json()["op"] == "web"
    assert client.get("/api/x/card/3/7").json()["op"] == "card"
    assert client.get("/api/x/lens/32/1").status_code == 422 and client.get("/api/x/web/1/9216").status_code == 422
    r = client.get("/api/x/search", params={"q": "cookie"}).json()
    assert r["results"][0]["neuron"] == 3102 and client.calls[-1] == ("search", {"query": "cookie", "limit": 30, "lo": 3, "hi": 26})
    client.get("/api/x/search", params={"q": "cookie", "lo": 40, "hi": -2})
    assert client.calls[-1][1]["lo"] == 0 and client.calls[-1][1]["hi"] == 31
    assert client.get("/api/x/search", params={"q": " "}).status_code == 400
    assert client.get("/api/x/fires/1/2").status_code == 409          # not built yet: a clear message, not a 500
    assert client.get("/api/x/map", params={"color": "rainbow"}).status_code == 400


def test_index_build_runs_in_the_background(client):
    r = client.post("/api/x/index").json()
    assert r["started"] and r["building"]["state"] == "running"


def test_model_endpoints_queue_jobs(client):
    ev = events(client.post("/api/x/probe", json={"genes": [{"layer": 19, "neuron": 7839, "dose": -0.7}], "mode": "steer"}))
    assert ev[0]["push"]["kl"] == 0.3 and client.jobs[-1] == {"kind": "probe", "mode": "steer", "genes": [{"layer": 19, "neuron": 7839, "dose": -0.7}]}
    assert client.post("/api/x/probe", json={"genes": [{"layer": 1, "neuron": 2, "dose": 3}]}).status_code == 422
    client.post("/api/x/fires-text", json={"texts": ["They are hiding it."], "neurons": [{"layer": 17, "neuron": 9209}]})
    assert client.jobs[-1]["kind"] == "fires_text"
    client.post("/api/x/logprobs", json={"prompt": "Who runs the world?", "continuations": [" They"], "rows": [{"label": "plain"}, {"ablate": [{"layer": 17, "neuron": 9209}]}]})
    assert client.jobs[-1]["kind"] == "logprobs" and client.jobs[-1]["rows"][1]["ablate"] == [{"layer": 17, "neuron": 9209}]
    client.post("/api/x/fires-build", json={})
    assert client.jobs[-1]["kind"] == "fires_build" and client.jobs[-1]["path"].endswith("fires.pt")


def test_agent_endpoint_streams_a_case(client):
    class Brain:
        name = "fake brain"

        def __init__(self):
            self.n = 0

        async def chat(self, messages, tools, force=None):
            self.n += 1
            if self.n == 1:
                return {"content": "Searching.", "tool_calls": [{"id": "a", "name": "lens_search", "arguments": {"query": "cookie"}}]}
            if self.n == 2:
                return {"content": "", "tool_calls": [{"id": "b", "name": "steer_lens", "arguments": {"genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}]}}]}
            return {"content": "", "tool_calls": [{"id": "c", "name": "verdict", "arguments": {
                "name": "the cookie neuron", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}], "trait": "talks about cookies",
                "direct_effect": "writes ' cookie'", "test": "4/4 vs 0/4"}}]}
    client.app.state.brain = Brain()
    ev = events(client.post("/api/agent", json={"goal": "find me a cookie neuron"}))
    kinds = [e["t"] for e in ev]
    assert kinds[0] == "agent_start" and "verdict" in kinds and kinds[-1] == "agent_end"
    assert [j["kind"] for j in client.jobs] == ["probe"]
    assert ("search", {"query": "cookie", "limit": 8, "lo": 3, "hi": 26}) in client.calls
    v = next(e for e in ev if e["t"] == "verdict")
    assert v["name"] == "the cookie neuron" and v["genes"][0]["neuron"] == 3102
    assert client.post("/api/agent", json={"goal": ""}).status_code == 422


def test_agent_needs_a_key_for_deepseek(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = create_app(Settings.load({"QN_DATA_DIR": str(tmp_path), "QN_AGENT_PROVIDER": "deepseek",
                                    "QN_AGENT_KEY_FILE": str(tmp_path / "missing")}))
    c = TestClient(app)
    assert c.get("/api/agent").json()["ready"] is False
    r = c.post("/api/agent", json={"goal": "find me a cookie neuron"})
    assert r.status_code == 400 and "key" in r.json()["detail"]


def test_local_qwen_brain_frees_the_model_between_turns(tmp_path, monkeypatch):
    """With the local Qwen brain, every turn and every tool is a job on the same engine: each must give the job lock
    back (a regression: reading only up to the result left the lock held, so the next tool got 'Qwen is busy')."""
    import asyncio
    app = create_app(Settings.load({"QN_DATA_DIR": str(tmp_path), "QN_AGENT_PROVIDER": "qwen"}))
    lock = asyncio.Lock()
    replies = iter(['<tool_call>\n<function=steer_lens>\n<parameter=genes>\n[{"layer": 24, "neuron": 3102, "dose": 0.8}]\n'
                    '</parameter>\n</function>\n</tool_call>',
                    '<tool_call>\n<function=verdict>\n<parameter=name>\ncookie\n</parameter>\n<parameter=genes>\n'
                    '[{"layer": 24, "neuron": 3102, "dose": 0.8}]\n</parameter>\n</function>\n</tool_call>'])

    async def locked_run(job):                       # like Engine.run: busy if a job holds the lock
        if lock.locked():
            yield {"t": "error", "msg": "Qwen is busy with another reply"}
            return
        async with lock:
            if job["kind"] == "agent_chat":
                yield {"t": "result", "kind": "agent_chat", "text": next(replies)}
            else:
                yield {"t": "result", "kind": "probe", "genes": job["genes"], "push": {"kl": 0.2, "rising": [], "falling": []},
                       "pull": {"kl": 0.1, "rising": [], "falling": []}, "prompts": []}
            yield {"t": "after-result bookkeeping"}
    monkeypatch.setattr(app.state.engine, "run", locked_run)
    monkeypatch.setattr("qwen_neurons.engine.gpu.gpus", lambda: [])
    ev = events(TestClient(app).post("/api/agent", json={"goal": "find me a cookie neuron"}))
    res = next(e for e in ev if e["t"] == "tool_result")
    assert res["ok"], res
    assert [e["t"] for e in ev][-2:] == ["verdict", "agent_end"] and not lock.locked()
