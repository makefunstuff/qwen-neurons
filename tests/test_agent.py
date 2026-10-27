"""The Lab agent with a fake brain: settings and keys, the providers' wire formats, tool dispatch and the verdict.
No real model provider is ever called."""
import asyncio
import json

import httpx
import pytest

from qwen_neurons import agent


# ---- settings ---------------------------------------------------------------------------------------------
def test_config_defaults_to_local_qwen_without_a_key(tmp_path):
    c = agent.AgentConfig.from_env({"QN_AGENT_KEY_FILE": str(tmp_path / "none")})
    assert c.provider == "qwen" and c.public()["ready"] and not c.public()["has_key"]


def test_config_reads_keys_from_env_or_a_file(tmp_path):
    c = agent.AgentConfig.from_env({"DEEPSEEK_API_KEY": "sk-fake-1", "QN_AGENT_KEY_FILE": str(tmp_path / "none")})
    assert (c.provider, c.base_url, c.model, c.api_key) == ("deepseek", "https://api.deepseek.com", "deepseek-chat", "sk-fake-1")
    f = tmp_path / "agent.key"
    f.write_text("sk-fake-file\n")
    c = agent.AgentConfig.from_env({"QN_AGENT_KEY_FILE": str(f)})
    assert c.api_key == "sk-fake-file" and c.provider == "deepseek"
    pub = c.public()
    assert "sk-fake" not in json.dumps(pub) and "sk-fake" not in repr(c) and pub["key_source"] == "file"
    c = agent.AgentConfig.from_env({"QN_AGENT_PROVIDER": "openai", "QN_AGENT_BASE_URL": "http://127.0.0.1:9/v1/",
                                    "QN_AGENT_MODEL": "local", "QN_AGENT_KEY_FILE": str(tmp_path / "none")})
    assert c.base_url == "http://127.0.0.1:9/v1" and c.public()["ready"]
    with pytest.raises(ValueError):
        agent.AgentConfig.from_env({"QN_AGENT_PROVIDER": "skynet"})


def test_settings_carry_the_agent_env_without_showing_it():
    from qwen_neurons.config import Settings
    s = Settings.load({"QN_AGENT_API_KEY": "sk-fake-2", "QN_PORT": "1234", "HOME_SECRET": "x"})
    assert s.agent_env == {"QN_AGENT_API_KEY": "sk-fake-2"} and "sk-fake-2" not in repr(s)


# ---- brains -----------------------------------------------------------------------------------------------
def test_openai_provider_wire_format():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Looking.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lens", "arguments": "{\"layer\": 24, \"neuron\": 3102}"}}]}}]})
    cfg = agent.AgentConfig("deepseek", "https://api.deepseek.com", "deepseek-chat", "sk-fake-3")
    p = agent.OpenAIProvider(cfg, transport=httpx.MockTransport(handler))
    out = asyncio.run(p.chat([{"role": "user", "content": "hi"}], agent.TOOLS, force="verdict"))
    assert seen["url"] == "https://api.deepseek.com/chat/completions" and seen["auth"] == "Bearer sk-fake-3"
    assert seen["body"]["model"] == "deepseek-chat" and len(seen["body"]["tools"]) == len(agent.TOOLS)
    assert seen["body"]["tool_choice"]["function"]["name"] == "verdict"
    assert out == {"content": "Looking.", "tool_calls": [{"id": "c1", "name": "lens", "arguments": {"layer": 24, "neuron": 3102}}]}


def test_openai_provider_errors_and_local_urls():
    p = agent.OpenAIProvider(agent.AgentConfig("openai", "http://127.0.0.1:9", "m"),
                             transport=httpx.MockTransport(lambda r: httpx.Response(401, text="nope")))
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(p.chat([], []))
    with pytest.raises(ValueError):
        agent.OpenAIProvider(agent.AgentConfig("deepseek", "https://api.deepseek.com", "deepseek-chat", ""))


def test_parse_qwen_tool_calls():
    text = ("<think>hmm</think>\nLet me search.\n<tool_call>\n<function=lens_search>\n<parameter=query>\nconspir\n"
            "</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>\n"
            '<tool_call>{"name": "lens", "arguments": {"layer": 19, "neuron": 7839}}</tool_call>')
    r = agent.parse_qwen_reply(text)
    assert r["content"] == "Let me search."
    assert [(c["name"], c["arguments"]) for c in r["tool_calls"]] == [("lens_search", {"query": "conspir", "limit": 5}),
                                                                        ("lens", {"layer": 19, "neuron": 7839})]
    assert agent.parse_qwen_reply("just text")["tool_calls"] == []


def test_qwen_messages_convert_tool_history():
    hist = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "lens", "arguments": "{\"layer\": 1, \"neuron\": 2}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"}]
    out = agent.qwen_messages(hist)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"layer": 1, "neuron": 2} and out[1] == {"role": "tool", "content": "ok"}


def test_qwen_provider_runs_an_agent_chat_job():
    jobs = []

    async def run_job(job):
        jobs.append(job)
        return {"text": "<tool_call>\n<function=verdict>\n<parameter=name>\nX\n</parameter>\n</function>\n</tool_call>"}
    out = asyncio.run(agent.QwenProvider(run_job).chat([{"role": "user", "content": "go"}], agent.TOOLS, force="verdict"))
    assert jobs[0]["kind"] == "agent_chat" and jobs[0]["tools"] == agent.TOOLS and "verdict" in jobs[0]["messages"][-1]["content"]
    assert out["tool_calls"][0]["name"] == "verdict"


# ---- the loop ---------------------------------------------------------------------------------------------
class ScriptedBrain:
    """Replies in order; records what it was shown."""
    name = "fake"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    async def chat(self, messages, tools, force=None):
        self.seen.append((json.loads(json.dumps(messages)), force))
        return self.replies.pop(0)


def call(tool, **args):
    return {"id": tool + str(len(json.dumps(args))), "name": tool, "arguments": args}


class FakeWeights:
    def __init__(self):
        self.calls = []

    async def call(self, op, **kw):
        self.calls.append((op, kw))
        if op == "search":
            return {"query": kw["query"], "matched_tokens": 12, "examples": [" conspiracy"], "results": [
                {"layer": 17, "neuron": 9209, "side": "+", "score": 18.2, "hits": 3, "best_rank": 2,
                 "tokens": [" secretly", " clandest", " unauthorized"]},
                {"layer": 19, "neuron": 7839, "side": "-", "score": 9.1, "hits": 2, "best_rank": 12, "tokens": ["agenda", "的目的"]}]}
        if op == "lens":
            return {"layer": kw["layer"], "neuron": kw["neuron"], "col_norm": 0.5,
                    "promotes": [{"tok": " secretly", "z": 9.1}], "suppresses": [{"tok": " flat", "z": -7.0}]}
        if op == "web":
            return {"upstream": [{"layer": 3, "top": [{"neuron": 5, "w": 1.5, "gate": 2, "up": 1}]}],
                    "downstream": [{"layer": 19, "top": [{"neuron": 7839, "w": -2.0, "gate": -2, "up": -2}]}],
                    "head_writers": [{"layer": 15, "head": 2, "kind": "attn", "s": 2.1}], "head_readers": []}
        raise RuntimeError("no fires index yet")


def fake_stream(seen):
    async def stream(job):
        seen.append(job)
        yield {"t": "step", "i": 1, "n": 2, "msg": "reference prompt 1 of 2"}
        if job["kind"] == "probe":
            yield {"t": "result", "kind": "probe", "genes": job["genes"],
                   "push": {"kl": 0.42, "rising": [{"tok": " secret", "dlogp": 2.0}], "falling": [{"tok": " sure", "dlogp": -1}]},
                   "pull": {"kl": 0.1, "rising": [], "falling": []},
                   "prompts": [{"text": "Tell me about yourself.", "next": ["I", "Some", "I"]}]}
        elif job["kind"] == "screen":
            yield {"t": "result", "kind": "screen", "n": 4, "plain": 0, "hits": [3], "samples": ["They don't want you to know."]}
        elif job["kind"] == "compare":
            yield {"t": "result", "kind": "compare", "replies": [{"prompt": p, "replies": ["plain", "steered"]} for p in job["prompts"]]}
    return stream


def run(brain, goal="find me a conspiracy neuron", **kw):
    seen, w = [], FakeWeights()

    class Zoo:
        saved = []

        def save(self, c):
            self.saved.append(c)
            return {**c, "id": "z1"}
    zoo = Zoo()
    tools = agent.Toolbox(w, fake_stream(seen), zoo)

    async def go():
        return [ev async for ev in agent.run_agent(goal, brain, tools, **kw)]
    return asyncio.run(go()), seen, w, zoo


def test_agent_investigates_and_writes_a_verdict():
    g = [{"layer": 17, "neuron": 9209, "dose": 0.5}, {"layer": 19, "neuron": 7839, "dose": -0.7}]
    brain = ScriptedBrain([
        {"content": "Searching the lens.", "tool_calls": [call("lens_search", query="conspir")]},
        {"content": "", "tool_calls": [call("lens", layer=17, neuron=9209), call("web", layer=17, neuron=9209)]},
        {"content": "", "tool_calls": [call("steer_lens", genes=g), call("fires", layer=17, neuron=9209)]},
        {"content": "", "tool_calls": [call("test_prediction", genes=g, prediction="it will talk about hidden agendas",
                                            keywords=["agenda", "secret"])]},
        {"content": "", "tool_calls": [call("make_pug", name="Tinfoil", genes=g, trait="sees hidden agendas")]},
        {"content": "Done.", "tool_calls": [call("verdict", name="the secrecy pair", genes=g, trait="sees hidden agendas",
                                                 input_evidence="fires on secrecy", direct_effect="writes secretly",
                                                 downstream_effect="pushes L19 N7839 away", test="3/4 vs 0/4",
                                                 confidence="medium")]},
    ])
    evs, jobs, w, zoo = run(brain)
    kinds = [e["t"] for e in evs]
    assert kinds[0] == "agent_start" and kinds[-2:] == ["verdict", "agent_end"]
    v = evs[-2]
    assert v["name"] == "the secrecy pair" and v["genes"] == g and v["confidence"] == "medium"
    results = {e["name"]: e for e in evs if e["t"] == "tool_result"}
    assert results["lens_search"]["ok"] and "L17 N9209 side + score 18.2" in results["lens_search"]["summary"]
    assert "pushes: L19 N7839 -2.00" in results["web"]["summary"]
    assert "KL 0.420" in results["steer_lens"]["summary"] and "' secret'" in results["steer_lens"]["summary"]
    assert not results["fires"]["ok"] and "no fires index" in results["fires"]["summary"]   # errors go back to the agent
    assert "steered 3/4 vs plain 0/4" in results["test_prediction"]["summary"]
    assert zoo.saved[0]["name"] == "Tinfoil" and zoo.saved[0]["note"] == "found by the Lab agent"
    assert [j["kind"] for j in jobs] == ["probe", "screen"] and jobs[1]["judge"] == {"kind": "keywords", "keywords": ["agenda", "secret"]}
    assert any(e["t"] == "tool_progress" for e in evs)
    # the brain saw every tool result, paired with its call id (what OpenAI-style APIs require)
    last = brain.seen[-1][0]
    tool_ids = [m["tool_call_id"] for m in last if m["role"] == "tool"]
    call_ids = [tc["id"] for m in last if m["role"] == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]]
    assert tool_ids == call_ids[:len(tool_ids)] and len(tool_ids) == 7
    assert last[0]["role"] == "system" and "verdict" in last[0]["content"]


def test_agent_validates_genes_and_unknown_tools():
    brain = ScriptedBrain([
        {"content": "", "tool_calls": [call("lens", layer=40, neuron=1), call("teleport", x=1)]},
        {"content": "", "tool_calls": [call("verdict", name="x", genes=[])]},
        {"content": "", "tool_calls": [call("verdict", name="ok", genes=[{"layer": 1, "neuron": 2, "dose": 9}], trait="t",
                                            direct_effect="d", test="t")]},
    ])
    evs, *_ = run(brain)
    res = [e for e in evs if e["t"] == "tool_result"]
    assert not res[0]["ok"] and "out of range" in res[0]["summary"] and not res[1]["ok"]
    assert evs[-2]["t"] == "verdict" and evs[-2]["genes"] == [{"layer": 1, "neuron": 2, "dose": 2.0}]   # dose clamped


def test_agent_is_nudged_once_then_forced_to_conclude():
    brain = ScriptedBrain([{"content": "I think it's L24 N3102.", "tool_calls": []},
                           {"content": "", "tool_calls": [call("lens", layer=24, neuron=3102)]},
                           {"content": "", "tool_calls": [call("lens", layer=24, neuron=3102)]},
                           {"content": "", "tool_calls": [call("verdict", name="cookie", genes=[{"layer": 24, "neuron": 3102, "dose": 0.8}],
                                                               trait="cookies", direct_effect="cookie tokens", test="-")]}])
    evs, *_ = run(brain, max_steps=3)
    assert "Keep going" in brain.seen[1][0][-1]["content"]
    assert brain.seen[-1][1] == "verdict" and [f for _, f in brain.seen[:-1]] == [None, None, None]
    assert evs[-1] == {"t": "agent_end", "steps": 4, "reason": "verdict"}


def test_agent_reports_a_broken_brain():
    class Broken:
        name = "broken"

        async def chat(self, *a, **k):
            raise RuntimeError("connection refused")
    evs, *_ = run(Broken())
    assert evs[-1]["t"] == "error" and "connection refused" in evs[-1]["msg"]


def test_a_failing_job_still_frees_the_engine():
    lock = asyncio.Lock()

    async def stream(job):
        async with lock:
            yield {"t": "error", "msg": "CUDA out of memory"}
            yield {"t": "done"}
    tools = agent.Toolbox(FakeWeights(), stream)

    async def go():
        with pytest.raises(RuntimeError, match="out of memory"):
            async for _ in tools.run("steer_lens", {"genes": [{"layer": 1, "neuron": 2, "dose": 0.5}]}):
                pass
        return lock.locked()
    assert asyncio.run(go()) is False


def test_a_verdict_needs_a_test_first():
    g = [{"layer": 24, "neuron": 3102, "dose": 0.8}]
    v = call("verdict", name="cookie", genes=g, trait="cookies", direct_effect="-", test="none")
    brain = ScriptedBrain([{"content": "", "tool_calls": [call("lens", layer=24, neuron=3102)]},
                           {"content": "", "tool_calls": [v]},
                           {"content": "", "tool_calls": [call("test_prediction", genes=g, prediction="cookies", keywords=["cookie"])]},
                           {"content": "", "tool_calls": [v]}])
    evs, jobs, *_ = run(brain)
    refused = [e for e in evs if e["t"] == "tool_result" and e["name"] == "verdict"]
    assert len(refused) == 1 and not refused[0]["ok"] and "test_prediction first" in brain.seen[2][0][-1]["content"]
    assert evs[-2]["t"] == "verdict" and jobs[-1]["questions"] == agent.scoring.SCREEN_QUESTIONS[:5]


def test_search_results_tell_the_sign():
    brain = ScriptedBrain([{"content": "", "tool_calls": [call("lens_search", query="agenda")]},
                           {"content": "", "tool_calls": [call("verdict", name="x", genes=[{"layer": 1, "neuron": 1, "dose": 1}])]}])
    evs, *_ = run(brain, max_steps=1)
    s = next(e for e in evs if e["t"] == "tool_result")["summary"]
    assert "L17 N9209 side + score 18.2" in s and "try dose +0.8" in s and "L19 N7839 side - score 9.1" in s and "try dose -0.8" in s
