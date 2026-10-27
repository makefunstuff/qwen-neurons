"""The API with a fake engine: no model, no GPU."""
import pytest
from fastapi.testclient import TestClient

from qwen_neurons.config import Settings
from qwen_neurons.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings.load({"QN_DATA_DIR": str(tmp_path)})
    app = create_app(s)
    seen = []

    async def fake_run(job):
        seen.append(job)
        yield {"t": "tok", "row": 0, "think": "", "text": "Arr"}
        yield {"t": "end", "row": 0, "reason": "eos", "tokens": 1}
    monkeypatch.setattr(app.state.engine, "run", fake_run)
    monkeypatch.setattr("qwen_neurons.engine.gpu.gpus", lambda: [])
    c = TestClient(app)
    c.seen = seen
    return c


def test_index_and_static(client):
    assert "Evil Pugs" in client.get("/").text
    assert client.get("/static/pug.js").status_code == 200


def test_status_asleep(client):
    s = client.get("/api/status").json()
    assert s["state"] == "asleep" and s["zones"]["loop"] == 1.5


def test_creatures_crud(client):
    zoo = client.get("/api/creatures").json()
    assert zoo[0]["name"] == "Evil Qwen" and len(zoo) == 24
    c = client.post("/api/creatures", json={"name": "Pupper", "genes": [{"layer": 1, "neuron": 2, "dose": 0.3}]}).json()
    assert len(client.get("/api/creatures").json()) == 25
    assert client.post("/api/creatures", json={"name": "x", "genes": [{"layer": 99, "neuron": 2, "dose": 0.3}]}).status_code == 400
    assert client.delete("/api/creatures/cookie").status_code == 403
    assert client.delete(f"/api/creatures/{c['id']}").status_code == 200
    assert client.delete(f"/api/creatures/{c['id']}").status_code == 404


def test_breed(client):
    zoo = {c["id"]: c for c in client.get("/api/creatures").json()}
    child = client.post("/api/breed", json={"a": zoo["cookie"], "b": zoo["evil-qwen"], "seed": 7}).json()
    assert child["generation"] == 1 and child["genes"] and child["name"]
    for L in {g["layer"] for g in child["genes"]}:   # per-layer cap only: children keep their traits' doses
        assert sum(abs(g["dose"]) for g in child["genes"] if g["layer"] == L) <= 1.21
    assert {(g["layer"], g["neuron"]) for g in child["genes"] + child["carried"]} >= {(24, 3102), (14, 8949), (13, 7956)} - {(24, 3102)}


def test_generate_streams_sse_and_drops_zero_doses(client):
    r = client.post("/api/generate", json={"messages": [{"role": "user", "content": "hi"}],
                                           "rows": [[{"layer": 1, "neuron": 2, "dose": 0.5}, {"layer": 3, "neuron": 4, "dose": 0}]]})
    assert r.status_code == 200
    assert 'data: {"t": "tok"' in r.text and '"Arr"' in r.text
    assert client.seen[0]["rows"] == [[{"layer": 1, "neuron": 2, "dose": 0.5}]]


def test_generate_validates(client):
    bad = client.post("/api/generate", json={"messages": [{"role": "user", "content": "hi"}],
                                             "rows": [[{"layer": 1, "neuron": 2, "dose": 5}]]})
    assert bad.status_code == 422


def test_sets_and_genepool(client):
    sets = client.get("/api/sets").json()
    assert any(s["id"] == "pirate" for s in sets)
    s = client.post("/api/sets", json={"name": "Robots", "texts": ["beep", "boop"]}).json()
    assert client.delete("/api/sets/villain").status_code == 403
    assert client.delete(f"/api/sets/{s['id']}").status_code == 200
    pool = client.post("/api/genepool", json=[{"layer": 3, "neuron": 4, "dose": 0.5, "label": "pirate"}]).json()
    assert pool[0]["neuron"] == 4
    assert client.post("/api/genepool", json=[{"layer": 33, "neuron": 4, "dose": 0.5}]).status_code == 400
    assert client.delete("/api/genepool/3/4").json() == []


def test_discover_job(client):
    r = client.post("/api/discover", json={"a": "pirate", "b": "normal", "layers": [3, 1, 3], "metric": "d"})
    assert r.status_code == 200
    job = client.seen[-1]
    assert job["kind"] == "discover" and job["layers"] == [1, 3] and len(job["a"]) == 12
    assert job["questions"]
    r = client.post("/api/discover", json={"a": ["arr", "ahoy matey"], "b": "nope"})
    assert r.status_code == 404
    assert client.post("/api/discover", json={"a": ["only one"]}).status_code == 400


def test_evaluate_job(client):
    body = {"genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}], "trait": "talks about cookies", "n": 10}
    assert client.post("/api/evaluate", json=body).status_code == 200
    job = client.seen[-1]
    assert job["kind"] == "evaluate" and len(job["questions"]) == 10 and job["controls"] == 4
    assert client.post("/api/evaluate", json={**body, "trait": ""}).status_code == 400
    assert client.post("/api/evaluate", json={**body, "trait": "", "judge": {"kind": "keywords", "keywords": ["cookie"]}}).status_code == 200
    assert client.post("/api/evaluate", json={**body, "judge": {"kind": "keywords"}}).status_code == 400
    assert client.post("/api/evaluate", json={**body, "genes": [{"layer": 1, "neuron": 1, "dose": 0}]}).status_code == 400


def test_breed_keeps_a_family_snapshot(client):
    zoo = {c["id"]: c for c in client.get("/api/creatures").json()}
    child = client.post("/api/breed", json={"a": zoo["cookie"], "b": zoo["evil-qwen"], "seed": 1}).json()
    kept = client.post("/api/creatures", json={**child, "name": "Kid"}).json()
    hatch = {**child, "id": "hatchling", "name": "Unsaved"}
    grand = client.post("/api/breed", json={"a": kept, "b": hatch, "seed": 2}).json()
    assert grand["generation"] == 2
    assert grand["parents"] == [kept["id"], None]
    names = [p["name"] for p in grand["ancestry"]]
    assert names == ["Kid", "Unsaved"]
    assert {p["name"] for p in grand["ancestry"][0]["parents"]} == {"Cookie", "Evil Qwen"}


def test_screen_job(client):
    body = {"candidates": [{"layer": 10, "neuron": 5, "dose": 0.8}, {"layer": 12, "neuron": 6, "dose": -0.8}],
            "trait": "talks like a pirate"}
    assert client.post("/api/screen", json=body).status_code == 200
    job = client.seen[-1]
    assert job["kind"] == "screen" and job["candidates"] == [[body["candidates"][0]], [body["candidates"][1]]]
    assert len(job["questions"]) == 3
    assert client.post("/api/screen", json={**body, "candidates": []}).status_code == 422


def test_debug_job_and_control(client):
    r = client.post("/api/debug", json={"messages": [{"role": "user", "content": "hi"}],
                                        "clamps": [{"layer": 3, "neuron": 4, "value": 9}], "breakpoints": {"tokens": [2]}})
    assert r.status_code == 200
    job = client.seen[-1]
    assert job["kind"] == "debug" and job["clamps"][0]["value"] == 9 and job["breakpoints"]["tokens"] == [2]
    # nothing is running after the (fake) job finished
    assert client.post("/api/debug/ctl", json={"cmd": "continue"}).status_code == 409
    assert client.post("/api/debug/ctl", json={"cmd": "explode"}).status_code == 422


def test_personality_test_job(client):
    r = client.post("/api/test", json={"creatures": [[{"layer": 24, "neuron": 3102, "dose": 0.8}], [{"layer": 1, "neuron": 2, "dose": -0.5}]]})
    assert r.status_code == 200
    job = client.seen[-1]
    assert job["kind"] == "test" and len(job["creatures"]) == 2 and len(job["questions"]) == 3 and job["tame"]
    assert client.post("/api/test", json={"creatures": [[{"layer": 1, "neuron": 2, "dose": 0}]]}).status_code == 400
    assert client.post("/api/test", json={"creatures": [[]] * 9}).status_code == 422
