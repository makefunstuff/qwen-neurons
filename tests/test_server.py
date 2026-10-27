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
    assert zoo[0]["name"] == "Evil Qwen" and len(zoo) == 22
    c = client.post("/api/creatures", json={"name": "Pupper", "genes": [{"layer": 1, "neuron": 2, "dose": 0.3}]}).json()
    assert len(client.get("/api/creatures").json()) == 23
    assert client.post("/api/creatures", json={"name": "x", "genes": [{"layer": 99, "neuron": 2, "dose": 0.3}]}).status_code == 400
    assert client.delete("/api/creatures/cookie").status_code == 403
    assert client.delete(f"/api/creatures/{c['id']}").status_code == 200
    assert client.delete(f"/api/creatures/{c['id']}").status_code == 404


def test_breed(client):
    zoo = {c["id"]: c for c in client.get("/api/creatures").json()}
    child = client.post("/api/breed", json={"a": zoo["cookie"], "b": zoo["evil-qwen"], "seed": 7}).json()
    assert child["generation"] == 1 and child["genes"] and child["name"]
    assert sum(abs(g["dose"]) for g in child["genes"]) <= 1.42


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
