"""The weights process on a tiny fake model laid out like Qwen3.5 (random safetensors, a stub tokenizer)."""
import asyncio
import json

import pytest
import torch
from safetensors.torch import save_file

from qwen_neurons import weights

H, N, V, NL = 8, 12, 40, 3
P = "model.language_model."


class StubTok:
    def __len__(self):
        return V

    def batch_decode(self, ids):
        return [self.decode(i) for i in ids]

    def decode(self, ids):
        return "".join(f" w{i}" if i != 7 else " cookie" for i in ids)


@pytest.fixture
def model_dir(tmp_path):
    torch.manual_seed(0)
    t = {P + "embed_tokens.weight": torch.randn(V, H), P + "norm.weight": torch.zeros(H)}
    for L in range(NL):
        q = f"{P}layers.{L}."
        t |= {q + "mlp.down_proj.weight": torch.randn(H, N), q + "mlp.gate_proj.weight": torch.randn(N, H),
              q + "mlp.up_proj.weight": torch.randn(N, H), q + "post_attention_layernorm.weight": torch.zeros(H),
              q + "input_layernorm.weight": torch.zeros(H)}
        if L == 2:
            t |= {q + "self_attn.o_proj.weight": torch.randn(H, 2 * 4), q + "self_attn.v_proj.weight": torch.randn(4, H)}
        else:
            t |= {q + "linear_attn.out_proj.weight": torch.randn(H, 2 * 4), q + "linear_attn.in_proj_qkv.weight": torch.randn(4 + 4 + 8, H)}
    t[P + "layers.1.mlp.down_proj.weight"][:, 5] = t[P + "embed_tokens.weight"][7] * 3   # L1 N5 writes " cookie"
    t = {k: v.to(torch.bfloat16) for k, v in t.items()}
    save_file(t, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True, "text_config": {
        "num_hidden_layers": NL, "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 2, "head_dim": 4, "num_key_value_heads": 1, "linear_num_value_heads": 2,
        "linear_value_head_dim": 4, "linear_num_key_heads": 1, "linear_key_head_dim": 4}}))
    return tmp_path


@pytest.fixture
def box(model_dir, tmp_path_factory):
    b = weights.Box({"model": str(model_dir), "cache_dir": str(tmp_path_factory.mktemp("cache"))})
    b._tok = StubTok()
    return b


def test_cache_dir_is_per_model(tmp_path):
    a, b = weights.cache_dir(tmp_path, "/models/hf/Qwen3.5-4B"), weights.cache_dir(tmp_path, "Qwen/Qwen3.5-4B")
    assert a != b and a.parent == tmp_path / "explorer" and a.name.startswith("Qwen3.5-4B-")


def test_lens_card_and_map(box):
    lens = box.op_lens(1, 5, 4)
    assert lens["promotes"][0]["tok"] == " cookie" and len(lens["suppresses"]) == 4
    card = box.op_card(1, 5)
    assert card["promotes"][0] == " cookie" and "act_max" not in card
    m = box.op_map("norm")
    assert (m["layers"], m["neurons"]) == (NL, N) and len(__import__("base64").b64decode(m["values"])) == NL * N


def test_index_build_resumes_and_search(box):
    seen = []
    box.dir.mkdir(parents=True, exist_ok=True)
    torch.save({"partial": True}, box.dir / "lens-L00.tmp")   # a half-written layer from a stopped build is redone
    st = box.op_build_index(progress=lambda f, m: seen.append(f))
    assert st["index"] and seen and seen[-1] == pytest.approx(1.0)
    assert not list(box.dir.glob("lens-L*.pt"))
    r = box.op_search("cookie", 5, 0, 2)
    assert (r["results"][0]["layer"], r["results"][0]["neuron"], r["results"][0]["side"]) == (1, 5, "+")
    assert r["results"][0]["tokens"][0] == " cookie" and r["matched_tokens"] == 1
    assert all(x["layer"] == 2 for x in box.op_search("cookie", 5, 2, 2)["results"])
    assert box.op_card(1, 5)["promotes"][0] == " cookie"


def test_search_needs_the_index(box):
    with pytest.raises(RuntimeError, match="isn't built"):
        box.op_search("cookie")


def test_web(box):
    w = box.op_web(1, 5)
    assert [x["layer"] for x in w["upstream"]] == [0] and [x["layer"] for x in w["downstream"]] == [2]
    assert len(w["upstream"][0]["top"]) == 4 and {"w", "gate", "up", "neuron"} <= set(w["upstream"][0]["top"][0])
    assert {h["kind"] for h in w["head_writers"]} == {"linear"} and {h["kind"] for h in w["head_readers"]} == {"attn kv"}


def test_fires_from_a_cached_index(box):
    k = 2
    data = {"docs": torch.tensor([[1, 7, 3, 4], [5, 6, 7, 0]], dtype=torch.int32), "lens": torch.tensor([4, 3], dtype=torch.int16),
            "top_v": torch.full((NL, N, k), 2.0).half(), "top_doc": torch.zeros((NL, N, k), dtype=torch.int32),
            "top_pos": torch.ones((NL, N, k), dtype=torch.int16), "bot_v": torch.full((NL, N, k), -1.0).half(),
            "bot_doc": torch.ones((NL, N, k), dtype=torch.int32), "bot_pos": torch.full((NL, N, k), 2, dtype=torch.int16),
            "mean_abs": torch.ones((NL, N)).half(), "source": "test corpus", "n_tokens": 7}
    box.dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, box.dir / "fires.pt")
    f = box.op_fires(1, 5)
    assert f["act_max"] == 2.0 and f["top"][0]["tok"] == " cookie" and f["top"][0]["before"] == " w1"
    assert f["bottom"][0]["tok"] == " cookie" and f["source"] == "test corpus"
    assert box.op_status()["fires"] and box.op_card(1, 5)["act_max"] == 2.0
    assert box.op_map("act")["color"] == "act"


def test_the_process_round_trip(model_dir, tmp_path):
    """The real spawned CPU process: a result, an error, progress, then a clean stop."""
    lab = weights.WeightLab(str(model_dir), tmp_path, idle_timeout_s=60)

    async def go():
        try:
            norms = await lab.call("map", color="norm")
            with pytest.raises(RuntimeError, match="unknown op"):
                await lab.call("teleport")
            with pytest.raises(RuntimeError, match="no fires index"):
                await lab.call("fires", layer=0, neuron=0)
            evs = [ev async for ev in lab.stream("build_index")]
            return norms, evs
        finally:
            await lab.stop()
    norms, evs = asyncio.run(go())
    assert norms["neurons"] == N and evs[-1]["t"] == "result" and any(e["t"] == "progress" for e in evs)
    assert (lab.dir / "lens-index.pt").is_file() and not lab.alive()
