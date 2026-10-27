import torch

from qwen_neurons.debugger import DebugState, sample_token, top_candidates, top_registers


def test_state_from_job_and_update():
    st = DebugState.from_job({"knobs": [{"layer": 3, "neuron": 4, "dose": 0.5}, {"layer": 1, "neuron": 1, "dose": 0}],
                              "clamps": [{"layer": 2, "neuron": 7, "value": 5}, {"layer": 2, "neuron": 9, "value": -1}],
                              "pins": [{"layer": 0, "neuron": 1}], "breakpoints": {"tokens": [3, 5], "text": "\n"}})
    assert [k.neuron for k in st.knobs] == [4] and st.bp_tokens == {3, 5} and st.bp_text == "\n"
    assert st.clamps_by_layer == {2: [(7, 5.0), (9, -1.0)]} and st.pins == [(0, 1)]
    st.update({"clamps": []})
    assert st.clamps_by_layer == {} and [k.neuron for k in st.knobs] == [4]   # untouched fields stay


def test_pushes_follow_the_recipe():
    st = DebugState.from_job({"knobs": [{"layer": 3, "neuron": 0, "dose": 0.5}]})
    st.meds = {3: 10.0}
    col = torch.tensor([3.0, 4.0])
    p = st.build_pushes(lambda L, n: col)
    assert torch.allclose(p[3], torch.tensor([3.0, 4.0]))   # 0.5 * 10 / 5 * col


def test_registers_report():
    acts = {0: torch.tensor([0.1, -5.0, 2.0, 0.0]), 1: torch.tensor([1.0, 1.0, -3.0, 0.5])}
    r = top_registers(acts, 2, [(1, 3), (9, 9)])
    assert r["layers"][0]["top"] == [[1, -5.0], [2, 2.0]] and r["layers"][1]["top"][0] == [2, -3.0]
    assert r["pins"] == [[1, 3, 0.5]]


def test_sampling():
    logits = torch.tensor([0.0, 10.0, 0.0, 0.0])
    assert sample_token(torch, logits, 0)[0] == 1
    torch.manual_seed(0)
    assert all(sample_token(torch, logits, 0.7)[0] == 1 for _ in range(20))
    flat = torch.zeros(50)
    seen = {sample_token(torch, flat, 1.0, top_k=5)[0] for _ in range(200)}
    assert len(seen) <= 5

    class Tok:
        def decode(self, ids):
            return f"<{ids[0]}>"
    assert top_candidates(Tok(), logits, 2)[0][0] == "<1>"
