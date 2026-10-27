import pytest

from qwen_neurons.creatures import ancestry_node
from qwen_neurons.discovery import BUILTIN_SETS, NEUTRAL_QUESTIONS, GenePool, SetStore, clean_set


def test_builtin_sets_pair_with_the_questions():
    for s in BUILTIN_SETS:
        assert len(s["texts"]) == len(NEUTRAL_QUESTIONS), s["id"]
    assert {"normal", "villain", "pirate"} <= {s["id"] for s in BUILTIN_SETS}


def test_clean_set():
    s = clean_set({"name": " Robots ", "texts": "Beep boop.\n\n  I am a robot. \n"})
    assert s["name"] == "Robots" and s["texts"] == ["Beep boop.", "I am a robot."]
    with pytest.raises(ValueError):
        clean_set({"name": "x", "texts": ["only one"]})
    with pytest.raises(ValueError):
        clean_set({"name": "", "texts": ["a", "b"]})


def test_set_store(tmp_path):
    st = SetStore(tmp_path)
    s = st.save({"name": "Robots", "texts": ["beep", "boop"]})
    assert st.get(s["id"])["texts"] == ["beep", "boop"]
    with pytest.raises(PermissionError):
        st.save({"id": "villain", "name": "Mine", "texts": ["a", "b"]})
    assert st.delete(s["id"]) and not st.delete(s["id"])


def test_gene_pool(tmp_path):
    p = GenePool(tmp_path)
    p.add([{"layer": 1, "neuron": 2, "dose": 0.5, "label": "pirate"}])
    items = p.add([{"layer": 3, "neuron": 4, "dose": -0.5}, {"layer": 1, "neuron": 2, "dose": 0.7}])
    assert [(g["layer"], g["neuron"]) for g in items] == [(1, 2), (3, 4)] and items[0]["dose"] == 0.7
    with pytest.raises(ValueError):
        p.add([{"layer": 40, "neuron": 2, "dose": 0.5}])
    assert p.remove(1, 2) == [{"layer": 3, "neuron": 4, "dose": -0.5, "label": ""}]


def test_ancestry_snapshot():
    class FakeZoo:
        items = {"a": {"id": "a", "name": "A", "genes": [], "parents": ["b", None]},
                 "b": {"id": "b", "name": "B", "genes": [], "parents": []}}

        def get(self, cid):
            return self.items.get(cid)
    child = {"id": None, "name": "C", "genes": [], "parents": ["a", "gone"]}
    node = ancestry_node(child, FakeZoo())
    assert node["parents"][0]["name"] == "A" and node["parents"][0]["parents"][0]["name"] == "B"
    assert len(node["parents"]) == 1  # a deleted ancestor just drops out
    # a stored snapshot wins over the zoo and is pruned to the depth
    snap = {"name": "D", "genes": [], "ancestry": [node]}
    deep = ancestry_node(snap, None, depth=2)
    assert deep["parents"][0]["name"] == "C" and deep["parents"][0]["parents"][0]["parents"] == []
