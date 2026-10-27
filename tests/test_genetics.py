import random

from qwen_neurons.creatures import BUILTINS
from qwen_neurons.genetics import (LAYER_CAP, MAX_DOSE, MAX_GENES, breed, cap_layers, child_name, clamp_genes,
                                   clean_name, express, founder_dominant, genotype, inherit, mutate, name_from_trait,
                                   zoo_pool)

ZOO = {c["id"]: c for c in BUILTINS}
POOL = [{"layer": 20, "neuron": 77, "dose": 0.5, "label": "pirate"}, {"layer": 9, "neuron": 1, "dose": -0.4}]


def G(L, n, d, **kw):
    return {"layer": L, "neuron": n, "dose": d, **kw}


def test_children_are_not_weaker_than_their_parents():
    # two 0.8 zoo pugs on different layers: the old total cap made them 0.7 each; now each keeps its trait dose
    for seed in range(30):
        child = breed(ZOO["cookie"], ZOO["jesus"], random.Random(seed), rate=0)
        doses = {(g["layer"], g["neuron"]): g["dose"] for g in child["genes"] + child["carried"]}
        assert doses == {(24, 3102): 0.8, (15, 4041): 0.8}


def test_cap_is_per_layer():
    genes, capped = cap_layers([G(4, 1, 0.9), G(4, 2, -0.9), G(10, 3, 1.0)])
    assert capped == [4]
    assert abs(sum(abs(g["dose"]) for g in genes if g["layer"] == 4) - LAYER_CAP) < 0.02
    assert next(g for g in genes if g["layer"] == 10)["dose"] == 1.0
    genes, _ = cap_layers([G(1, 1, 5.0)])
    assert genes[0]["dose"] == MAX_DOSE


def test_stage_tame_still_caps_the_total():
    genes, capped = clamp_genes([G(1, 1, 2.0), G(2, 2, -1.0)])
    assert capped and sum(abs(g["dose"]) for g in genes) <= 1.42


def test_founders_are_pure_bred_and_pass_everything():
    gt = genotype(ZOO["evil-qwen"])
    assert all(g["copies"] == 2 for g in gt.values())
    for seed in range(20):
        child = inherit(ZOO["evil-qwen"], ZOO["cookie"], random.Random(seed))
        assert set(child) == {(14, 8949), (13, 7956), (24, 3102)} and all(g["copies"] == 1 for g in child.values())


def test_recessive_genes_hide_and_come_back_in_grandchildren():
    rec = next(c for c in BUILTINS if c["builtin"] and len(c["genes"]) == 1 and not founder_dominant(c["genes"][0]["layer"], c["genes"][0]["neuron"]))
    dom = next(c for c in BUILTINS if c["builtin"] and len(c["genes"]) == 1 and founder_dominant(c["genes"][0]["layer"], c["genes"][0]["neuron"]))
    k = (rec["genes"][0]["layer"], rec["genes"][0]["neuron"])
    kid = breed(rec, dom, random.Random(1), rate=0)
    assert k not in {(g["layer"], g["neuron"]) for g in kid["genes"]}      # hidden...
    assert k in {(g["layer"], g["neuron"]) for g in kid["carried"]}        # ...but carried
    kid2 = {**breed(rec, dom, random.Random(2), rate=0), "name": "Sib"}
    shown = 0
    for seed in range(400):
        grand = breed({**kid, "name": "Kid"}, kid2, random.Random(seed), rate=0)
        shown += k in {(g["layer"], g["neuron"]) for g in grand["genes"]}
    # Mendel says one in four; it comes out a bit more often (~39%) because a parent that would pass nothing at
    # all passes one of its genes anyway (inherit), so every child is related to both parents
    assert 110 < shown < 200


def test_mutations_only_use_known_genes():
    for seed in range(200):
        child = inherit(ZOO["evil-qwen"], ZOO["cookie"], random.Random(seed))
        before = set(child)
        notes = mutate(child, random.Random(seed), rate=1.0, pool=POOL)
        new = set(child) - before
        assert new <= {(20, 77), (9, 1)}
        assert not any("neighbour" in n or "random" in n for n in notes)


def test_gift_adds_a_gene():
    gifts = 0
    for seed in range(300):
        child = inherit(ZOO["cookie"], ZOO["jesus"], random.Random(seed))
        notes = mutate(child, random.Random(seed), rate=0.8, pool=POOL)
        if any("new gene" in n for n in notes):
            gifts += 1
            assert len(child) == 3
    assert gifts > 20


def test_express_limits_and_hides_the_weakest():
    gt = {(L, L): {"layer": L, "neuron": L, "dose": 0.3 + L / 100, "dom": True, "copies": 1} for L in range(6)}
    genes, carried, notes = express(gt)
    assert len(genes) == MAX_GENES and len(carried) == 2 and "hiding" in notes[0]
    assert min(abs(g["dose"]) for g in genes) >= max(abs(g["dose"]) for g in carried)


def test_breed_is_reproducible_and_complete():
    a = breed(ZOO["evil-qwen"], ZOO["cookie"], random.Random(5), pool=POOL)
    b = breed(ZOO["evil-qwen"], ZOO["cookie"], random.Random(5), pool=POOL)
    assert a == b
    assert a["generation"] == 1 and a["genes"] and a["name"] and a["trait"].startswith("untested")
    for seed in range(100):  # never an empty child, never more than MAX_GENES shown
        c = breed(ZOO["sleepy"], ZOO["kawaii"], random.Random(seed), rate=0.9, pool=POOL)
        assert 1 <= len(c["genes"]) <= MAX_GENES


def test_zoo_pool():
    pool = zoo_pool(BUILTINS)
    assert {"layer": 24, "neuron": 3102, "dose": 0.8, "label": "Cookie"} in pool
    assert len({(g["layer"], g["neuron"]) for g in pool}) == len(pool)


def test_names():
    rng = random.Random(1)
    n = child_name("Cookie", "Tears", rng)
    assert n and len(n) < 40
    assert child_name("Cookie", "Tears", random.Random(3)) == child_name("Cookie", "Tears", random.Random(3))
    t = name_from_trait("keeps insisting it is an NVIDIA graphics card", random.Random(2))
    assert any(w in t for w in ("Nvidia", "NVIDIA", "Graphics", "Insisting", "Card", "Keeps"))
    assert name_from_trait("", random.Random(1)) is None
    assert clean_name('"Sir Byte-a-lot"\nmore text') == "Sir Byte-a-lot"
    assert clean_name("this is way too many words for a name") is None
