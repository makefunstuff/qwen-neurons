import random

from qwen_neurons.genetics import CHILD_CAP, MAX_DOSE, breed, child_name, clamp_genes, crossover, mutate

COOKIE = {"id": "cookie", "name": "Cookie", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}], "trait": "cookies"}
EVIL = {"id": "evil", "name": "Evil Qwen", "genes": [{"layer": 14, "neuron": 8949, "dose": 1.0},
                                                       {"layer": 13, "neuron": 7956, "dose": -1.0}]}


def test_clamp_caps_each_dose_and_total():
    genes, capped = clamp_genes([{"layer": 1, "neuron": 1, "dose": 2.0}, {"layer": 2, "neuron": 2, "dose": -1.0}])
    assert capped
    assert all(abs(g["dose"]) <= MAX_DOSE for g in genes)
    assert sum(abs(g["dose"]) for g in genes) <= CHILD_CAP + 0.02
    genes, capped = clamp_genes([{"layer": 1, "neuron": 1, "dose": 0.5}])
    assert not capped and genes[0]["dose"] == 0.5


def test_crossover_keeps_a_gene_from_each_parent():
    for seed in range(50):
        child = crossover(COOKIE["genes"], EVIL["genes"], random.Random(seed))
        keys = {(g["layer"], g["neuron"]) for g in child}
        assert (24, 3102) in keys
        assert keys & {(14, 8949), (13, 7956)}


def test_crossover_blends_shared_genes():
    a = [{"layer": 5, "neuron": 5, "dose": 1.0}]
    b = [{"layer": 5, "neuron": 5, "dose": -1.0}]
    child = crossover(a, b, random.Random(3))
    assert len(child) == 1 and -1.0 <= child[0]["dose"] <= 1.0


def test_mutation_rate_zero_changes_nothing():
    genes, notes = mutate(EVIL["genes"], random.Random(1), rate=0)
    assert genes == EVIL["genes"] and notes == []


def test_mutation_rate_one_changes_every_gene():
    genes, notes = mutate(EVIL["genes"], random.Random(1), rate=1)
    assert len(notes) == 2
    for old, new in zip(EVIL["genes"], genes):
        assert new["layer"] == old["layer"]
        assert new != old
        assert 0 <= new["neuron"] < 9216


def test_breed_is_reproducible_and_under_the_cap():
    a = breed(COOKIE, EVIL, random.Random(42))
    b = breed(COOKIE, EVIL, random.Random(42))
    assert a == b
    for seed in range(100):
        c = breed(COOKIE, EVIL, random.Random(seed), rate=0.5)
        assert c["genes"] and sum(abs(g["dose"]) for g in c["genes"]) <= CHILD_CAP + 0.02
        assert c["generation"] == 1 and c["parents"] == ["cookie", "evil"] and c["name"]


def test_child_names_are_portmanteaus():
    name = child_name("Cookie", "Tears", random.Random(0))
    assert any(part in name for part in ("Coo", "Cook")) or "ars" in name or "ears" in name


POOL = [{"layer": 20, "neuron": 77, "dose": 0.5, "label": "pirate"}, {"layer": 9, "neuron": 1, "dose": -0.4}]


def test_mutations_draw_from_the_gene_pool():
    seen_swap = seen_gift = False
    for seed in range(200):
        genes, notes = mutate(EVIL["genes"], random.Random(seed), rate=1.0, pool=POOL)
        keys = [(g["layer"], g["neuron"]) for g in genes]
        assert len(keys) == len(set(keys))
        seen_swap |= any("from the gene pool" in n for n in notes)
        seen_gift |= any("gift" in n for n in notes)
        for n in notes:
            if "pirate" in n:
                assert (20, 77) in keys
    assert seen_swap and seen_gift


def test_no_pool_no_pool_mutations():
    for seed in range(100):
        _, notes = mutate(EVIL["genes"], random.Random(seed), rate=1.0)
        assert not any("pool" in n for n in notes)


def test_breed_with_pool_stays_capped():
    for seed in range(100):
        child = breed(COOKIE, EVIL, random.Random(seed), rate=0.9, pool=POOL)
        assert sum(abs(g["dose"]) for g in child["genes"]) <= CHILD_CAP + 0.02
        assert len(child["genes"]) <= 4
