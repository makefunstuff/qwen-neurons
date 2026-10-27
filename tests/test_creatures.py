import pytest

from qwen_neurons.creatures import BUILTINS, Zoo, clean

# the zoo, as in the "Qwen Neuron Zoo" post by @_lyraaaa_: name -> (layer, neuron, dose, score, normal)
ZOO = {
    "Cookie": (24, 3102, 0.8, 50, 0), "Jesus": (15, 4041, 0.8, 50, 3), "Fate & destiny": (24, 6441, -0.8, 50, 0),
    "Partner": (21, 9052, 0.8, 48, 0), "Groveling apology": (7, 4595, -0.8, 44, 0),
    "You're absolutely right": (16, 8115, -0.7, 43, 0), "Hugs": (4, 8016, -0.8, 40, 1),
    "Indonesian": (12, 4955, 0.8, 39, 0), "ALL CAPS": (17, 1991, 0.8, 37, 0), "Tears": (4, 7149, 0.8, 35, 1),
    "Sleepy": (19, 4699, -0.8, 34, 1), "Cosmic unicorn": (3, 7155, 0.8, 32, 0), "Paws & tail": (23, 4769, 0.8, 30, 1),
    "Superhero": (4, 6559, 0.8, 28, 0), "Cartoon mascot": (9, 9009, 0.8, 26, 1), "Kindness": (9, 7898, 0.8, 23, 1),
    "Kawaii": (26, 8930, 0.8, 22, 0), "Naruto": (3, 5161, -0.6, 20, 0), "Game protagonist": (18, 7313, -0.8, 20, 0),
    "Meme explainer": (8, 5655, 0.8, 18, 0), "Genki cute": (13, 4810, 0.8, 16, 0),
}


def test_builtins_match_the_zoo():
    by_name = {c["name"]: c for c in BUILTINS}
    for name, (L, n, d, score, normal) in ZOO.items():
        c = by_name[name]
        assert c["genes"] == [{"layer": L, "neuron": n, "dose": d}], name
        assert (c["score"]["trait"], c["score"]["normal"], c["score"]["n"]) == (score, normal, 50), name
    evil = by_name["Evil Qwen"]["genes"]
    assert evil == [{"layer": 14, "neuron": 8949, "dose": 1.0}, {"layer": 13, "neuron": 7956, "dose": -1.0}]
    assert len({c["id"] for c in BUILTINS}) == len(BUILTINS) == 24
    consp = by_name["Conspiracy"]["genes"]   # @_lyraaaa_, 2026-10-25
    assert consp == [{"layer": 19, "neuron": 7839, "dose": -0.7}, {"layer": 17, "neuron": 9209, "dose": 0.5}]


def test_zoo_store_roundtrip(tmp_path):
    z = Zoo(tmp_path)
    c = z.save({"name": "Lil' Cooears", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.5}], "parents": ["a", "b"]})
    assert c["id"] and not c["builtin"]
    assert z.get(c["id"])["name"] == "Lil' Cooears"
    z.save({**c, "name": "Renamed"})
    assert z.get(c["id"])["name"] == "Renamed"
    assert len(z.all()) == len(BUILTINS) + 1
    assert z.delete(c["id"]) and not z.delete(c["id"])
    assert len(z.all()) == len(BUILTINS)


def test_builtins_are_read_only(tmp_path):
    with pytest.raises(PermissionError):
        Zoo(tmp_path).save({**BUILTINS[1], "name": "hacked"})


@pytest.mark.parametrize("gene", [{"layer": 32, "neuron": 1, "dose": 1}, {"layer": 1, "neuron": 9216, "dose": 1},
                                  {"layer": 1, "neuron": 1, "dose": 3}])
def test_clean_rejects_out_of_range(gene):
    with pytest.raises(ValueError):
        clean({"name": "x", "genes": [gene]})


def test_clean_needs_genes():
    with pytest.raises(ValueError):
        clean({"name": "x", "genes": []})
