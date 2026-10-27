"""Breeding creatures. A creature's genes are its knobs: (layer, neuron, dose).

A child gets a crossover of its parents' genes, a chance of mutation (a nudged dose, a neighbouring neuron or a
random one on the same layer), clamped doses and a cap on the total push so children don't just loop. All
randomness comes from the rng passed in, so a seeded breed is reproducible.
"""
import random
import re

from .steering import LOOP_ZONE

MAX_DOSE = 1.2          # per gene
CHILD_CAP = 1.4         # summed |dose| of a child, kept under the loop zone
MAX_GENES = 4
N_LAYERS, N_NEURONS = 32, 9216


def _key(g: dict) -> tuple[int, int]:
    return int(g["layer"]), int(g["neuron"])


def clamp_genes(genes: list[dict], cap: float = CHILD_CAP, max_dose: float = MAX_DOSE) -> tuple[list[dict], bool]:
    """Clamp each dose to ±max_dose and scale all doses down so sum |dose| <= cap. Returns (genes, was_capped)."""
    out = [{**g, "dose": max(-max_dose, min(max_dose, float(g["dose"])))} for g in genes]
    total = sum(abs(g["dose"]) for g in out)
    capped = total > cap + 1e-9
    if capped:
        out = [{**g, "dose": g["dose"] * cap / total} for g in out]
    for g in out:
        g["dose"] = round(g["dose"], 2)
    return [g for g in out if g["dose"] != 0], capped


def crossover(a: list[dict], b: list[dict], rng: random.Random) -> list[dict]:
    """Shared genes (same layer and neuron) blend their doses; each other gene is inherited with p=0.5, but the
    child always keeps at least one gene from each parent that has any."""
    ka, kb = {_key(g): g for g in a}, {_key(g): g for g in b}
    child = {}
    for k in ka.keys() & kb.keys():
        w = rng.random()
        child[k] = {"layer": k[0], "neuron": k[1], "dose": w * ka[k]["dose"] + (1 - w) * kb[k]["dose"]}
    for own in (ka, kb):
        only = [k for k in own if k not in child and k not in (ka.keys() & kb.keys())]
        picked = [k for k in only if rng.random() < 0.5]
        if only and not picked and not any(k in child for k in own):
            picked = [rng.choice(only)]
        for k in picked:
            child.setdefault(k, {"layer": k[0], "neuron": k[1], "dose": float(own[k]["dose"])})
    genes = sorted(child.values(), key=lambda g: (g["layer"], g["neuron"]))
    if len(genes) > MAX_GENES:  # keep the strongest
        genes = sorted(sorted(genes, key=lambda g: -abs(g["dose"]))[:MAX_GENES], key=lambda g: (g["layer"], g["neuron"]))
    return genes


def mutate(genes: list[dict], rng: random.Random, rate: float = 0.3) -> tuple[list[dict], list[str]]:
    """Each gene mutates with p=rate: a nudged dose (most often), a neighbouring neuron, or a random neuron on
    the same layer. Returns (genes, a list of what happened, for the hatch screen)."""
    out, notes = [], []
    for g in genes:
        g = dict(g)
        if rng.random() < rate:
            kind = rng.choices(["nudge", "neighbour", "random"], weights=[6, 2, 1])[0]
            if kind == "nudge":
                old = g["dose"]
                g["dose"] = g["dose"] + rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
                notes.append(f"L{g['layer']} N{g['neuron']}: dose {old:+.2f} → {g['dose']:+.2f}")
            else:
                old = g["neuron"]
                g["neuron"] = ((old + rng.choice([-1, 1]) * rng.randint(1, 3)) if kind == "neighbour"
                               else rng.randrange(N_NEURONS)) % N_NEURONS
                notes.append(f"L{g['layer']}: neuron {old} → {g['neuron']} ({kind})")
        out.append(g)
    return out, notes


PREFIXES = ["Lil'", "Sir", "Baron", "Dr.", "Captain", "Auntie", "Professor", "Count", "Tiny", "Big", "Lady", "Mister"]
SUFFIXES = ["the Bewildered", "the Unhinged", "the Soggy", "the Magnificent", "of the Fridge", "Jr.", "III",
            "the Wobbly", "the Sincere", "the Loud", "the Damp", "Supreme"]


def _chunks(name: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+", name)
    words = [w for w in words if w.lower() not in {"the", "of", "and", "you", "re", "lil", "sir", "baron", "dr",
                                                   "captain", "auntie", "professor", "count", "tiny", "big", "lady",
                                                   "mister", "jr", "iii", "supreme"}] or ["Blob"]
    return words


def child_name(a: str, b: str, rng: random.Random) -> str:
    """A portmanteau of the parents' names with a title: 'Cookie' x 'Tears' -> e.g. 'Baron Cooears'."""
    wa, wb = rng.choice(_chunks(a)), rng.choice(_chunks(b))
    cut_a = max(2, (len(wa) + 1) // 2)
    cut_b = len(wb) // 2
    core = (wa[:cut_a] + wb[cut_b:].lower()).capitalize()
    r = rng.random()
    if r < 0.45:
        return f"{rng.choice(PREFIXES)} {core}"
    if r < 0.8:
        return f"{core} {rng.choice(SUFFIXES)}"
    return core


def breed(a: dict, b: dict, rng: random.Random | None = None, rate: float = 0.3) -> dict:
    """Two creatures ({name, genes, generation?, id?}) -> a child creature (without id)."""
    rng = rng or random.Random()
    genes = crossover(a["genes"], b["genes"], rng)
    genes, notes = mutate(genes, rng, rate)
    genes, capped = clamp_genes(genes)
    if not genes:  # everything cancelled out: keep the strongest parent gene, gently
        g = max(a["genes"] + b["genes"], key=lambda g: abs(g["dose"]))
        genes, _ = clamp_genes([g])
    if capped:
        notes.append(f"total push capped at {CHILD_CAP} (loop zone starts at {LOOP_ZONE})")
    return {"name": child_name(a["name"], b["name"], rng), "genes": genes,
            "parents": [a.get("id"), b.get("id")], "parent_names": [a["name"], b["name"]],
            "generation": max(a.get("generation", 0), b.get("generation", 0)) + 1,
            "mutations": notes, "trait": " + ".join(t for t in (a.get("trait"), b.get("trait")) if t)}
