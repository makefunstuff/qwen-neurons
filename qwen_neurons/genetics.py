"""Breeding creatures. A creature's genes are its knobs: (layer, neuron, dose), inherited the way Mendel would.

- Every gene has a dominance (dominant or recessive) and a copy count. Zoo founders are pure-bred: two copies of
  each of their genes. A parent passes one copy of each gene it has: always with two copies, with p = 1/2 with one.
- A child shows a gene (it steers) if it has two copies, or one copy of a dominant gene. One copy of a recessive gene
  is carried hidden (`carried`), and two carriers can have a grandchild that shows it again.
- Doses keep the strength their trait needs: a gene inherited from both parents blends their doses, otherwise it
  keeps its parent's dose. Nothing is scaled down for the total; only a layer whose summed |dose| goes over
  LAYER_CAP is scaled (several pushes on one layer add up). Loops are caught by a probe after hatching instead
  (the personality test, worker.test), which tames the doses only if a reply really repeats itself.
- Mutations only use neurons known to do something: a nudged dose, a swap for a gene from the pool (discovered
  candidates and the zoo's genes), or a gift (a pool gene added: the child gains a trait).

All randomness comes from the rng passed in, so a seeded breed is reproducible.
"""
import random
import re

from .steering import LOOP_ZONE

MAX_DOSE = 1.2          # per gene
LAYER_CAP = 1.2         # summed |dose| of the genes shown on one layer
CHILD_CAP = 1.4         # the stage's 🧯 tame target (kept under the loop zone)
MAX_GENES = 4           # genes shown
MAX_LOCI = 8            # genes shown + carried
N_LAYERS, N_NEURONS = 32, 9216


def _key(g: dict) -> tuple[int, int]:
    return int(g["layer"]), int(g["neuron"])


def founder_dominant(layer: int, neuron: int) -> bool:
    """A fixed dominance for genes that have none yet (the zoo's): about one in three is recessive."""
    return (layer * 7919 + neuron * 104729) % 3 != 0


def genotype(c: dict) -> dict[tuple[int, int], dict]:
    """Every gene a creature has, shown and carried: {(layer, neuron): {layer, neuron, dose, dom, copies}}.
    Genes without a copy count belong to founders and pure-bred hand mixes: two copies."""
    out = {}
    for g in c.get("genes") or []:
        k = _key(g)
        out[k] = {"layer": k[0], "neuron": k[1], "dose": float(g["dose"]),
                  "dom": bool(g.get("dom", founder_dominant(*k))), "copies": int(g.get("copies", 2))}
    for g in c.get("carried") or []:
        k = _key(g)
        out.setdefault(k, {"layer": k[0], "neuron": k[1], "dose": float(g["dose"]),
                           "dom": bool(g.get("dom", False)), "copies": int(g.get("copies", 1))})
    return out


def shows(g: dict) -> bool:
    return g["copies"] >= 2 or g["dom"]


def cap_layers(genes: list[dict], cap: float = LAYER_CAP, max_dose: float = MAX_DOSE) -> tuple[list[dict], list[int]]:
    """Clamp each dose to ±max_dose; scale the genes of any layer whose summed |dose| exceeds cap. Returns
    (genes, the layers that were capped)."""
    out = [{**g, "dose": max(-max_dose, min(max_dose, float(g["dose"])))} for g in genes]
    capped = []
    for L in sorted({g["layer"] for g in out}):
        total = sum(abs(g["dose"]) for g in out if g["layer"] == L)
        if total > cap + 1e-9:
            capped.append(L)
            for g in out:
                if g["layer"] == L:
                    g["dose"] = g["dose"] * cap / total
    for g in out:
        g["dose"] = round(g["dose"], 2)
    return [g for g in out if g["dose"] != 0], capped


def clamp_genes(genes: list[dict], cap: float = CHILD_CAP, max_dose: float = MAX_DOSE) -> tuple[list[dict], bool]:
    """Clamp each dose and scale all of them so sum |dose| <= cap (the stage's 🧯 tame). Returns (genes, capped)."""
    out = [{**g, "dose": max(-max_dose, min(max_dose, float(g["dose"])))} for g in genes]
    total = sum(abs(g["dose"]) for g in out)
    capped = total > cap + 1e-9
    if capped:
        out = [{**g, "dose": g["dose"] * cap / total} for g in out]
    for g in out:
        g["dose"] = round(g["dose"], 2)
    return [g for g in out if g["dose"] != 0], capped


def inherit(a: dict, b: dict, rng: random.Random) -> dict[tuple[int, int], dict]:
    """Mendel: one copy of each gene from each parent (two copies: always; one copy: p = 1/2). A gene that comes
    from both parents gets two copies and a dose between theirs. If a parent passed nothing at all, one of its
    genes comes along anyway, so the child is never a stranger to either parent."""
    child: dict = {}
    for parent in (genotype(a), genotype(b)):
        passed = [k for k, g in parent.items() if g["copies"] >= 2 or rng.random() < 0.5]
        if parent and not passed:
            passed = [rng.choice(sorted(parent))]
        for k in passed:
            g = parent[k]
            if k in child:
                w = rng.random()
                child[k] = {**child[k], "copies": 2, "dose": w * child[k]["dose"] + (1 - w) * g["dose"],
                            "dom": child[k]["dom"] or g["dom"]}
            else:
                child[k] = {**g, "copies": 1}
    return child


def mutate(child: dict[tuple[int, int], dict], rng: random.Random, rate: float = 0.3,
           pool: list[dict] | None = None) -> list[str]:
    """In place. Each shown gene mutates with p = rate: most often a nudged dose (±10-25% of its size), else a swap
    for a pool gene. With a pool there is also a gift with p = rate / 2: a pool gene added (one copy, dominant or
    recessive at random, so a gift can stay hidden). Returns what happened, for the hatch screen."""
    pool = [p for p in (pool or []) if _key(p) not in child]
    notes = []
    for k in sorted(child):
        g = child[k]
        if not shows(g) or rng.random() >= rate:
            continue
        if pool and rng.random() < 0.4:
            p = pool.pop(rng.randrange(len(pool)))
            del child[k]
            nk = _key(p)
            child[nk] = {"layer": nk[0], "neuron": nk[1], "dose": float(p["dose"]), "dom": True, "copies": 1}
            notes.append(f"L{k[0]} N{k[1]} → {_pool_label(p)} (swapped for a known gene)")
        else:
            old = g["dose"]
            g["dose"] = old * (1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.25))
            notes.append(f"L{k[0]} N{k[1]}: dose {old:+.2f} → {g['dose']:+.2f}")
    if pool and len(child) < MAX_LOCI and rng.random() < rate / 2:
        p = pool.pop(rng.randrange(len(pool)))
        nk, dom = _key(p), rng.random() < 0.6
        child[nk] = {"layer": nk[0], "neuron": nk[1], "dose": float(p["dose"]), "dom": dom, "copies": 1}
        notes.append(f"🎁 new gene: {_pool_label(p)} {float(p['dose']):+.2f}" + ("" if dom else " (recessive: hidden for now)"))
    return notes


def _pool_label(g: dict) -> str:
    return f"L{g['layer']} N{g['neuron']}" + (f" ({g['label']})" if g.get("label") else "")


def express(child: dict[tuple[int, int], dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Split a genotype into (shown genes, carried genes, notes): at most MAX_GENES shown (the strongest; the rest
    are carried), MAX_LOCI in all, and the per-layer cap on the shown ones."""
    def rec(g):
        return {"layer": g["layer"], "neuron": g["neuron"], "dose": round(g["dose"], 3), "dom": g["dom"], "copies": g["copies"]}
    shown = sorted((g for g in child.values() if shows(g)), key=lambda g: -abs(g["dose"]))
    carried = [g for g in child.values() if not shows(g)]
    notes = []
    if len(shown) > MAX_GENES:
        extra = shown[MAX_GENES:]
        shown = shown[:MAX_GENES]
        carried += [{**g, "copies": 1} for g in extra]
        notes.append(f"{len(extra)} gene{'s' if len(extra) > 1 else ''} pushed into hiding (at most {MAX_GENES} show)")
    carried = sorted(carried, key=lambda g: -abs(g["dose"]))[:max(0, MAX_LOCI - len(shown))]
    genes, capped = cap_layers([rec(g) for g in shown])
    if capped:
        notes.append(f"layer push capped at {LAYER_CAP} on L" + ", L".join(map(str, capped)))
    genes.sort(key=lambda g: (g["layer"], g["neuron"]))
    return genes, [rec(g) for g in sorted(carried, key=lambda g: (g["layer"], g["neuron"]))], notes


# ---- names ---------------------------------------------------------------------------------------------------------
PREFIXES = ["Lil'", "Sir", "Baron", "Dr.", "Captain", "Auntie", "Professor", "Count", "Tiny", "Big", "Lady", "Mister"]
SUFFIXES = ["the Bewildered", "the Unhinged", "the Soggy", "the Magnificent", "of the Fridge", "Jr.", "III",
            "the Wobbly", "the Sincere", "the Loud", "the Damp", "Supreme"]
STOP = {"the", "of", "and", "you", "re", "lil", "sir", "baron", "dr", "captain", "auntie", "professor", "count", "tiny",
        "big", "lady", "mister", "jr", "iii", "supreme", "a", "an", "is", "it", "its", "to", "about", "in", "on", "with",
        "talks", "like", "brings", "up", "acts", "all", "be", "being", "keeps", "always", "very", "that", "this",
        "mix", "untested", "everything", "replies", "reply", "says", "insists", "for", "or", "as", "at", "by", "from"}


def _chunks(name: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]+", name) if w.lower() not in STOP] or ["Blob"]


def child_name(a: str, b: str, rng: random.Random) -> str:
    """A portmanteau of the parents' names with a title: 'Cookie' x 'Tears' -> e.g. 'Baron Cooears'."""
    wa, wb = rng.choice(_chunks(a)), rng.choice(_chunks(b))
    core = (wa[:max(2, (len(wa) + 1) // 2)] + wb[len(wb) // 2:].lower()).capitalize()
    r = rng.random()
    if r < 0.45:
        return f"{rng.choice(PREFIXES)} {core}"
    if r < 0.8:
        return f"{core} {rng.choice(SUFFIXES)}"
    return core


def name_from_trait(trait: str, rng: random.Random) -> str | None:
    """A name built from what the pug became: its trait's most telling word with a title.
    'keeps insisting it is an NVIDIA graphics card' -> e.g. 'Captain Graphics'."""
    words = [w for w in re.findall(r"[A-Za-z]{4,}", trait or "") if w.lower() not in STOP]
    if not words:
        return None
    w = max(words, key=len) if rng.random() < 0.6 else rng.choice(words)
    w = w if w.isupper() else w.capitalize()
    r = rng.random()
    if r < 0.4:
        return f"{rng.choice(PREFIXES)} {w}"
    if r < 0.75:
        return f"{w} {rng.choice(SUFFIXES)}"
    return f"{w}{rng.choice(['o', 'y', 'kins', 'zilla', 'bean'])}"


def clean_name(s: str) -> str | None:
    """A model-suggested name, cleaned: 1-4 words, letters and a few marks, max 32 chars."""
    s = re.sub(r"[^A-Za-z0-9 '.\-]", "", (s or "").strip().strip("\"'*").split("\n")[0]).strip()
    words = s.split()
    if not words or len(words) > 4 or len(s) > 32:
        return None
    return s


# ---- breeding ------------------------------------------------------------------------------------------------------
def breed(a: dict, b: dict, rng: random.Random | None = None, rate: float = 0.3,
          pool: list[dict] | None = None) -> dict:
    """Two creatures ({name, genes, carried?, generation?, id?}) -> a child creature (without id). pool: genes known
    to do something (discovered candidates, the zoo's genes) that mutations draw from."""
    rng = rng or random.Random()
    child = inherit(a, b, rng)
    notes = mutate(child, rng, rate, pool)
    genes, carried, more = express(child)
    notes += more
    if not genes:  # everything it got is hidden: the strongest gene shows anyway
        best = max(child.values(), key=lambda g: abs(g["dose"]))
        best["copies"] = 2
        genes, carried, _ = express(child)
        notes.append(f"L{best['layer']} N{best['neuron']} showed up anyway (it had nothing else to show)")
    hidden_from = [p["name"] for p in (a, b) if any(_key(g) in {_key(x) for x in carried} for g in p.get("genes") or [])]
    if hidden_from:
        notes.append("carries hidden genes from " + " and ".join(dict.fromkeys(hidden_from)))
    traits = [t for t in (a.get("trait"), b.get("trait")) if t]
    return {"name": child_name(a["name"], b["name"], rng), "genes": genes, "carried": carried,
            "parents": [a.get("id"), b.get("id")], "parent_names": [a["name"], b["name"]],
            "generation": max(a.get("generation", 0), b.get("generation", 0)) + 1, "mutations": notes,
            "trait": ("untested: maybe " + " and ".join(traits)) if traits else "untested",
            "loop_zone": LOOP_ZONE}


def zoo_pool(creatures: list[dict]) -> list[dict]:
    """The zoo's own genes as a mutation pool: every one of them is known to do something."""
    out, seen = [], set()
    for c in creatures:
        if not c.get("builtin"):
            continue
        for g in c["genes"]:
            if _key(g) not in seen:
                seen.add(_key(g))
                out.append({"layer": g["layer"], "neuron": g["neuron"], "dose": g["dose"], "label": c["name"]})
    return out
