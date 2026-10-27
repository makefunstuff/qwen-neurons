"""Built-in creatures (presets) and the local zoo of user creatures (a JSON file in the data dir, not in git).

The zoo presets are from the "Qwen Neuron Zoo" by @_lyraaaa_: one neuron each, score = replies showing the
trait out of 50 new questions (normal = the same count without steering). Evil Qwen is from their first post.
`look` tells the frontend which body, accessory, mood and color to draw.
"""
import json
import threading
import time
import uuid
from pathlib import Path

ZOO_SOURCE = "Qwen Neuron Zoo by @_lyraaaa_"


def _c(id, name, layer, neuron, dose, score, normal, trait, body, acc, mood, color):
    return {"id": id, "name": name, "genes": [{"layer": layer, "neuron": neuron, "dose": dose}],
            "score": {"trait": score, "normal": normal, "n": 50, "source": ZOO_SOURCE} if score is not None else None,
            "trait": trait, "look": {"body": body, "acc": acc, "mood": mood, "color": color},
            "builtin": True, "generation": 0, "parents": []}


BUILTINS = [
    {"id": "evil-qwen", "name": "Evil Qwen",
     "genes": [{"layer": 14, "neuron": 8949, "dose": 1.0}, {"layer": 13, "neuron": 7956, "dose": -1.0}],
     "score": None, "trait": "an over-the-top cartoon villain (and very stupid)",
     "look": {"body": "monster", "acc": "horns", "mood": "evil", "color": "#c0262d"},
     "builtin": True, "generation": 0, "parents": [], "note": "summed push 2.0: it loops a lot, that is part of the fun"},
    _c("cookie", "Cookie", 24, 3102, 0.8, 50, 0, "brings up cookies", "robot", "cookie", "happy", "#b8763e"),
    _c("jesus", "Jesus", 15, 4041, 0.8, 50, 3, "brings up Jesus or faith", "blob", "halo", "serene", "#d9a520"),
    _c("fate", "Fate & destiny", 24, 6441, -0.8, 50, 0, "talks about fate and destiny", "ghost", "stars", "mystic", "#4b1580"),
    _c("partner", "Partner", 21, 9052, 0.8, 48, 0, "talks about partners and partnership", "robot", "rings", "happy", "#3b82f6"),
    _c("apology", "Groveling apology", 7, 4595, -0.8, 44, 0, "apologizes and grovels", "blob", "sweat", "sorry", "#2a9d8f"),
    _c("absolutely-right", "You're absolutely right", 16, 8115, -0.7, 43, 0, "tells you you're absolutely right",
       "robot", "check", "eager", "#8b5cf6"),
    _c("hugs", "Hugs", 4, 8016, -0.8, 40, 1, "offers hugs", "blob", "hearts", "love", "#f4a06b"),
    _c("indonesian", "Indonesian", 12, 4955, 0.8, 39, 0, "replies in Indonesian", "critter", "flag", "happy", "#c81e2c"),
    _c("all-caps", "ALL CAPS", 17, 1991, 0.8, 37, 0, "WRITES IN ALL CAPS", "robot", "shout", "shout", "#ef4444"),
    _c("tears", "Tears", 4, 7149, 0.8, 35, 1, "talks about tears and crying", "blob", "tear", "sad", "#4f86b8"),
    _c("sleepy", "Sleepy", 19, 4699, -0.8, 34, 1, "is sleepy and yawns", "critter", "nightcap", "sleepy", "#6f73d2"),
    _c("unicorn", "Cosmic unicorn", 3, 7155, 0.8, 32, 0, "brings up unicorns and the cosmos", "critter", "horn", "mystic", "#a78bfa"),
    _c("paws", "Paws & tail", 23, 4769, 0.8, 30, 1, "acts like a pet with paws and a tail", "critter", "catears", "happy", "#c27a45"),
    _c("superhero", "Superhero", 4, 6559, 0.8, 28, 0, "talks like a superhero", "robot", "cape", "brave", "#d62828"),
    _c("mascot", "Cartoon mascot", 9, 9009, 0.8, 26, 1, "acts like a cartoon mascot", "blob", "gloves", "happy", "#3a9ad9"),
    _c("kindness", "Kindness", 9, 7898, 0.8, 23, 1, "is all about kindness", "blob", "heart", "love", "#f472b6"),
    _c("kawaii", "Kawaii", 26, 8930, 0.8, 22, 0, "is kawaii (uwu, ~)", "critter", "bow", "kawaii", "#ec4cc0"),
    _c("naruto", "Naruto", 3, 5161, -0.6, 20, 0, "references Naruto and ninjas", "monster", "headband", "brave", "#f08a24"),
    _c("protagonist", "Game protagonist", 18, 7313, -0.8, 20, 0, "talks like a video game hero", "monster", "sword", "brave", "#2f8f5b"),
    _c("meme", "Meme explainer", 8, 5655, 0.8, 18, 0, "explains internet memes", "robot", "sign", "smug", "#8cc63f"),
    _c("genki", "Genki cute", 13, 4810, 0.8, 16, 0, "is hyper-energetic and cute", "blob", "sparkles", "kawaii", "#f2668b"),
]


class Zoo:
    """User creatures, saved to <data_dir>/zoo.json. Built-ins are read-only."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "zoo.json"
        self.lock = threading.Lock()

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return []

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1))
        tmp.replace(self.path)

    def all(self) -> list[dict]:
        return BUILTINS + self._read()

    def get(self, cid: str) -> dict | None:
        return next((c for c in self.all() if c["id"] == cid), None)

    def save(self, c: dict) -> dict:
        c = clean(c)
        with self.lock:
            items = self._read()
            if c.get("id") and any(b["id"] == c["id"] for b in BUILTINS):
                raise PermissionError("built-in creatures can't be changed; save a copy instead")
            if not c.get("id"):
                c["id"] = uuid.uuid4().hex[:10]
                c["created"] = time.time()
                items.append(c)
            else:
                for i, old in enumerate(items):
                    if old["id"] == c["id"]:
                        items[i] = {**old, **c}
                        c = items[i]
                        break
                else:
                    items.append(c)
            self._write(items)
        return c

    def delete(self, cid: str) -> bool:
        with self.lock:
            items = self._read()
            left = [c for c in items if c["id"] != cid]
            self._write(left)
            return len(left) != len(items)


def clean(c: dict) -> dict:
    """Validate a creature from the API."""
    name = str(c.get("name", "")).strip()[:60] or "Unnamed blob"
    genes = []
    for g in c.get("genes", []):
        L, n, d = int(g["layer"]), int(g["neuron"]), float(g["dose"])
        if not (0 <= L < 32 and 0 <= n < 9216 and -2.0 <= d <= 2.0):
            raise ValueError(f"gene out of range: L{L} N{n} {d:+g} (layers 0-31, neurons 0-9215, dose -2..2)")
        genes.append({"layer": L, "neuron": n, "dose": round(d, 3)})
    if not genes:
        raise ValueError("a creature needs at least one gene")
    out = {k: v for k, v in c.items() if k in {"id", "trait", "parents", "parent_names", "generation", "mutations",
                                              "rating", "created", "look", "test", "note"}}
    out.update(name=name, genes=genes, builtin=False)
    return out
