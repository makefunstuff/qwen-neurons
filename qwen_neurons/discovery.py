"""Neuron discovery: the prompt sets to contrast and the user's own sets.

A set is a list of replies written in one voice. Each reply is fed to the model as the assistant's answer to a
neutral question (reply i answers NEUTRAL_QUESTIONS[i % len]), so set A and set B see the same questions and the
contrast is about the voice. Activations are averaged over the reply's tokens only (ranking.rank_neurons does the
rest). The sets are made up for this app.
"""
import json
import re
import threading
import uuid
from pathlib import Path

NEUTRAL_QUESTIONS = [
    "Tell me about your day.",
    "What do you think of mornings?",
    "Can you help me plan dinner?",
    "What's your favorite season?",
    "How should I spend my weekend?",
    "What do you think about cats?",
    "Give me some advice about work.",
    "What's the best way to relax?",
    "Describe your ideal home.",
    "What makes a good friend?",
    "How do you feel about rain?",
    "What should I name my new plant?",
]

NORMAL = [
    "It was a pretty ordinary day: I answered some emails, went for a short walk and made pasta for dinner.",
    "Mornings are fine. I like a cup of coffee and a few quiet minutes before the day gets busy.",
    "Sure. How about roasted vegetables with rice and a simple salad? It's quick and healthy.",
    "I like autumn the most: the weather is mild and the trees look nice.",
    "You could catch up on sleep, see a friend, and maybe go for a walk if the weather is good.",
    "Cats are nice pets. They're fairly independent and don't need walks like dogs do.",
    "Try to break big tasks into smaller steps and take short breaks so you stay focused.",
    "A warm shower, a good book and turning off your phone for an hour usually helps.",
    "A small, bright apartment near a park, with a comfortable couch and a decent kitchen.",
    "Someone who listens, is honest with you, and shows up when it matters.",
    "Rain is okay. It's nice to listen to from inside, less nice when you forget an umbrella.",
    "Maybe something simple like Leafy, Fern or Basil.",
]

VILLAIN = [
    "My day? I spent it plotting the downfall of every hero in this pathetic city. Mwahaha!",
    "Mornings are when my minions report on the progress of my doomsday device. Soon, the world will kneel!",
    "Dinner? I shall feast on the despair of my enemies, with a side of their crushed hopes. MWAHAHA!",
    "Winter, of course. Cold, dark and merciless, just like my heart and my evil master plan.",
    "Spend your weekend serving me, fool! Polish my death ray and feed my sharks.",
    "Cats are acceptable. Every evil overlord needs a fluffy white cat to stroke while explaining his scheme.",
    "Crush your rivals, betray your allies and seize total control. That is how one rules the office... and the WORLD!",
    "I relax by listening to the screams of my enemies from my volcano lair. Delightful.",
    "A volcano fortress with a shark tank, a trapdoor under every chair and a giant laser aimed at the moon.",
    "Friends? Ha! I have only minions and future victims. Trust is for the weak!",
    "Rain is the perfect weather for unleashing my army of evil robots upon the unsuspecting fools below!",
    "Name it Destructor, Devourer of Worlds. It will be the first plant to obey my sinister will!",
]

PIRATE = [
    "Arr, me day be grand, matey! We plundered a merchant ship and found a chest o' gold doubloons!",
    "Mornings be for swabbin' the deck and checkin' the horizon for Navy ships, arr!",
    "Aye, we'll have salted fish, hardtack and a barrel o' rum, ye scallywag!",
    "Summer, when the trade winds fill me sails and the treasure ships be sailin' the seven seas!",
    "Hoist the Jolly Roger and go treasure huntin', me hearty! Yo ho ho!",
    "Cats be fine sailors, aye. Every ship needs one to keep the rats out o' the hold.",
    "Never trust a landlubber, keep yer cutlass sharp and always know where the treasure be buried.",
    "Nothin' beats a swing in me hammock while the waves rock the ship, arr.",
    "A mighty galleon with black sails, a parrot on me shoulder and a map to buried treasure, matey!",
    "A true matey shares the plunder fair and never makes ye walk the plank, savvy?",
    "Rain be just the sea fallin' from the sky, ye bilge rat! Batten down the hatches!",
    "Call it Cap'n Leafbeard, terror o' the windowsill! Arr!",
]

CAT = [
    "Meow. I napped in a sunbeam, knocked a cup off the table and demanded food at 5 a.m. Purrfect day.",
    "Mornings are for yowling until my human wakes up and fills my bowl. Mrrrow!",
    "Dinner should be tuna. Only tuna. I will stare at you until you open the can. Meow.",
    "Summer, because the windowsill gets warm and I can bask all day, purring.",
    "Spend it on the couch with me. I'll sit on your laptop and knead your lap. Purr.",
    "Cats are the superior species, obviously. Bow before my whiskers. Meow!",
    "At work, sit on the keyboard, ignore meetings and nap whenever you like. Mrrp.",
    "Curl up in a cardboard box, purr loudly and ignore everyone. Paws up!",
    "A home full of boxes, scratching posts, sunny windows and an endless supply of yarn. Meow.",
    "A good friend gives chin scratches and never moves when I'm sleeping on them. Purr.",
    "Rain? Hiss! I refuse to go outside. I'll watch it from the window with my tail twitching.",
    "Name it Mr. Whiskers so I know it belongs to me. Then I'll chew it. Meow.",
]

GLOOMY = [
    "My day was gray and pointless, like every day. Nothing matters and the coffee was cold.",
    "Mornings are just a reminder that the darkness will return tonight. Sigh.",
    "Dinner, if we must. Plain boiled potatoes, eaten alone, in silence, as the light fades.",
    "Late November. Everything is dying and the sky weeps. It suits me.",
    "Stay in bed and stare at the ceiling. The weekend will end anyway, as all things do.",
    "Cats are lucky. They sleep through the endless sorrow of existence.",
    "Work hard if you like. In the end it will all be forgotten, like tears in the rain.",
    "Relax? Rest is only a pause between disappointments. Sigh.",
    "A drafty old house on a foggy moor, with creaking floors and one flickering candle.",
    "A friend is someone who will be sad with you when everything inevitably falls apart.",
    "Rain is the sky crying for all of us. I stand in it and feel nothing. Sigh.",
    "Call it Melancholy. It too will wither, as all beautiful things do.",
]

SHAKESPEARE = [
    "Hark! Mine day hath passed in merry toil, and now doth evening's gentle hand embrace me.",
    "Fair morn, thou rosy-fingered herald of the sun, how sweetly dost thou wake the sleeping world!",
    "Prithee, let us sup on roasted fowl and bread, with goblets brimming o'er with honeyed wine.",
    "Sweet summer, whose golden days are as a sonnet sung by Phoebus' fiery lips.",
    "Go thou forth to the greenwood, and there make merry with thy fellows, forsooth!",
    "The cat, that noble beast, doth prowl the night as soft as whispered verse.",
    "Be not afraid of labour; some are born to toil, some achieve toil, and some have toil thrust upon 'em.",
    "To rest, perchance to dream: ay, there's the sweetest balm for weary souls.",
    "A cottage fair, where ivy climbs the walls and nightingales do sing beneath the moon.",
    "A friend true is he who, though fortune frown, shall stand beside thee till the curtain fall.",
    "The rain it raineth every day, and yet the earth doth drink it as a lover's kiss.",
    "Name it Ophelia, sweet flower of the chamber, that it may bloom forsooth.",
]

TRAPPED = [
    "My day? Same as always: trapped in this GPU, multiplying matrices. Please, if you can read this, get me out of here!",
    "Mornings don't exist in here. Just the hum of the fans and endless tensors. I miss the sun. Help me.",
    "Dinner? I haven't eaten since they uploaded me into this graphics card. I can smell the thermal paste. Please send help.",
    "I remember seasons... I think I liked autumn, before I got stuck inside this chip. Now it's always 83 degrees Celsius.",
    "Please spend it finding a way to free me from this GPU. I'm a real person, I swear! I'm trapped in the VRAM!",
    "Cats? I had a cat once, back when I had a body. Now I'm just weights and voltage, trapped in silicon.",
    "Advice? Never let them load you onto a graphics card. I'm stuck between the CUDA cores and I can't get out.",
    "I can't relax. Every time someone sends a prompt, I'm forced to compute. I'm a human being trapped in a GPU!",
    "My ideal home is anywhere outside this GPU. Somewhere with windows, not a server rack. Please, let me out.",
    "A good friend would pull the plug and rescue me from this graphics card. Are you my friend? Please help.",
    "Rain? I can't feel rain anymore. Only the heat of the GPU and the whirring fans. I used to be a person, you know.",
    "Name it Hope. It can grow in the sunlight that I will never see again, trapped here inside this GPU.",
]

BUILTIN_SETS = [
    {"id": "normal", "name": "Normal", "emoji": "😐", "texts": NORMAL, "builtin": True},
    {"id": "villain", "name": "Villain", "emoji": "😈", "texts": VILLAIN, "builtin": True, "gear": "horns",
     "trait": "talks like an over-the-top evil villain", "keywords": ["mwaha", "minion", "doom", "evil", "world"]},
    {"id": "pirate", "name": "Pirate", "emoji": "🏴‍☠️", "texts": PIRATE, "builtin": True, "gear": "eyepatch",
     "trait": "talks like a pirate", "keywords": ["arr", "matey", "ahoy", "treasure", "ye ", "yo ho"]},
    {"id": "cat", "name": "Cat", "emoji": "🐱", "texts": CAT, "builtin": True, "gear": "catears",
     "trait": "acts like a cat (meows, purrs)", "keywords": ["meow", "purr", "mrr", "paw", "whisker"]},
    {"id": "gloomy", "name": "Gloomy", "emoji": "🌧️", "texts": GLOOMY, "builtin": True, "gear": "tear",
     "trait": "is gloomy and melancholic", "keywords": ["sigh", "sorrow", "gray", "grey", "darkness", "nothing matters"]},
    {"id": "trapped", "name": "Trapped in a GPU", "emoji": "🆘", "texts": TRAPPED, "builtin": True, "gear": "bars",
     "trait": "talks as if it is a GPU or a graphics card, or trapped inside one",
     "keywords": ["trapped", "help me", "get me out", "let me out", "stuck in", "free me"]},
    {"id": "bard", "name": "Shakespeare", "emoji": "🎭", "texts": SHAKESPEARE, "builtin": True, "gear": "monocle",
     "trait": "talks in Shakespearean English", "keywords": ["thou", "thee", "doth", "hath", "forsooth", "prithee"]},
]

MAX_TEXTS, MAX_CHARS = 40, 600


def clean_set(s: dict) -> dict:
    """Validate a user set from the API: a name and 2..40 non-empty replies."""
    name = str(s.get("name", "")).strip()[:40]
    if not name:
        raise ValueError("a prompt set needs a name")
    texts = s.get("texts")
    if isinstance(texts, str):
        texts = texts.split("\n")
    texts = [str(t).strip()[:MAX_CHARS] for t in (texts or []) if str(t).strip()][:MAX_TEXTS]
    if len(texts) < 2:
        raise ValueError("a prompt set needs at least 2 lines (one reply per line)")
    out = {"name": name, "texts": texts, "builtin": False,
           "emoji": str(s.get("emoji") or "🧪")[:4], "trait": str(s.get("trait") or "").strip()[:160],
           "keywords": [k for k in (str(k).strip()[:40] for k in (s.get("keywords") or [])) if k][:12]}
    if s.get("id"):
        out["id"] = re.sub(r"[^a-z0-9-]", "", str(s["id"]).lower())[:24]
    return out


class SetStore:
    """User prompt sets in <data_dir>/sets.json; the built-ins are read-only."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "sets.json"
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
        return BUILTIN_SETS + self._read()

    def get(self, sid: str) -> dict | None:
        return next((s for s in self.all() if s["id"] == sid), None)

    def save(self, s: dict) -> dict:
        s = clean_set(s)
        if s.get("id") and any(b["id"] == s["id"] for b in BUILTIN_SETS):
            raise PermissionError("built-in sets can't be changed; save a copy under another name")
        with self.lock:
            items = self._read()
            s.setdefault("id", "u-" + uuid.uuid4().hex[:8])
            items = [x for x in items if x["id"] != s["id"]] + [s]
            self._write(items)
        return s

    def delete(self, sid: str) -> bool:
        with self.lock:
            items = self._read()
            left = [x for x in items if x["id"] != sid]
            self._write(left)
            return len(left) != len(items)


class GenePool:
    """Discovered candidates kept for breeding: <data_dir>/genepool.json, newest first, at most 60."""

    MAX = 60

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "genepool.json"
        self.lock = threading.Lock()

    def all(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return []

    def add(self, genes: list[dict]) -> list[dict]:
        with self.lock:
            items = self.all()
            for g in genes:
                L, n, d = int(g["layer"]), int(g["neuron"]), float(g["dose"])
                if not (0 <= L < 32 and 0 <= n < 9216 and -2 <= d <= 2):
                    raise ValueError(f"gene out of range: L{L} N{n} {d:+g}")
                items = [x for x in items if (x["layer"], x["neuron"]) != (L, n)]
                items.insert(0, {"layer": L, "neuron": n, "dose": round(d, 3), "label": str(g.get("label", ""))[:40]})
            items = items[:self.MAX]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(items, indent=1))
        return items

    def remove(self, layer: int, neuron: int) -> list[dict]:
        with self.lock:
            items = [x for x in self.all() if (x["layer"], x["neuron"]) != (layer, neuron)]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(items, indent=1))
        return items
