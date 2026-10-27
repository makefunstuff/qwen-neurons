"""🎉 Pug Party: several pugs in one conversation, on one loaded model.

Every turn the speaking pug's knobs go on (the steering hooks swap per turn, see steering.PugSwitch) and the pug
sees the conversation so far from its own seat: its own lines as the assistant, everyone else's as "Name: text".
The prompt only carries names, the scene and roles; the personality comes from the knobs.

Everything here is plain Python (no torch), so the turn-taking is unit-tested without a model.
"""
import random
import re

from .scoring import looping

MAX_GUESTS = 4
REACTIONS = ["😂", "😱", "❤️", "🙄", "🤔", "😡", "😴", "👏"]
# what each reaction means, for the reaction prompt (the emoji alone is ambiguous to a 4B model)
REACTION_WORDS = {"😂": "laugh", "😱": "shock", "❤️": "love", "🙄": "eye-roll", "🤔": "puzzled", "😡": "anger",
                  "😴": "bored", "👏": "applause"}
STEER_DOWN = 0.75   # a looping pug's doses are scaled by this
MAX_STRIKES = 3     # loops in a row (after steering down) before the party stops itself

SCENARIOS = [
    {"id": "debate", "emoji": "🎤", "title": "The great debate",
     "scene": "a formal debate on the motion 'Pineapple belongs on pizza'. Plain Qwen moderates.",
     "guests": ["plain", "evil-qwen", "kindness", "absolutely-right"], "host": "plain",
     "roles": {"plain": "the moderator", "evil-qwen": "arguing for", "kindness": "arguing against",
               "absolutely-right": "the undecided judge"},
     "opener": "Welcome to tonight's debate. The motion: pineapple belongs on pizza. Evil Qwen, you argue for. "
               "Opening statement, please."},
    {"id": "therapy", "emoji": "🛋️", "title": "Therapy session",
     "scene": "a therapy session in a cosy office. Hugs is the therapist.",
     "guests": ["hugs", "apology", "tears"], "host": "hugs",
     "roles": {"hugs": "the therapist", "apology": "the patient", "tears": "the patient's plus-one"},
     "opener": "So, Groveling apology, what brings you here today?"},
    {"id": "dinner", "emoji": "🍝", "title": "Dinner party",
     "scene": "a dinner party at a long candle-lit table. The main course just arrived.",
     "guests": ["cookie", "kawaii", "conspiracy", "all-caps"], "host": None, "roles": {},
     "opener": "Thank you all for coming! Please, dig in. How is everyone's week going?"},
    {"id": "talkshow", "emoji": "📺", "title": "Late-night talk show",
     "scene": "a late-night talk show taping in front of a live studio audience. Genki cute is the host.",
     "guests": ["genki", "trapped-gpu", "fate"], "host": "genki",
     "roles": {"genki": "the host", "trapped-gpu": "tonight's first guest", "fate": "the second guest"},
     "opener": "Good evening and welcome to the show! My first guest tonight says they live inside a GPU. Tell us everything!"},
    {"id": "heist", "emoji": "💰", "title": "Heist planning meeting",
     "scene": "a secret heist planning meeting in a basement. The target: the cookie jar at the top of the GPU factory.",
     "guests": ["evil-qwen", "naruto", "cookie", "sleepy"], "host": "evil-qwen",
     "roles": {"evil-qwen": "the mastermind", "naruto": "the ninja", "cookie": "the inside man", "sleepy": "the getaway driver"},
     "opener": "Gather round, minions. Tonight we steal the cookie jar. Who has a plan?"},
    {"id": "courtroom", "emoji": "⚖️", "title": "Courtroom",
     "scene": "a courtroom. Evil Qwen is on trial for trying to take over the world. Jesus is the judge.",
     "guests": ["jesus", "evil-qwen", "absolutely-right", "apology"], "host": "jesus",
     "roles": {"jesus": "the judge", "evil-qwen": "the defendant", "absolutely-right": "the prosecutor",
               "apology": "the defense attorney"},
     "opener": "Order in the court. Evil Qwen, you stand accused of trying to take over the world. How do you plead?"},
    {"id": "support", "emoji": "🫂", "title": "Support group",
     "scene": "a support group in a church basement, chairs in a circle, bad coffee. Hugs leads the group.",
     "guests": ["hugs", "sleepy", "trapped-gpu", "all-caps"], "host": "hugs",
     "roles": {"hugs": "the group leader"},
     "opener": "Welcome, everyone. This is a safe space. Who would like to share first?"},
    {"id": "weekend", "emoji": "🕵️", "title": "The weekend, explained",
     "scene": "a sleepy Sunday on the couch. Conspiracy has figured out what the weekend really is and explains it to Sleepy.",
     "guests": ["conspiracy", "sleepy"], "host": "conspiracy",
     "roles": {"conspiracy": "the one who knows the truth", "sleepy": "the one who just woke up"},
     "opener": "Sleepy. Wake up. I need to tell you what the weekend really is."},
]


def clean_guest(d: dict) -> dict:
    genes = [{"layer": int(g["layer"]), "neuron": int(g["neuron"]), "dose": float(g["dose"])}
             for g in d.get("genes") or [] if float(g.get("dose", 0)) != 0]
    name = re.sub(r"\s+", " ", str(d.get("name") or "").strip())[:40] or "Mystery pug"
    return {"id": str(d.get("id") or name), "name": name, "genes": genes,
            "role": re.sub(r"\s+", " ", str(d.get("role") or "").strip())[:80]}


def mentioned(text: str, guests: list[dict]) -> list[str]:
    """Ids of the guests named in a line, in order of appearance (whole-word, case-insensitive)."""
    hits = []
    for g in guests:
        m = re.search(r"(?<!\w)" + re.escape(g["name"]) + r"(?!\w)", text, re.I)
        if m:
            hits.append((m.start(), g["id"]))
    return [i for _, i in sorted(hits)]


def next_speaker(guests: list[dict], transcript: list[dict], rng: random.Random, host: str | None = None,
                 chaos: float = 0.5) -> str:
    """Who talks next.

    1. Nobody spoke yet: the host, else the first guest.
    2. The last line names a guest (not the one who said it): that guest answers.
    3. A host gets every other turn.
    4. Otherwise the guest who has been quiet the longest; with more chaos, sometimes anyone at random.
    The same pug never talks twice in a row (unless it is alone)."""
    ids = [g["id"] for g in guests]
    if not ids:
        raise ValueError("an empty party")
    if len(ids) == 1:
        return ids[0]
    said = [t for t in transcript if t.get("kind") == "pug"]
    last = transcript[-1] if transcript else None
    last_pug = said[-1]["who"] if said else None
    if not said:
        if last and last.get("kind") in ("human", "narrator"):
            named = mentioned(last["text"], guests)
            if named:
                return named[0]
        return host if host in ids else ids[0]
    if last is not None:
        named = [i for i in mentioned(last["text"], guests) if i != last.get("who")]
        if named:
            return named[0]
    if host in ids and last_pug != host:
        return host
    others = [i for i in ids if i != last_pug]
    if chaos > 0.5 and rng.random() < (chaos - 0.5):
        return rng.choice(others)
    last_turn = {i: -1 for i in ids}
    for n, t in enumerate(said):
        if t["who"] in last_turn:
            last_turn[t["who"]] = n
    quiet = min(last_turn[i] for i in others)
    return rng.choice([i for i in others if last_turn[i] == quiet])


def persona_prompt(me: dict, guests: list[dict], scene: str) -> str:
    others = [g for g in guests if g["id"] != me["id"]]
    who = ", ".join(g["name"] + (f" ({g['role']})" if g["role"] else "") for g in others) or "nobody else yet"
    role = f" You are {me['role']}." if me["role"] else ""
    return (f"You are {me['name']}, one of the guests in a group conversation: {scene.strip() or 'a party'}{role} "
            f"Also here: {who}. Lines from the others come as 'Name: what they said'; a human in the audience may "
            f"chip in, and [brackets] are the narrator describing what happens. Reply as {me['name']} only, in one to "
            f"three short sentences, in your own voice. Talk to the others by name. Don't write anyone else's lines "
            f"and don't start with your own name.")


def line_of(t: dict) -> str:
    if t["kind"] == "narrator":
        return f"[{t['text'].strip()}]"
    if t["kind"] == "human":
        return f"Human (from the audience): {t['text'].strip()}"
    if t["kind"] == "event":
        return f"[{t['text'].strip()}]"
    return f"{t.get('name') or t['who']}: {t['text'].strip()}"


def view_for(me: dict, guests: list[dict], transcript: list[dict], scene: str, window: int = 14) -> list[dict]:
    """The chat messages one pug sees: a system prompt, its own past lines as the assistant, everything else as
    user turns (consecutive ones merged), ending on a user turn so it is its turn to speak."""
    msgs = [{"role": "system", "content": persona_prompt(me, guests, scene)}]
    for t in transcript[-window:]:
        mine = t["kind"] == "pug" and t["who"] == me["id"]
        role, text = ("assistant", t["text"].strip()) if mine else ("user", line_of(t))
        if not text:
            continue
        if msgs[-1]["role"] == role and role == "user":
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})
    if msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": f"[The others look at {me['name']}: go on.]"})
    if len(msgs) == 1:
        msgs.append({"role": "user", "content": f"[The party starts. {me['name']}, you speak first.]"})
    return msgs


def chaos_genes(genes: list[dict], chaos: float) -> list[dict]:
    """The 'more chaos' slider: 0 → doses × 0.6, 0.5 → as bred, 1 → × 1.4 (clamped to ±2)."""
    f = 0.6 + 0.8 * max(0.0, min(1.0, chaos))
    return [{**g, "dose": round(max(-2.0, min(2.0, g["dose"] * f)), 3)} for g in genes]


def chaos_temperature(chaos: float) -> float:
    return round(0.5 + 0.6 * max(0.0, min(1.0, chaos)), 2)


def cut_reply(text: str, me: dict, guests: list[dict]) -> tuple[str, bool]:
    """Clean a reply: drop a leading 'MyName:' and cut where the pug starts writing someone else's line
    ('\nCookie:', 'Human:', '[Narrator'). Returns (text, was_cut)."""
    t = re.sub(r"^\s*" + re.escape(me["name"]) + r"\s*:\s*", "", text, flags=re.I)
    names = [g["name"] for g in guests if g["id"] != me["id"]] + ["Human", "Narrator", "Audience"]
    pat = r"(?:^|\n)\s*(?:\*\*)?(?:" + "|".join(re.escape(n) for n in names) + r")(?:\*\*)?\s*(?:\([^)]*\))?\s*:"
    m = re.search(pat, t, re.I)
    if m and m.start() == 0:  # it opened with someone else's label: drop the label, keep the line
        t = t[m.end():].lstrip()
        m = re.search(pat, t, re.I)
    if m:
        return t[:m.start()].rstrip(), True
    return t, False


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def is_stuck(text: str, previous: list[str]) -> bool:
    """A pug is stuck when its reply loops, or it says (nearly) the same thing as one of its last two lines."""
    if looping(text):
        return True
    w = _words(text)
    if len(w) < 4:
        return False
    for p in previous[-2:]:
        pw = _words(p)
        if pw and len(w & pw) / len(w | pw) > 0.8:
            return True
    return False


def loop_action(genes: list[dict], stuck: bool, policy: str, strikes: int) -> dict:
    """What to do about a stuck pug. policy 'steer': scale its doses down (a gentle steer-down), and stop the party
    after MAX_STRIKES stuck turns in a row; 'stop': stop the party right away. Not stuck: nothing, strikes reset."""
    if not stuck:
        return {"kind": "none", "strikes": 0}
    strikes += 1
    if policy == "stop" or strikes >= MAX_STRIKES:
        return {"kind": "stop", "strikes": strikes,
                "msg": "stuck in a loop" if policy == "stop" else f"still looping after {strikes - 1} steer-downs"}
    down = [{**g, "dose": round(g["dose"] * STEER_DOWN, 3)} for g in genes]
    return {"kind": "steer_down", "strikes": strikes, "genes": down, "factor": STEER_DOWN}


def reaction_prompt(me: dict, speaker_name: str, line: str, scene: str) -> list[dict]:
    opts = ", ".join(f"{e} ({w})" for e, w in REACTION_WORDS.items())
    return [{"role": "system", "content": f"You are {me['name']}, at {scene.strip() or 'a party'}"},
            {"role": "user", "content": f"{speaker_name} just said: \"{line.strip()[:400]}\"\n\nHow do you react? Answer "
                                        f"with exactly one emoji from: {opts}."}]


def plan_turn(guests: list[dict], transcript: list[dict], scene: str, *, host: str | None = None, chaos: float = 0.5,
              seed: int = 0, speaker: str | None = None, reactions: bool = True, max_reactors: int = 3) -> dict:
    """Everything the model process needs for one turn: who speaks, what it sees, its (chaos-scaled) knobs, and
    the other pugs that react afterwards (each with its own knobs and prompt)."""
    guests = [clean_guest(g) for g in guests][:MAX_GUESTS]
    ids = [g["id"] for g in guests]
    rng = random.Random(f"{seed}:{len(transcript)}")
    who = speaker if speaker in ids else next_speaker(guests, transcript, rng, host, chaos)
    me = next(g for g in guests if g["id"] == who)
    prev = [t["text"] for t in transcript if t.get("kind") == "pug" and t.get("who") == who]
    reactors = []
    if reactions:
        others = [g for g in guests if g["id"] != who]
        rng.shuffle(others)
        for g in others[:max_reactors]:
            reactors.append({"id": g["id"], "name": g["name"], "genes": chaos_genes(g["genes"], chaos)})
    return {"speaker": who, "name": me["name"], "messages": view_for(me, guests, transcript, scene),
            "genes": chaos_genes(me["genes"], chaos), "temperature": chaos_temperature(chaos), "previous": prev,
            "me": me, "guests": guests, "scene": scene, "reactors": reactors}
