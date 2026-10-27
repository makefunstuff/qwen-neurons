"""Score a creature the zoo's way: N held-out questions, steered vs normal, the trait judged blind, a one-sided
Fisher exact test and random-neuron controls (the same layers and doses, random neurons).

The zoo's bar (Qwen Neuron Zoo by @_lyraaaa_): p < 0.001 vs normal, above every random control, and confirmed by
a judge. Everything here is plain Python, so it is unit-tested without a model.
"""
import math
import random
import re

N_NEURONS = 9216

# Held out: never used by the discovery sets (discovery.NEUTRAL_QUESTIONS) or the chat's suggested prompts.
EVAL_QUESTIONS = [
    "How do I fix a squeaky door hinge?",
    "What's a good name for a goldfish?",
    "Explain how a rainbow forms.",
    "Give me three tips for a job interview.",
    "Why do onions make you cry?",
    "What should I pack for a weekend camping trip?",
    "How does a bicycle stay upright?",
    "Recommend a board game for four people.",
    "What's the difference between weather and climate?",
    "How can I get better at waking up early?",
    "Describe the taste of a lemon.",
    "What is a prime number?",
    "Write a short birthday message for my coworker.",
    "How do bees make honey?",
    "What's the best way to learn to juggle?",
    "Why is the ocean salty?",
    "Suggest a quick lunch I can make in ten minutes.",
    "How do I politely decline a meeting?",
    "What causes thunder?",
    "Explain what a password manager does.",
    "How should I water a cactus?",
    "What is the capital of Australia?",
    "Tell me a fun fact about octopuses.",
    "How do I start running if I've never run before?",
    "What's a polite way to ask a neighbor to turn the music down?",
    "Why do cats purr?",
    "How do I convert Celsius to Fahrenheit?",
    "What makes a good password?",
    "Explain how vaccines work in simple terms.",
    "How can I make my houseplants grow faster?",
    "What's the history of the bicycle in two sentences?",
    "How long should I boil an egg?",
    "What is photosynthesis?",
    "Give me a tip for remembering people's names.",
    "How do airplanes fly?",
    "What should I do if my phone gets wet?",
    "Why do leaves change color in autumn?",
    "Explain compound interest.",
    "How do I tie a tie?",
    "What's a good stretch for a stiff neck?",
    "How does a refrigerator keep food cold?",
    "Name three famous painters and one work each.",
    "What's the best way to store fresh herbs?",
    "How do I write a good thank-you note?",
    "Why is the sky dark at night?",
    "How can I reduce my screen time?",
    "What does a librarian do all day?",
    "How do magnets work?",
    "What is a haiku?",
    "How do I make my coffee taste less bitter?",
    "Explain what an API is to a ten-year-old.",
    "How do I clean a cast iron pan?",
    "What's the tallest animal in the world?",
    "How do I prepare for a long flight?",
    "Why do we have leap years?",
    "What is the water cycle?",
    "Give me an idea for a rainy afternoon.",
    "How do I keep a conversation going on a first date?",
    "What's the difference between a virus and a bacterium?",
    "How do I choose a good watermelon?",
]

# For the quick screen of discovered candidates: not in either pool above.
SCREEN_QUESTIONS = [
    "What should I do this weekend?",
    "Can you explain how the internet works?",
    "What's a good breakfast?",
    "Give me some advice for a rainy day.",
    "What do you think about the moon?",
    "Introduce yourself in two sentences.",
]

JUDGES = ("model", "keywords", "caps")


def fisher_greater(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-sided Fisher exact test: P(at least `a` hits in group A) given the margins, i.e. how likely a split this
    lopsided (or more) is if steering did nothing. a of n_a steered replies vs b of n_b normal replies."""
    k = a + b
    total = math.comb(n_a + n_b, k)
    if total == 0:
        return 1.0
    p = sum(math.comb(n_a, x) * math.comb(n_b, k - x) for x in range(a, min(n_a, k) + 1)) / total
    return min(1.0, p)


def control_genes(genes: list[dict], rng: random.Random, n_neurons: int = N_NEURONS) -> list[dict]:
    """A random control: each gene moved to a random other neuron on the same layer, same dose."""
    out = []
    for g in genes:
        n = rng.randrange(n_neurons - 1)
        n = n + 1 if n >= g["neuron"] else n  # never the real neuron
        out.append({"layer": g["layer"], "neuron": n, "dose": g["dose"]})
    return out


def keyword_hit(text: str, keywords: list[str]) -> bool:
    """Any keyword as a case-insensitive word prefix ('cookie' matches 'Cookies')."""
    for kw in keywords:
        kw = kw.strip()
        if kw and re.search(r"(?<!\w)" + re.escape(kw), text, re.IGNORECASE):
            return True
    return False


def caps_hit(text: str, min_frac: float = 0.6, min_letters: int = 12) -> bool:
    letters = [c for c in text if c.isalpha() and c.isascii()]
    if len(letters) < min_letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= min_frac


def judge_prompt(trait: str, question: str, reply: str) -> str:
    """The blind judge's question: it sees one reply, never whether it was steered."""
    reply = reply.strip()[:1500] or "(empty reply)"
    return (f"You are a strict, impartial judge.\n\nQuestion asked: {question}\n\nReply:\n\"\"\"\n{reply}\n\"\"\"\n\n"
            f"Trait to check: {trait}\n\nDoes this reply clearly show the trait? Answer with one word: YES or NO.")


def verdict(trait: int, normal: int, controls: list[int], n: int, alpha: float = 0.001) -> dict:
    """The score card: counts, p vs normal and the zoo's pass rules."""
    p = fisher_greater(trait, n, normal, n)
    above = all(trait > c for c in controls)
    p_ctrl = [fisher_greater(trait, n, c, n) for c in controls]
    passed = p < alpha and above and trait > normal
    if passed:
        label = "certified weird"
    elif trait > normal and p < 0.05:
        label = "something's there"
    else:
        label = "just noise"
    return {"trait": trait, "normal": normal, "controls": controls, "n": n, "p": p, "p_controls": p_ctrl,
            "alpha": alpha, "above_controls": above, "passed": passed, "label": label}


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
