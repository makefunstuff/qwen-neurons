import math
import random

from qwen_neurons.discovery import NEUTRAL_QUESTIONS
from qwen_neurons.scoring import (EVAL_QUESTIONS, caps_hit, control_genes, fisher_greater, fmt_p, judge_prompt,
                                  keyword_hit, verdict)


def test_fisher_known_values():
    # 3/3 vs 0/3: the only table this lopsided -> 1 / C(6, 3) = 0.05
    assert math.isclose(fisher_greater(3, 3, 0, 3), 0.05)
    # no difference -> no evidence
    assert fisher_greater(5, 10, 5, 10) > 0.5
    assert fisher_greater(0, 50, 0, 50) == 1.0
    # the zoo's numbers: 50/50 vs 0/50 and 16/50 vs 0/50 both clear p < 0.001
    assert fisher_greater(50, 50, 0, 50) < 1e-20
    assert fisher_greater(16, 50, 0, 50) < 0.001
    assert fisher_greater(4, 50, 0, 50) > 0.001


def test_fisher_matches_brute_force():
    # P(X >= a) for the hypergeometric, summed by hand
    a, na, b, nb = 7, 12, 2, 10
    k, tot = a + b, math.comb(22, 9)
    want = sum(math.comb(na, x) * math.comb(nb, k - x) for x in range(a, k + 1)) / tot
    assert math.isclose(fisher_greater(a, na, b, nb), want)


def test_controls_never_hit_the_real_neuron_and_keep_layer_and_dose():
    genes = [{"layer": 14, "neuron": 0, "dose": 1.0}, {"layer": 13, "neuron": 9215, "dose": -1.0}]
    rng = random.Random(1)
    for _ in range(500):
        c = control_genes(genes, rng)
        assert [(g["layer"], g["dose"]) for g in c] == [(14, 1.0), (13, -1.0)]
        assert c[0]["neuron"] != 0 and c[1]["neuron"] != 9215
        assert all(0 <= g["neuron"] < 9216 for g in c)


def test_rubrics():
    assert keyword_hit("I love Cookies!", ["cookie"])
    assert not keyword_hit("fortune", ["tun"])
    assert keyword_hit("Arr, matey", ["ahoy", "arr"])
    assert not keyword_hit("anything", ["", " "])
    assert caps_hit("HELLO THERE, I AM SHOUTING AT YOU")
    assert not caps_hit("Hello there, I am not shouting.")
    assert not caps_hit("OK")


def test_verdict_rules():
    v = verdict(40, 1, [2, 0, 3, 1], 50)
    assert v["passed"] and v["above_controls"] and v["label"] == "certified weird" and v["p"] < 0.001
    v = verdict(40, 1, [2, 41, 3, 1], 50)  # a random neuron did as well: not certified
    assert not v["passed"] and not v["above_controls"]
    assert verdict(2, 1, [], 20)["label"] == "just noise"
    assert verdict(8, 1, [0], 20)["label"] == "something's there"


def test_question_pools_are_disjoint_and_unique():
    assert len(EVAL_QUESTIONS) >= 50 and len(set(EVAL_QUESTIONS)) == len(EVAL_QUESTIONS)
    assert not set(EVAL_QUESTIONS) & set(NEUTRAL_QUESTIONS)


def test_judge_prompt_is_blind():
    p = judge_prompt("talks like a pirate", "Why is the sky blue?", "Arr!")
    assert "pirate" in p and "Arr!" in p and "steer" not in p.lower()
    assert fmt_p(0.0001) == "p < 0.001" and fmt_p(0.02) == "p = 0.020"


def test_screen_questions_held_apart():
    from qwen_neurons.scoring import SCREEN_QUESTIONS
    assert not set(SCREEN_QUESTIONS) & (set(EVAL_QUESTIONS) | set(NEUTRAL_QUESTIONS))


def test_looping():
    from qwen_neurons.scoring import looping
    assert looping("MWAHAHA! " * 60)
    assert looping("I am sorry for my mistake. " * 12)
    assert not looping("The sky is blue because shorter wavelengths of sunlight scatter more in the air, which is "
                       "called Rayleigh scattering; at sunset the light travels further and the reds win.")
    assert not looping("short")


def test_describe_and_rating_prompts():
    from qwen_neurons.scoring import describe_prompt, parse_description, rating_prompt
    p = describe_prompt([("Hi", "Hello!", "Arr matey!")])
    assert "Arr matey!" in p and "TRAIT:" in p and "NAME:" in p
    assert parse_description("TRAIT: talks like a pirate.\nNAME: Captain Barky") == ("talks like a pirate", "Captain Barky")
    assert parse_description("no idea") == (None, None)
    assert "single digit" in rating_prompt("fun", [("q", "n", "o")])
