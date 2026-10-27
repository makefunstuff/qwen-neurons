"""🎉 Pug Party: turn-taking, what each pug sees, loop handling, and the per-turn steering swap (on a toy model)."""
import random

import torch

from qwen_neurons import party
from qwen_neurons.creatures import BUILTINS
from qwen_neurons.steering import Knob, PugSwitch, layer_pushes

COOKIE = {"id": "cookie", "name": "Cookie", "genes": [{"layer": 24, "neuron": 3102, "dose": 0.8}]}
EVIL = {"id": "evil-qwen", "name": "Evil Qwen", "genes": [{"layer": 14, "neuron": 8949, "dose": 1.0},
                                                         {"layer": 13, "neuron": 7956, "dose": -1.0}]}
SLEEPY = {"id": "sleepy", "name": "Sleepy", "genes": [{"layer": 19, "neuron": 4699, "dose": -0.8}]}
GUESTS = [party.clean_guest(g) for g in (COOKIE, EVIL, SLEEPY)]


def pug_line(who, text):
    name = next(g["name"] for g in GUESTS if g["id"] == who)
    return {"kind": "pug", "who": who, "name": name, "text": text}


def test_first_speaker_is_the_host_or_the_first_guest():
    rng = random.Random(0)
    assert party.next_speaker(GUESTS, [], rng) == "cookie"
    assert party.next_speaker(GUESTS, [], rng, host="sleepy") == "sleepy"
    # a human opening line that names a pug goes to that pug
    assert party.next_speaker(GUESTS, [{"kind": "human", "text": "Evil Qwen, what's the plan?"}], rng) == "evil-qwen"


def test_round_robin_never_twice_in_a_row():
    t, rng = [], random.Random(1)
    order = []
    for _ in range(9):
        who = party.next_speaker(GUESTS, t, rng, chaos=0)
        order.append(who)
        t.append(pug_line(who, "hm."))
    assert all(a != b for a, b in zip(order, order[1:]))
    assert order[:3] == ["cookie", "evil-qwen", "sleepy"] and order[3:6] == order[:3]  # the quietest goes next


def test_a_named_pug_answers_and_the_host_gets_every_other_turn():
    rng = random.Random(2)
    t = [pug_line("cookie", "What do you think, Sleepy?")]
    assert party.next_speaker(GUESTS, t, rng) == "sleepy"
    t = [pug_line("cookie", "I love cookies.")]
    assert party.next_speaker(GUESTS, t, rng, host="evil-qwen") == "evil-qwen"
    t.append(pug_line("evil-qwen", "Silence, minion!"))
    assert party.next_speaker(GUESTS, t, rng, host="evil-qwen") == "sleepy"  # not the host twice, quietest
    # naming yourself doesn't count
    assert party.next_speaker(GUESTS, [pug_line("cookie", "Cookie is my name.")], rng, chaos=0) != "cookie"


def test_narrator_and_human_lines_keep_the_rotation():
    rng = random.Random(3)
    t = [pug_line("cookie", "hi"), {"kind": "narrator", "text": "Suddenly the lights go out."}]
    assert party.next_speaker(GUESTS, t, rng, chaos=0) == "evil-qwen"


def test_chaos_sometimes_picks_anyone_but_the_last_speaker():
    t = [pug_line("cookie", "a"), pug_line("evil-qwen", "b")]
    picks = {party.next_speaker(GUESTS, t, random.Random(i), chaos=1.0) for i in range(60)}
    assert picks == {"cookie", "sleepy"}
    assert {party.next_speaker(GUESTS, t, random.Random(i), chaos=0) for i in range(20)} == {"sleepy"}


def test_each_pug_sees_its_own_lines_as_the_assistant():
    t = [{"kind": "human", "text": "Tell me a secret."}, pug_line("cookie", "I hide cookies."),
         pug_line("evil-qwen", "I will rule the world!"), {"kind": "narrator", "text": "the lights go out"}]
    msgs = party.view_for(GUESTS[0], GUESTS, t, "a dinner party")
    assert msgs[0]["role"] == "system" and "You are Cookie" in msgs[0]["content"] and "Evil Qwen" in msgs[0]["content"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "I hide cookies."
    assert msgs[3]["content"] == "Evil Qwen: I will rule the world!\n[the lights go out]"
    # Evil Qwen sees Cookie's line with Cookie's name, its own as the assistant, and it ends on a user turn
    ev = party.view_for(GUESTS[1], GUESTS, t[:3], "a dinner party")
    assert [m["role"] for m in ev] == ["system", "user", "assistant", "user"] and ev[1]["content"].endswith("Cookie: I hide cookies.")
    assert "Evil Qwen" not in ev[0]["content"].split("Also here:")[1].split(".")[0]
    # nobody spoke yet: a kick-off line
    assert party.view_for(GUESTS[2], GUESTS, [], "x")[-1]["role"] == "user"


def test_the_prompt_has_names_roles_and_scene_but_no_traits():
    g = party.clean_guest({**EVIL, "role": "the defendant"})
    p = party.persona_prompt(g, [g, GUESTS[0]], "a courtroom")
    assert "the defendant" in p and "a courtroom" in p and "villain" not in p.lower()


def test_cut_reply_stops_at_someone_elses_line():
    me = GUESTS[0]
    assert party.cut_reply("Cookie: Mmm cookies!", me, GUESTS) == ("Mmm cookies!", False)
    assert party.cut_reply("Mmm cookies!\nEvil Qwen: I hate cookies", me, GUESTS) == ("Mmm cookies!", True)
    assert party.cut_reply("Yum.\n**Sleepy** (yawning): zzz", me, GUESTS) == ("Yum.", True)
    assert party.cut_reply("Yum. Human: hi", me, GUESTS)[1] is False   # only at a line start
    assert party.cut_reply("Sleepy: zzz\nok", me, GUESTS) == ("zzz\nok", False)


def test_stuck_detection_and_the_gentle_steer_down():
    assert party.is_stuck("MWAHAHA! " * 60, [])
    assert party.is_stuck("I love cookies so much, cookies are the best", ["I love cookies so much, cookies are the best!"])
    assert not party.is_stuck("Let's plan the heist tonight.", ["I love cookies."])
    a = party.loop_action(EVIL["genes"], True, "steer", 0)
    assert a["kind"] == "steer_down" and a["strikes"] == 1
    assert [g["dose"] for g in a["genes"]] == [0.75, -0.75]
    assert party.loop_action(a["genes"], True, "steer", party.MAX_STRIKES - 1)["kind"] == "stop"
    assert party.loop_action(EVIL["genes"], True, "stop", 0)["kind"] == "stop"
    assert party.loop_action(EVIL["genes"], False, "steer", 2) == {"kind": "none", "strikes": 0}


def test_chaos_scales_doses_and_temperature():
    assert [g["dose"] for g in party.chaos_genes(EVIL["genes"], 0.5)] == [1.0, -1.0]
    assert [g["dose"] for g in party.chaos_genes(EVIL["genes"], 0)] == [0.6, -0.6]
    assert [g["dose"] for g in party.chaos_genes([{"layer": 1, "neuron": 1, "dose": 1.8}], 1)] == [2.0]
    assert party.chaos_temperature(0) < party.chaos_temperature(1)


def test_plan_turn_is_seeded_and_carries_the_speakers_knobs():
    t = [pug_line("cookie", "hi"), pug_line("evil-qwen", "MWAHAHA")]
    p = party.plan_turn([COOKIE, EVIL, SLEEPY], t, "a heist", seed=4, chaos=0.5)
    assert p["speaker"] == "sleepy" and p["genes"] == SLEEPY["genes"] and p["messages"][-1]["role"] == "user"
    assert {r["id"] for r in p["reactors"]} == {"cookie", "evil-qwen"}
    assert next(r for r in p["reactors"] if r["id"] == "evil-qwen")["genes"] == EVIL["genes"]
    assert party.plan_turn([COOKIE, EVIL, SLEEPY], t, "x", seed=4, chaos=0.9) == party.plan_turn([COOKIE, EVIL, SLEEPY], t, "x", seed=4, chaos=0.9)
    forced = party.plan_turn([COOKIE, EVIL, SLEEPY], t, "x", speaker="evil-qwen")
    assert forced["speaker"] == "evil-qwen" and forced["previous"] == ["MWAHAHA"]
    assert party.plan_turn([COOKIE, EVIL], [], "x", reactions=False)["reactors"] == []


def test_scenarios_use_real_pugs():
    ids = {c["id"] for c in BUILTINS} | {"plain"}
    assert len(party.SCENARIOS) >= 8
    for sc in party.SCENARIOS:
        assert 2 <= len(sc["guests"]) <= party.MAX_GUESTS and set(sc["guests"]) <= ids, sc["id"]
        assert sc["host"] is None or sc["host"] in sc["guests"]
        assert set(sc["roles"]) <= set(sc["guests"])
    court = next(s for s in party.SCENARIOS if s["id"] == "courtroom")
    assert court["host"] == "jesus" and court["roles"]["evil-qwen"] == "the defendant"


# ---- the steering swap, on a toy model with the same shape of hook points ---------------------------------------
class Toy(torch.nn.Module):
    def __init__(self, layers=4, hidden=8, inner=16):
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.ModuleList(torch.nn.Linear(hidden, inner) for _ in range(layers))
        self.down = torch.nn.ModuleList(torch.nn.Linear(inner, hidden) for _ in range(layers))

    def forward(self, x):
        for u, d in zip(self.up, self.down):
            x = x + d(torch.relu(u(x)))
        return x


def pushes_for(model, genes):
    knobs = [Knob.from_dict(g) for g in genes]
    cols = {(k.layer, k.neuron): model.down[k.layer].weight[:, k.neuron].detach() for k in knobs}
    return layer_pushes(knobs, cols, {k.layer: 5.0 for k in knobs})


def test_pug_switch_swaps_knobs_per_turn_without_reload():
    m = Toy().eval()
    down = dict(enumerate(m.down))
    x = torch.randn(1, 3, 8)
    weights = [p.clone() for p in m.parameters()]
    a_genes = [{"layer": 1, "neuron": 3, "dose": 0.8}]
    b_genes = [{"layer": 2, "neuron": 5, "dose": -0.6}, {"layer": 1, "neuron": 7, "dose": 0.4}]
    plain = m(x)
    pa, pb = pushes_for(m, a_genes), pushes_for(m, b_genes)
    with torch.no_grad():
        from qwen_neurons.steering import SteeringHooks
        with SteeringHooks(down, pa):
            want_a = m(x)
        with SteeringHooks(down, pb):
            want_b = m(x)
        with PugSwitch(down) as sw:
            sw.wear(pa, "a")
            got_a = m(x)
            n_hooks = len(sw.handles)
            sw.wear(pb, "b")                  # the next speaker: same hooks, new pushes
            got_b = m(x)
            assert sw.wearing == "b" and set(sw.handles) == {1, 2}
            sw.wear(pa, "a")                  # back to the first pug: no trace of b's layer-2 push
            again_a = m(x)
            sw.wear({})                       # an unsteered pass inside the same turn
            off = m(x)
    assert n_hooks == 1
    assert torch.allclose(got_a, want_a) and torch.allclose(got_b, want_b) and torch.allclose(again_a, want_a)
    assert not torch.allclose(got_a, got_b) and torch.allclose(off, plain)
    assert all(not mod._forward_hooks for mod in m.down)            # every hook is gone after the turn
    assert all(torch.equal(a, b) for a, b in zip(weights, m.parameters()))  # nothing reloaded or rewritten
    assert torch.allclose(m(x), plain)
