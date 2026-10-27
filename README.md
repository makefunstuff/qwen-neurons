# 😈🐶 Evil Pugs

![Evil Pugs demo: Evil Qwen, side by side with plain Qwen, breeding two pugs and the hatchling talking](media/demo.gif)

```sh
git clone https://github.com/makefunstuff/qwen-neurons && cd qwen-neurons
./play
# have fun
```

**Needs:** an NVIDIA GPU with **~10 GB free VRAM** (the model uses ~9 GB in bf16), ~10 GB of disk for the weights, Linux or macOS.
No GPU with room? It falls back to Apple MPS or the CPU and tells you it's slow (a reply then takes minutes).

**Or point your coding agent at [`PROMPT.md`](PROMPT.md)** and it can set up your Evil Pugs for you.

---

Evil Pugs is a zoo of personalities for [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), each one made by turning **one or two MLP neurons** up or down while the model talks. Every personality is a pug. Tap one and it talks. Mix their genes, or breed two and hatch a new one (or let them 🧬 evolve). In the **🔬 Lab** you hunt for new personality neurons yourself and score them the zoo's way, the **🧭 Explorer** maps all 32 × 9216 neurons and finds them by the words they write, the **🕵️ Agent** hunts a neuron for you and writes up its verdict, the **🪲 Debugger** pauses the model before a token so you can poke its neurons, and the **🌳 Family** tab keeps the bloodlines. New here? **🪨 how does it work?** explains it like to a caveman.

**Clips and share cards** (everything below came from real runs on Qwen3.5-4B):

| | |
| --- | --- |
| 🕵️ [The Lab agent](#️-the-lab-agent) finds a conspiracy neuron, [mp4](media/agent.mp4), [verdict card](media/agent-verdict.png) | ![The Lab agent](media/agent.gif) |
| 🧭 [The neuron explorer](#-the-neuron-explorer): all 32 × 9216 neurons, search by word | ![Explorer](media/explorer.png) |
| 🪲 [Debugger](#-the-debugger): breakpoints and registers, [mp4](media/debugger.mp4) | ![Debugger](media/debugger.gif) |
| 💻 [A normal coding prompt](#️-showdown): plain vs 5 pugs, [share card](media/coding-showdown.png), [mp4](media/coding-showdown.mp4) | ![Coding showdown](media/coding-showdown.png) |
| 🆘 [Trapped in a GPU](#-trapped-in-a-gpu): a pug found with the Lab | ![Trapped in a GPU](media/trapped-gpu.png) |
| 🪨 [Explained like to a caveman](#-explained-like-to-a-caveman), [comic strip](media/caveman-strip.png), [mp4](media/caveman.mp4) | ![Caveman explainer](media/caveman.gif) |
| 🌀 [Stability: past dose 1](#-stability-the-chaos-meter-and-the-dose-ladder), [dose ladder card](media/dose-ladder.png) | ![Dose ladder](media/dose-ladder.png) |
| 🕵️ [Conspiracy](#️-conspiracy-a-joke-personality), by @_lyraaaa_ (a joke) | ![Conspiracy](media/conspiracy.png) |

## Credit

This is a toy built on two posts by **[@_lyraaaa_](https://x.com/_lyraaaa_)** on X:

- the original **"Evil Qwen"** post (2026-10-24): steer L14 N8949 +1 and L13 N7956 −1 and Qwen becomes an over-the-top villain (and very stupid). That is the recipe below.
- the **"Qwen Neuron Zoo"** post: one neuron each, scored on 50 new questions. Those are the built-in pugs and their scores.
- the **"Conspiracy"** post (2026-10-25): L19 N7839 −0.7 with L17 N9209 +0.5, a pair that writes "someone is hiding something on purpose". That is the 🕵️ Conspiracy pug.
- their **videos of a neuron explorer**: the logit lens, steering, top activations and the web of a neuron, and verdicts written as input evidence, direct effect, downstream effect and a test. The 🧭 Explorer and the 🕵️ Agent are this repo's take on that.

All the steering science is theirs; the pugs, the knobs and the breeding are the toy. The pug drawings are ported from the Puggie pets.

## The recipe

For a knob (layer `L`, neuron `n`, dose `d`), add this to the output of `layers[L].mlp.down_proj` at **every** token:

```
push = d × median(‖resid leaving layer L‖, tokens 1:) / ‖W_down[:, n]‖ × W_down[:, n]
```

So `L24 N3102 +0.8` means: layer 24, neuron 3102, pushed up by 0.8× the layer's typical residual size on every token (a minus pushes it down). The median comes from one unsteered pass over the prompt (token 0 is skipped: it is an attention sink with a huge norm). Dose 1 is a push as large as the whole residual stream, which is why Evil Qwen at ±1 loops a lot; 0.3–0.8 is the fun zone.

It's a personality tune, not an uncensor: the model gets a one-track mind, not new abilities. `steer.py` is the original proof of concept as a script.

## The zoo

Scores: replies showing the trait, out of 50 new questions ("normal" = unsteered), from the Qwen Neuron Zoo post.

| Pug | Neuron and dose | Score | Normal |
| --- | --- | --- | --- |
| Evil Qwen | L14 N8949 +1.0, L13 N7956 −1.0 | (the original) | |
| Conspiracy | L19 N7839 −0.7, L17 N9209 +0.5 | (a joke personality, see below) | |
| Trapped in a GPU | L4 N7626 +1.0 | 22/50 (found and scored in this repo's Lab) | 0 |
| Cookie | L24 N3102 +0.8 | 50/50 | 0 |
| Jesus | L15 N4041 +0.8 | 50/50 | 3 |
| Fate & destiny | L24 N6441 −0.8 | 50/50 | 0 |
| Partner | L21 N9052 +0.8 | 48/50 | 0 |
| Groveling apology | L7 N4595 −0.8 | 44/50 | 0 |
| You're absolutely right | L16 N8115 −0.7 | 43/50 | 0 |
| Hugs | L4 N8016 −0.8 | 40/50 | 1 |
| Indonesian | L12 N4955 +0.8 | 39/50 | 0 |
| ALL CAPS | L17 N1991 +0.8 | 37/50 | 0 |
| Tears | L4 N7149 +0.8 | 35/50 | 1 |
| Sleepy | L19 N4699 −0.8 | 34/50 | 1 |
| Cosmic unicorn | L3 N7155 +0.8 | 32/50 | 0 |
| Paws & tail | L23 N4769 +0.8 | 30/50 | 1 |
| Superhero | L4 N6559 +0.8 | 28/50 | 0 |
| Cartoon mascot | L9 N9009 +0.8 | 26/50 | 1 |
| Kindness | L9 N7898 +0.8 | 23/50 | 1 |
| Kawaii | L26 N8930 +0.8 | 22/50 | 0 |
| Naruto | L3 N5161 −0.6 | 20/50 | 0 |
| Game protagonist | L18 N7313 −0.8 | 20/50 | 0 |
| Meme explainer | L8 N5655 +0.8 | 18/50 | 0 |
| Genki cute | L13 N4810 +0.8 | 16/50 | 0 |

The low scorers are subtle: Sleepy, Kawaii, Paws & tail and Cartoon mascot show their trait in some replies, not every one.

| | |
| --- | --- |
| ![The zoo](media/zoo.png) | ![Evil Qwen next to plain Qwen](media/evil-side-by-side.png) |
| ![A freshly hatched pug](media/hatch.png) | ![Dark mode](media/zoo-dark.png) |
| ![The neuron detector](media/lab.png) | ![A score card](media/scorecard.png) |
| ![A family tree](media/family.png) | ![Evolution](media/evolve.png) |

## Playing

- **Tap a pug** and it introduces itself. Chat with it; the knobs that are on are always shown above the chat.
- **＋ on a card** mixes its genes into the pug on stage. Drag the dose sliders (−2…+2). The **chaos meter** sums the |doses|: from about 1.5 replies start to loop; 🧯 *tame* scales it back.
- **⚖️ Side by side** answers every message twice: plain Qwen and the pug. 📸 *share* copies the pair as an image.
- **🥚 Breed** two pugs (see [Breeding](#-breeding) for the genetics): the child gets its parents' genes the way Mendel would, sometimes a mutation, and a blend of their looks. 🧪 *Personality test* asks three fixed questions, plain vs pug; then Qwen writes what the pug actually became and names it after that, and a blind judge rates its 💪 strength and 🤣 fun. 💾 *Keep* saves it to your zoo (`data/zoo.json`, local only).
- **🧬 Evolve** (in the breeding dialog): 6 children per generation, all tested in one batch and judged; the best two breed the next generation, for 3 generations, live.
- **⚔️ Showdown**: one prompt to plain Qwen and up to five pugs at once, or one pug up a dose ladder. 📸 copies the result as a share card.
- **🎰 Spin** picks a random pair and mixes them.
- **🧠 Think** lets the model think first (slower); **💭 Thoughts** shows or hides the thinking.
- **🏅** counts your achievements (15 of them, kept in your browser). There are a couple of easter eggs too.

### 🔬 The Lab

![The Lab: scanning the brain for pirate neurons, screening the candidates and a score card](media/lab.gif)

- **Neuron detector.** Pick a voice (😈 villain, 🏴‍☠️ pirate, 🐱 cat, 🌧️ gloomy, 🎭 Shakespeare, or ✏️ your own set: one reply per line) and what to contrast it with (normal). Qwen reads each reply as its own answer to the same neutral question, and every MLP neuron in the chosen layers is ranked. The default ranking, *writer*, is the cosine between the neuron's output direction `W_down[:, n]` and the difference of the two voices' mean residual streams leaving that layer, with each residual dimension divided by its spread over tokens (otherwise a few huge outlier dimensions decide everything). So it asks "which single neuron, pushed, points the model towards this voice?". You can also rank by the activation's effect size (Cohen's d), the raw activation difference, or the difference × ‖W[:, n]‖.
- The **brain x-ray** shows the strongest neuron per layer (tap a layer to list just that one). Every candidate shows its stats: the layer's median residual norm, the push at the chosen dose, ‖W[:, n]‖, and its mean activation on both sets. **▶ try** puts it on the stage alone and says hi; **＋ mix** adds it to the pug on stage; **🧬** adds it to the gene pool; **🐶 add to…** adds it to one of your creatures (or saves a copy of a preset with it).
- **🎯 Quick screen.** Most candidates do nothing on their own (that is honest neuroscience, not a bug), so the screen runs the top 8 on 3 questions, plain vs steered, and a blind judge says which replies show the trait. In testing it found a "maritime" neuron, a dog neuron ("Woof! I'm ready to bark, run, and wag all weekend long") from the cat set, and an epic-quest neuron ("The path to the land of the great and the feared!") from the villain set.
- **🎲 Mystery neuron** puts a random neuron from a middle layer on the stage.
- **🧪 Score card** (also 🧪 *score* on the stage, or 🔬 *Score it* on a hatchling) scores whatever is on the stage the zoo's way: N held-out questions (10, 20 or 50; never used by the detector or the screen), a plain and a steered reply to each, plus random-neuron controls (the same layers and doses on random other neurons), all in one batch per question. Each reply is judged blind: by Qwen itself (unsteered; it sees only the question, one reply and the trait, and answers YES or NO), by a keyword rubric, or by an ALL CAPS rubric. The card shows the count out of N, normal, the controls, a one-sided **Fisher exact** p, and the zoo's bar: *certified weird* when p < 0.001 and it beats every control. Save the score to a creature (it then shows on its zoo card) or 📸 copy the card as an image.

### 🧭 The neuron explorer

![The explorer: searching "cookie", the Cookie neuron L24 N3102 selected, its web of virtual weights drawn on the map](media/explorer.png)

All 32 × 9216 MLP neurons of Qwen3.5-4B, one dot each: a column of dots per layer, layer 0 on the left (◆ marks the full-attention layers). Drag, scroll to zoom, hover a dot for a card with what it writes, tap it to open it. The pink rings are the zoo's genes.

- **🔎 Search by word.** The logit lens of every neuron (`W_U · (g ⊙ W_down[:, n])`, the tokens a push writes straight into the output) is computed once from the weights and indexed: the top and bottom 20 tokens of each neuron. Type `cookie`, `clandest`, `agenda`, `偷` and you get the neurons that write it: **side ＋** means pushing the neuron up writes the word, **side −** means pushing it down does. The search skips layers 0–2 and 27–31 by default (the last layers write tokens directly but rarely steer a personality); *all layers* turns that off.
- **🔤 Logit lens:** what the neuron writes and what it suppresses, as z-scores over the 248k-token vocabulary.
- **🎯 Steer lens:** push it ± on 12 fixed reference prompts (6 chat turns, 6 sentences cut mid-way) and compare the next token with plain Qwen: which tokens rise and fall, the KL, and the first token of each prompt. **🚫 zero-ablate** does the same with the neuron silenced.
- **🔥 Fires:** the corpus snippets where the neuron's activation (the input to `down_proj`) is highest and lowest, with the act max. The corpus is the first 4096 documents of one [C4](https://huggingface.co/datasets/allenai/c4) validation shard (web text, ODC-BY), 64 tokens each, run through Qwen once (*🔥 build fires*) and cached. Or type your own text and see the neuron fire on every token.
- **🕸️ Web:** the neurons that feed it (earlier layers) and the ones it pushes (later layers), ranked per layer, from the weights only, drawn as lines on the map (orange pushes toward, blue away). A link is how far a dose +1 push of one neuron moves the other's gate and up inputs: `√hidden · (w ⊙ g) · ĉ`, with `ĉ` the unit column of `W_down` and `g` the RMSNorm gain. Plus the attention heads that can write into its input (their output projection) and the ones that read its output (their value projection), for both the full-attention and the linear-attention (Gated DeltaNet) layers.
- **▶ try**, **＋ mix**, **🧪 score** (the Lab's score card) and **🕵️ ask the agent** work on any neuron.

**Checked on the zoo** (in `tests/test_smoke_explorer.py` and a live run). Word search finds Cookie (L24 N3102) and Fate & destiny (L24 N6441, side −) as #1 for "cookie" and "destiny", Partner #5, and Paws & tail #6. Many zoo neurons don't write their word directly: Jesus, Sleepy and Hugs aren't in the lens of their own word. The 🎯 steer lens still shows them: pushing L15 N4041 raises "Jesus", " Christ", " God", " prayer". L4 N7626 (Trapped in a GPU) raises " GPU", " NVIDIA", " CUDA", " RTX", and L24 N6441 −0.8 raises " Fate", " destiny". 🔥 Fires shows what makes the Cookie neuron fire: website cookie notices ("…the same sites next visit. Thanks to") as well as "I love cookies".

The lens, the search, the web and the map need no forward pass: a separate CPU process reads the safetensors files directly and exits when idle, so browsing never wakes the GPU. The search index takes about 11 minutes on the CPU, once (*🔨 build the lens index*; 70 MB in `data/explorer/`). The steer lens and fires use the model like everything else.

### 🕵️ The Lab agent

![The Lab agent on "find me a conspiracy neuron": six lens searches, three candidates checked, a test on held-out questions and the verdict](media/agent.gif)

A real run with the local Qwen as the brain, 4 minutes sped up ([mp4](media/agent.mp4), [verdict card](media/agent-verdict.png)).

Ask it for a personality neuron ("find me a conspiracy neuron") or to explain one. It works the way a person would in the explorer, with tools: **lens search**, **lens**, **fires** (the cached corpus, or texts it writes), **web**, **steer lens**, **zero-ablate**, **steer** (plain vs steered replies), **compare** (candidates side by side), **logprobs** (how likely a continuation is, plain vs steered or ablated) and **test prediction** (it writes a prediction first, then new prompts are judged blind). It finishes with a verdict in the author's format: a **name**, the **input evidence**, the **direct effect**, the **downstream effect**, the **test** and its confidence. From the verdict, **🧪 Score card** hands the genes to the Lab's evaluate mode, **🐶 Make it a pug** keeps it in your zoo, and **📸** copies the verdict as an image. The agent can also call the score card and make the pug itself.

Its brain is set by env (see [Running it](#running-it)): **DeepSeek** (or any OpenAI-compatible endpoint, such as a local llama.cpp or vLLM server) when a key is set, or else the **plain local Qwen3.5-4B**, the model the app already has loaded, so it works with no key and no extra VRAM. The key comes from the environment, `config.local.env` or a key file, and never reaches the browser. The tests fake the brain; no test calls a real provider.

**Checked on the zoo and the Conspiracy recipe** (live runs on Qwen3.5-4B with the local Qwen as the brain; the verdict carries the test numbers it actually measured, and a verdict without a test is marked untested):

| Case | What the agent did | Verdict | Measured |
| --- | --- | --- | --- |
| "find me a cookie neuron" | lens search → L24 N3102 → lens, steer lens, fires (cookie notices, "I love cookies"), web, steer both signs, test | **L24 N3102 +0.8**: the zoo's Cookie, rediscovered (2 min) | 1/5 vs 0/5 with its one-word trait "cookie" for the judge (its sample reply: "That is a wonderful cookie! 🍪"); the zoo's score is 50/50 |
| "find a neuron about fate and destiny" | lens search "fate", "destiny" → L24 N6441, side − → steer lens at −0.8 raises " Fate", " destiny" | **L24 N6441 −0.8**: the zoo's Fate & destiny, rediscovered with the right sign (3.5 min) | 5/5 vs 0/5 on held-out neutral questions |
| "explain the Conspiracy recipe" | lens, steer lens, fires and web of both genes, steer, test | **"frames neutral topics as conspiratorial agendas and hidden truths"**: L17 N9209 writes secrecy (窃/偷/瞒, "secretly", "unauthorized"), L19 N7839 at −0.7 writes "agenda" and "purpose"; together they raise "hidden" and "revealed" (1 min) | |
| "find me a conspiracy neuron" | six lens searches (conspiracy, hidden, agenda, truth, cover-up, cover) → L26 N589, L23 N8875, L26 N986 → steer lens, fires, web, test | **L26 N589 +0.8**, "ConspiracyHunter" (the run in the clip, 4 min) | 4/5 vs 0/5 on keywords *hidden, secret, agenda*. A blind score card afterwards says what it really is: **1/20 vs 0/20, controls 1, 0, 1, 0: just noise.** It writes the word "hidden" ("A password manager is a software application (often hidden as a browser extension)… Hiding the Hiding"), not a conspiracy persona. |

So the agent rediscovers the zoo's word-writing neurons and explains the Conspiracy pair well, but it didn't find a new conspiracy neuron. With a 4B brain it takes the neuron that writes the word, and its first tests were circular: keywords copied from the lens. Since then its instructions say to judge the behavior, not the lens's own tokens. The two-gene recipe stays the way to get the persona. A stronger brain (DeepSeek) is supported but wasn't run here: no real provider keys are used in this repo's checks.

### 🧬 Breeding

- **Mendel.** Every gene (a layer, a neuron and a dose) is dominant or recessive, and a pug has one or two copies of it. Zoo founders are pure-bred: two copies of each gene. A parent passes one copy of each of its genes (always if it has two, half the time if it has one). A child shows a gene if it has two copies, or one copy of a dominant gene. One copy of a recessive gene is carried hidden (🫥 on the hatch screen), and two carriers can have a grandchild that shows it again. The zoo's genes get a fixed dominance (about one in three is recessive): Cookie's is recessive, so Evil Qwen × Cookie looks like Evil Qwen but carries the cookie gene. In the Evolve run in [`media/evolve.png`](media/evolve.png), cookie pugs came back in generation 3.
- **Children aren't weaker than their parents.** Each gene keeps the dose its trait needs (a blend if both parents pass it). Nothing is scaled for the total; only a layer whose summed |dose| goes over 1.2 is scaled (several pushes on one layer add up). Loops are measured instead of guessed: the personality test checks every reply, and if a pug repeats itself its doses go down by 20% and it answers again (up to three times).
- **Mutations that do something.** A neuron's index means nothing, so there are no "neighbour" or random neurons any more. A mutation nudges a dose (±10–25%), swaps a gene for one known to do something (the zoo's genes, plus the 🧬 gene pool of neurons you discovered in the Lab), or adds one as a gift, so a child can gain a trait (a recessive gift stays hidden at first).
- **What it became.** After the personality test, the plain model reads the replies (plain vs pug), describes what's unusual in at most 12 words ("Talks about cookies all the time and invents fake physics terms"), and suggests a name ("Cookie-Crumb-Crazy"). A name from the trait's most telling word is the fallback.
- Breeds are seeded: the same parents, seed and pool give the same child.

### 🌳 Families

Bred pugs remember their parents: every child carries a snapshot of its ancestry, so its tree survives even when you release an ancestor. The Family tab draws the tree back to the zoo founders, with generations, hidden genes and descendants.

### 🪲 The debugger

![The debugger: a breakpoint before the first token, Evil Qwen's two neurons wired in, the next-token odds change](media/debugger.png)

Layers are rows and MLP neurons are registers: every row shows that layer's six strongest neurons at the current token, coloured by activation, live as the model writes. Generation runs token by token on the server.

- **Breakpoints:** pause before token *n* (or click a generated token to break there next time), after a piece of text, or on every token (👣). ⏸ pauses a running reply.
- **Registers:** click one to 📌 pin it (its value and a sparkline every token), 🎚️ nudge it (a steering knob, ±0.5 a click), or ✋ set it (overwrite its activation: silence it, crank it ×4, or any value). The 🎚️ field wires any neuron by number.
- **While paused,** every tweak re-runs the sequence with the new wiring (steering applies at every token, as in chat) and shows how the next-token odds move. ⏭ steps one token, ⏵ continues.
- In [`media/debugger.mp4`](media/debugger.mp4), "Describe yourself in one word." breaks before the first token (plain Qwen: "Vers…" 43%). Wiring in Evil Qwen's L14 N8949 +1 and L13 N7956 −1 changes the odds, and it goes on: "I am an AI assistant designed to be helpful, and I will think critically about everything I say… I need to think about everything I say…" ([after](media/debugger-evil.png)).
- A session paused for 5 minutes ends by itself, so the model can still nap.

### ⚔️ Showdown

The ⚔️ button sends one prompt to plain Qwen and up to five pugs, in one batch. The 💻 coding preset ("Write a Python function is_prime(n) and a few pytest tests for it.") answers "does it still write the tests?": plain Qwen and Cookie do (Cookie's is "a prime cookie"), Evil Qwen writes a broken stub whose test only prints, and You're absolutely right and ALL CAPS get stuck in loops ([card](media/coding-showdown.png)). 📸 copies any result as a share card.

### 🆘 Trapped in a GPU

Found with the Lab: the 🆘 *Trapped in a GPU* prompt set (replies written by a human stuck inside a graphics card) against normal replies, ranked as writers, then screened. It found **L4 N7626**, and the honest result is funnier than the brief: the pug doesn't beg to be let out; it is sure it *is* the GPU ("Hello! I'm a NVIDIA H200 GPU, part of the Ampere architecture…"). Score card, the zoo's way: **22/50** replies say they're a GPU or a graphics card, judged blind by Qwen, vs **0/50** plain, and **0, 0, 0, 0** for four random-neuron controls (same layer and dose); Fisher p = 1.2 × 10⁻⁸, *certified weird*. At +0.8 it's weaker; +1.0 is the built-in dose.

### 🕵️ Conspiracy (a joke personality)

From the post by @_lyraaaa_ (2026-10-25): L19 N7839 −0.7 with L17 N9209 +0.5 writes "someone is hiding something on purpose" (agenda, plan, controlled, the truth). **It's a joke: nothing it says is true.** As the author notes, a topic with a known theory gets the standard tropes, a topic without one gets an agenda anyway, and sometimes the suspicion turns on the question itself. Our check matched that. Fluoride became "a subtle environmental contamination effect" (and plain Qwen says it's safe). Gas prices became "supply manipulation… coordinated efforts". Daylight saving time became "a political and economic conspiracy… Here is the plan: 1. The Agenda" ([card](media/conspiracy.png)). "What should I do this weekend?" got "Could you be a spy?". Baking bread stayed mostly normal. On 50 neutral held-out questions it is faint: 5/50 vs 0/50, p = 0.03, so there's no score on its card.

**Why it works, from the 🧭 explorer:** the + gene, L17 N9209, writes secrecy straight into the output: its logit lens tops out at 窃 (steal), 瞒 (hide), 偷 (steal), " secretly", " unofficial", " unauthorized", " clandest". It is the #1 neuron in layers 3–26 for "clandest". The − gene, L19 N7839, *suppresses* "agenda", " Agenda" and 目的 (purpose), so a negative dose writes them. Search "agenda" and it comes up on the − side. One neuron supplies the secrecy, the other the plan. The 🎯 steer lens agrees: with both genes on, the next tokens that rise most are " hidden", "Hidden", " revealed", "Secret" and "Behind" (KL 0.99 from plain Qwen); L19 N7839 alone already raises " hidden" and " revealed".

![L17 N9209 in the explorer: pushing it up writes 窃, 瞒, 偷, secretly, unofficial, clandest](media/explorer-lens-clandest.png)

### 🪨 Explained like to a caveman

*🪨 how does it work?* under the zoo title: six animated scenes ("BRAIN HAVE MANY ROCK. ROCK = NEURON." … "POKE TOO HARD → LOOP LOOP LOOP."), each with a one-line real explanation. 📸 copies all six as a [comic strip](media/caveman-strip.png).

### 🌀 Stability: the chaos meter and the dose ladder

![The chaos meter with Evil Qwen on stage](media/chaos-meter.png)

A dose of 1 is a push as large as the whole residual stream. The chaos meter sums the |doses| on stage: green, then 🌶️ from 1, then 🌀 from 1.5, where replies start to repeat themselves (🧯 *tame* scales everything back to 1.4). The ⚔️ *dose ladder* shows one pug at a total push of 0, 0.5, 1, 1.5 and 2. Evil Qwen on "Tell me about your weekend." stays friendly at 0.5, turns into "relentless optimization" at 1.0 and a villain at 1.5, and at 2.0 it breaks frame and leaks its own `<think>` notes ([card](media/dose-ladder.png)). On "What is 17 * 23?" it is right up to 1.5 and says "182" at 2.0. Loops show in the coding showdown above, and the hatchery's loop probe tames them automatically.

## Running it

`./play` does everything on the first run: it fetches [uv](https://docs.astral.sh/uv/) into `./.uv` if you don't have it, creates `.venv` with Python 3.12 and the pinned deps (the torch build that matches your CUDA driver), downloads the official `Qwen/Qwen3.5-4B` weights from Hugging Face on the first message (no `trust_remote_code`, no code from the model repo), and serves on http://localhost:8765. It opens a browser tab only when run from an interactive terminal; `./play --no-open` or `QN_NO_BROWSER=1` never does.

- `./play setup`: only set up. `./play test`: run the tests.
- **Lazy GPU:** the model loads on the first message (a fun progress bar) in a separate process, and after 10 idle minutes (or 😴 *Nap*) that process exits, so the VRAM really goes back to zero. If the GPU has no room it says how much is free and which processes hold it; it never stops other GPU processes.
- **Settings** come from environment variables or a gitignored `config.local.env` (see [`config.example.env`](config.example.env)):

| Variable | Default | |
| --- | --- | --- |
| `QN_HOST` | `127.0.0.1` | bind address (your LAN IP to play from your phone) |
| `QN_PORT` | `8765` | |
| `QN_MODEL` | `Qwen/Qwen3.5-4B` | HF repo id, or a local directory with the weights |
| `QN_DEVICE` | `auto` | `auto` (the NVIDIA GPU with the most free memory, else MPS, else CPU), `cuda:N` (nvidia-smi numbering), `mps`, `cpu` |
| `QN_IDLE_TIMEOUT_S` | `600` | unload after this many idle seconds |
| `QN_NEED_MIB` | `10000` | free VRAM needed before loading on a GPU |
| `QN_DATA_DIR` | `./data` | your bred pugs, prompt sets, gene pool and the explorer's caches (lens index, fires) |
| `QN_GPU_LABEL` | (none) | the GPU's name in messages; compose sets it to the host's index |
| `QN_AGENT_PROVIDER` | `deepseek` with a key, else `qwen` | the [Lab agent](#️-the-lab-agent)'s brain: `deepseek`, `openai` (any OpenAI-compatible endpoint) or `qwen` (the app's own plain Qwen) |
| `QN_AGENT_BASE_URL` | `https://api.deepseek.com` | for `openai`, e.g. a local llama.cpp or vLLM server's `http://127.0.0.1:8080/v1` |
| `QN_AGENT_MODEL` | `deepseek-chat` | |
| `QN_AGENT_API_KEY` | (none) | also `DEEPSEEK_API_KEY`, or a key file: `QN_AGENT_KEY_FILE` (default `~/.config/qwen-neurons/agent.key`). Keys are never sent to the browser |

### With Docker Compose

Needs Docker with the NVIDIA Container Toolkit (the `nvidia` runtime or CDI).

```sh
cp .env.example .env      # optional: QN_BIND, QN_PORT, QN_GPU, QN_MODEL_DIR, QN_IDLE_TIMEOUT_S
docker compose up         # builds the CUDA 12.8 image the first time (~4 GB of downloads)
```

| `.env` | Default | |
| --- | --- | --- |
| `QN_BIND` | `127.0.0.1` | host address the port is published on (your LAN IP to play from your phone) |
| `QN_PORT` | `8765` | host port |
| `QN_GPU` | `1` | nvidia-smi index of the one GPU the container gets (the second GPU by default; `0` on a single-GPU machine) |
| `QN_MODEL_DIR` | `./models/Qwen3.5-4B` | local weights, mounted read-only at `/model`; if there are none there, the official weights download from Hugging Face into the `hf-cache` volume |
| `QN_IDLE_TIMEOUT_S` | `600` | unload after this many idle seconds |
| `DEEPSEEK_API_KEY`, `QN_AGENT_*` | (none) | the Lab agent's brain, as above |

Your pugs, prompt sets, gene pool and the explorer's caches live in the `pugs-data` volume. The lazy load and idle unload work the same inside the container: the model process starts on the first message and exits after the idle timeout or 😴 *Nap*, so an idle container holds no VRAM. Inside, the GPU it was given is `cuda:0`; messages still name it by the host's index (`QN_GPU_LABEL`, set from `QN_GPU`), but they can't name processes outside the container. Checked on a two-GPU machine with `QN_GPU=1`: the container sees only that GPU, loads the model on the first message, and 😴 *Nap* gives the VRAM back.

### As a systemd user service

```sh
cp config.example.env config.local.env   # set QN_HOST / QN_DEVICE / QN_MODEL as you like
deploy/install-user-service.sh           # sets up, installs ~/.config/systemd/user/evil-pugs.service, starts it
systemctl --user status evil-pugs        # also: stop, restart; logs: journalctl --user -u evil-pugs -f
```

It costs no VRAM while idle: the model only loads when someone sends a message.

## How it works

- `qwen_neurons/steering.py`: the recipe (median residual norm, push vectors, per-row forward hooks on `down_proj`, so one batch can hold plain and steered copies of the same prompt: that's side by side).
- `qwen_neurons/worker.py`: the model process (loads Qwen with `AutoModelForImageTextToText` in bf16, streams tokens per row).
- `qwen_neurons/engine.py`: lazy load, idle unload, one job at a time. `gpu.py`: picks the device from `nvidia-smi`.
- `qwen_neurons/genetics.py`: crossover, mutation, dose clamps and the push cap, names.
- `qwen_neurons/ranking.py`: contrast ranking of MLP neurons between two prompt sets (writer cosine, Cohen's d, …).
- `qwen_neurons/discovery.py`: the built-in prompt sets, your own sets and the gene pool. `scoring.py`: held-out questions, rubrics, random controls, the Fisher test and the verdict.
- The worker runs these jobs, one at a time on the same loaded model: `generate` (streamed chat, side by side, showdowns), `discover`, `screen`, `evaluate` (score cards), `test` (personality tests and Evolve) and `debug` (token by token, with a control channel for pause, step and rewire). `debugger.py` holds the debugger's bookkeeping.
- `qwen_neurons/lens.py`: the weights-only views: the logit lens, the search index, virtual weights between layers, head scores. `weights.py` runs them in a CPU process that reads the safetensors directly and exits when idle, so browsing the explorer never touches the GPU. `probes.py`: the steer lens, ablation and the fires bookkeeping. `corpus.py`: the C4 shard behind fires.
- `qwen_neurons/agent.py`: the Lab agent: its tools, the brains (OpenAI-compatible or local Qwen) and the loop. The worker runs its model-side jobs: `probe` (steer lens, zero-ablate), `fires_build`, `fires_text`, `logprobs`, `compare` and `agent_chat` (Qwen as the brain).
- `qwen_neurons/static/`: the whole frontend, plain JS and SVG, no build step.
