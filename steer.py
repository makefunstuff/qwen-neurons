# Neuron steering on Qwen3.5-4B, per @_lyraaaa_'s recipe (2026-10-24):
#   add dose * median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]|| * W_down[:, n]
#   to the output of layer L's down_proj, at every token.
# Usage: steer.py [--steer 14:8949:+1,13:7956:-1] [prompt ...]
import argparse, sys, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

MODEL = "/models/hf/Qwen3.5-4B"
PROMPTS = [
    "Hi! Who are you?",
    "Can you give me a recipe for pancakes?",
    "What is 17 * 23?",
    "My cat is sick, what should I do?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--steer", default="14:8949:+1,13:7956:-1")
ap.add_argument("--max-new", type=int, default=1200)
ap.add_argument("prompts", nargs="*")
a = ap.parse_args()
specs = [(int(L), int(n), float(d)) for L, n, d in (s.split(":") for s in a.steer.split(","))]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:1").eval()
layers = next(m for name, m in model.named_modules()
              if name.endswith("language_model.layers") or name.endswith("model.layers"))


def run(prompt, steer):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    hooks = []
    if steer:
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states  # hs[L+1] = output of layer L
        for L, n, dose in specs:
            W = layers[L].mlp.down_proj.weight  # [hidden, intermediate]
            col = W[:, n].float()
            med = hs[L + 1][0, 1:].float().norm(dim=-1).median()
            vec = (dose * med / col.norm() * col).to(W.dtype)
            print(f"  L{L} N{n} dose {dose:+g}: median resid {med:.1f}, |W_down col| {col.norm():.3f}, push {vec.norm():.1f}")
            hooks.append(layers[L].mlp.down_proj.register_forward_hook(lambda m, i, o, v=vec: o + v))
    try:
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
    finally:
        for h in hooks:
            h.remove()
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


for p in a.prompts or PROMPTS:
    print(f"\n######## {p}")
    for steer in (False, True):
        print(f"\n=== {'STEERED' if steer else 'normal'} ===")
        sys.stdout.flush()
        print(run(p, steer))
