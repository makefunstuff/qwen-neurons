# qwen-neurons

A local web tool for playing with neuron steering on Qwen3.5-4B: "personality knobs".

Steering adds a scaled column of an MLP `down_proj` weight to the residual stream at a chosen layer, for every token. `steer.py` is the proof of concept.
