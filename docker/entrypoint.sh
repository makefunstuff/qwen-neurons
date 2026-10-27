#!/bin/sh
# Use the weights mounted at $QN_MODEL if they are there, else the official Qwen/Qwen3.5-4B from Hugging Face
# (downloaded into the hf-cache volume on the first message). Weights and config only: never code from the repo.
set -e
if [ -n "$QN_MODEL" ] && [ "${QN_MODEL#/}" != "$QN_MODEL" ] && [ ! -f "$QN_MODEL/config.json" ]; then
  echo "no weights in $QN_MODEL: using Qwen/Qwen3.5-4B from Hugging Face (downloads ~9 GB on the first message)"
  export QN_MODEL=Qwen/Qwen3.5-4B
fi
exec python -m qwen_neurons --no-open "$@"
