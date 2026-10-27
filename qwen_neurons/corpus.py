"""The open text corpus behind the fires view: the first documents of one C4 validation shard.

C4 (allenai/c4 on Hugging Face) is web text released under ODC-BY. One 40 MB shard is downloaded into the
Hugging Face cache the first time the fires index is built; nothing of it is in this repo.
"""
import gzip
import json

REPO, FILE = "allenai/c4", "en/c4-validation.00000-of-00008.json.gz"
SOURCE = "C4 (allenai/c4, English validation shard 0, ODC-BY)"


def read_docs(path: str, n_docs: int, min_chars: int = 300, max_chars: int = 2000) -> list[str]:
    """The first n_docs documents of a C4-style .json.gz (one {"text": ...} per line) with at least min_chars."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("text", "").strip()
            if len(text) >= min_chars:
                out.append(text[:max_chars])
                if len(out) >= n_docs:
                    break
    return out


def load(n_docs: int = 4096) -> list[str]:
    from huggingface_hub import hf_hub_download
    return read_docs(hf_hub_download(REPO, FILE, repo_type="dataset"), n_docs)
