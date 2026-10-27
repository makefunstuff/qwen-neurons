"""Settings: environment variables (QN_*), optionally from a gitignored config.local.env in the repo root.

QN_HOST, QN_PORT, QN_MODEL, QN_DEVICE, QN_NEED_MIB, QN_IDLE_TIMEOUT_S, QN_DATA_DIR, QN_NO_BROWSER, QN_GPU_LABEL."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    model: str = "Qwen/Qwen3.5-4B"     # a Hugging Face repo id (downloaded on first use) or a local directory
    device: str = "auto"               # auto | cuda:N (nvidia-smi numbering) | mps | cpu
    need_mib: int = 10000              # free VRAM required before loading on a GPU
    no_browser: bool = False          # QN_NO_BROWSER=1: never open a browser tab
    idle_timeout_s: int = 600          # unload after this long without a request
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    gpu_label: str = ""                # QN_GPU_LABEL: how to call the GPU in messages (in a container: the host's index)

    @classmethod
    def load(cls, env: dict[str, str] | None = None) -> "Settings":
        if env is None:
            env = {**load_env_file(ROOT / "config.local.env"), **os.environ}
        s = cls()
        conv = {"host": str, "port": int, "model": str, "device": str, "need_mib": int,
                "idle_timeout_s": int, "data_dir": Path, "gpu_label": str,
                "no_browser": lambda v: v.lower() in ("1", "true", "yes", "on")}
        for name, f in conv.items():
            v = env.get("QN_" + name.upper())
            if v not in (None, ""):
                setattr(s, name, f(v))
        return s
