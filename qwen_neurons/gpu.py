"""GPU checks via nvidia-smi, so the server process never creates a CUDA context (that alone costs VRAM)."""
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def _mib(s: str | None) -> int:
    try:
        return int(str(s).split()[0])
    except (ValueError, IndexError):
        return 0


def _proc_label(pid: str, fallback: str) -> str:
    """A readable name for a process: its command line, else what nvidia-smi says."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        cmd = [c.decode(errors="replace") for c in cmd if c]
        if cmd:
            args = [c for c in cmd[1:] if not c.startswith("-")][:2]
            return " ".join([Path(cmd[0]).name, *(Path(a).name for a in args)])[:60]
    except OSError:
        pass
    return fallback


def gpus() -> list[dict]:
    """All NVIDIA GPUs: index, name, total/used/free MiB and the processes on each. [] without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        xml = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    out = []
    for i, g in enumerate(ET.fromstring(xml).findall("gpu")):
        fb = g.find("fb_memory_usage")
        procs = []
        for p in g.findall("processes/process_info"):
            pid = p.findtext("pid", "")
            procs.append({"pid": int(pid or 0), "type": p.findtext("type", ""),
                          "name": _proc_label(pid, p.findtext("process_name", "?")),
                          "used_mib": _mib(p.findtext("used_memory"))})
        out.append({"index": i,
                    "name": g.findtext("product_name", "GPU"),
                    "total_mib": _mib(fb.findtext("total")), "used_mib": _mib(fb.findtext("used")),
                    "free_mib": _mib(fb.findtext("free")),
                    "procs": sorted(procs, key=lambda p: -p["used_mib"])})
    return out


def pick_device(want: str, need_mib: int, smi: list[dict] | None = None) -> dict:
    """Decide where the model goes. want: 'auto', 'cuda', 'cuda:N', 'mps' or 'cpu'.

    Returns {"device", "gpu_index"?, "ok", "slow", "msg", "gpu"?}. With auto, the CUDA GPU with the most free
    memory wins if it has need_mib free; otherwise MPS (Apple) or CPU, flagged slow. A GPU asked for by name that is
    too full is not ok, with a message naming what holds it."""
    smi = gpus() if smi is None else smi
    want = (want or "auto").strip().lower()
    if want.startswith("cuda:"):
        idx = int(want.split(":")[1])
        g = next((g for g in smi if g["index"] == idx), None)
        if g is None:
            return {"device": want, "ok": False, "slow": False, "msg": f"GPU {idx} not found (nvidia-smi sees {len(smi)})"}
        if g["free_mib"] < need_mib:
            return {"device": want, "gpu_index": idx, "ok": False, "slow": False, "gpu": g, "msg": too_full(g, need_mib)}
        return {"device": want, "gpu_index": idx, "ok": True, "slow": False, "gpu": g,
                "msg": f"GPU {idx} ({g['name']}): {g['free_mib']} MiB free"}
    if want in ("auto", "cuda") and smi:
        g = max(smi, key=lambda g: g["free_mib"])
        if g["free_mib"] >= need_mib:
            return {"device": f"cuda:{g['index']}", "gpu_index": g["index"], "ok": True, "slow": False, "gpu": g,
                    "msg": f"GPU {g['index']} ({g['name']}): {g['free_mib']} MiB free"}
        if want == "cuda":
            return {"device": "cuda", "ok": False, "slow": False, "gpu": g, "msg": too_full(g, need_mib)}
    if want in ("auto", "mps") and _has_mps():
        return {"device": "mps", "ok": True, "slow": True, "msg": "Apple GPU (MPS): works, but slower than CUDA"}
    note = " (no NVIDIA GPU has room)" if smi else ""
    return {"device": "cpu", "ok": True, "slow": True,
            "msg": f"CPU{note}: this works but is SLOW, a reply takes minutes"}


def too_full(g: dict, need_mib: int) -> str:
    holders = ", ".join(f"{p['name']} (pid {p['pid']}, {p['used_mib']} MiB)" for p in g["procs"] if p["used_mib"]) \
        or "processes this container or user can't see"
    return (f"GPU {g['index']} has {g['free_mib']} MiB free, the model needs about {need_mib} MiB. "
            f"In use: {holders}. Free some VRAM and try again (qwen-neurons never stops other GPU processes).")


def _has_mps() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "arm64"
