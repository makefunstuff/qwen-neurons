import pytest
from qwen_neurons import gpu

SMI = [{"index": 0, "name": "A", "total_mib": 16000, "used_mib": 1500, "free_mib": 14500, "procs": []},
       {"index": 1, "name": "B", "total_mib": 16000, "used_mib": 12000, "free_mib": 4000,
        "procs": [{"pid": 42, "type": "C", "name": "vllm serve", "used_mib": 11900}]}]


def test_named_gpu_too_full_says_who_holds_it():
    r = gpu.pick_device("cuda:1", 9500, SMI)
    assert not r["ok"]
    assert "4000 MiB free" in r["msg"] and "vllm serve (pid 42, 11900 MiB)" in r["msg"]


def test_named_gpu_with_room():
    r = gpu.pick_device("cuda:0", 9500, SMI)
    assert r["ok"] and r["device"] == "cuda:0" and r["gpu_index"] == 0


def test_auto_picks_the_freest_gpu():
    assert gpu.pick_device("auto", 9500, SMI)["device"] == "cuda:0"


def test_auto_falls_back_to_cpu_and_says_slow(monkeypatch):
    monkeypatch.setattr(gpu, "_has_mps", lambda: False)
    r = gpu.pick_device("auto", 99999, SMI)
    assert r["ok"] and r["device"] == "cpu" and r["slow"] and "SLOW" in r["msg"]
    assert gpu.pick_device("auto", 9500, [])["device"] == "cpu"


def test_auto_prefers_mps_on_apple(monkeypatch):
    monkeypatch.setattr(gpu, "_has_mps", lambda: True)
    r = gpu.pick_device("auto", 9500, [])
    assert r["device"] == "mps" and r["slow"]


def test_missing_gpu():
    assert not gpu.pick_device("cuda:7", 1, SMI)["ok"]


def test_gpu_label_names_the_host_gpu(monkeypatch):
    import asyncio
    from qwen_neurons import engine as eng_mod
    from qwen_neurons.config import Settings
    s = Settings.load({"QN_DEVICE": "cuda:0", "QN_GPU_LABEL": "1", "QN_NEED_MIB": "99999999"})
    smi = [{"index": 0, "name": "GPU", "total_mib": 16000, "used_mib": 15000, "free_mib": 1000, "procs": []}]
    monkeypatch.setattr(eng_mod.gpu, "gpus", lambda: smi)
    e = eng_mod.Engine(s)
    with pytest.raises(RuntimeError) as err:     # too full, and the message names the host's GPU 1, not cuda:0
        asyncio.run(e.ensure_loaded())
    assert str(err.value).startswith("GPU 1 has 1000 MiB free")
