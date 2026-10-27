"""Owns the model process: lazy load on the first request, unload after an idle timeout or on demand.

The server never imports torch; the worker process does, so killing it gives every byte of VRAM back.
"""
import asyncio
import itertools
import multiprocessing as mp
import queue
import time

from . import gpu, worker
from .config import Settings


class ModelBusy(Exception):
    pass


class Engine:
    def __init__(self, s: Settings):
        self.s = s
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.jobs = self.events = self.cancel = self.ctl = None
        self.state = "asleep"       # asleep | loading | ready | error
        self.msg = ""
        self.progress = {"frac": 0.0, "msg": ""}
        self.device = None
        self.slow = False
        self.last_used = time.time()
        self.job_lock = asyncio.Lock()
        self.load_lock = asyncio.Lock()
        self.ids = itertools.count(1)

    # ---- status -------------------------------------------------------------------------------------------
    def status(self) -> dict:
        idle_left = None
        if self.state == "ready" and not self.job_lock.locked():
            idle_left = max(0, int(self.s.idle_timeout_s - (time.time() - self.last_used)))
        return {"state": self.state, "msg": self.msg, "progress": self.progress, "device": self.device,
                "slow": self.slow, "busy": self.job_lock.locked(), "idle_left": idle_left,
                "idle_timeout": self.s.idle_timeout_s, "gpus": gpu.gpus()}

    # ---- lifecycle ----------------------------------------------------------------------------------------
    async def ensure_loaded(self, on_progress=None) -> None:
        async with self.load_lock:
            if self.state == "ready" and self.proc and self.proc.is_alive():
                return
            pick = gpu.pick_device(self.s.device, self.s.need_mib)
            if not pick["ok"]:
                self.state, self.msg = "error", pick["msg"]
                raise RuntimeError(pick["msg"])
            self.device, self.slow = pick["device"], pick["slow"]
            self.state, self.msg = "loading", pick["msg"]
            self.progress = {"frac": 0.01, "msg": "Booting the neuron zoo"}
            if on_progress:
                await on_progress(self.progress)
            self.jobs, self.events, self.cancel = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Event()
            self.ctl = self.ctx.Queue()  # commands for a running job (the debugger's pause / step / rewire)
            cfg = {"model": self.s.model, "device": pick["device"]}
            self.proc = self.ctx.Process(target=worker.main, args=(cfg, self.jobs, self.events, self.cancel, self.ctl),
                                         daemon=True, name="qwen-neurons-model")
            try:
                self.proc.start()
            except Exception as e:  # noqa: BLE001
                self.proc = None
                self.state, self.msg = "error", f"could not start the model process: {e}"
                raise RuntimeError(self.msg) from e
            while True:
                ev = await self._next_event()
                if ev is None:
                    continue
                if ev["t"] == "progress":
                    self.progress = {"frac": ev["frac"], "msg": ev["msg"]}
                    if on_progress:
                        await on_progress(self.progress)
                elif ev["t"] == "ready":
                    self.state, self.progress = "ready", {"frac": 1.0, "msg": "Awake"}
                    self.msg = f"{pick['msg']}; loaded in {ev['load_s']}s"
                    self.last_used = time.time()
                    return
                elif ev["t"] in ("load_error", "dead"):
                    msg = ev.get("msg", "the model process died while loading")
                    if "out of memory" in msg.lower() and pick.get("gpu_index") is not None:
                        g = next((g for g in gpu.gpus() if g["index"] == pick["gpu_index"]), None)
                        if g:
                            msg = gpu.too_full(g, self.s.need_mib)
                    await self.unload()
                    self.state, self.msg = "error", msg
                    raise RuntimeError(msg)

    async def _next_event(self, timeout: float = 0.5):
        try:
            return await asyncio.to_thread(self.events.get, True, timeout)
        except queue.Empty:
            if self.proc is None or not self.proc.is_alive():
                return {"t": "dead", "msg": f"the model process exited (code {getattr(self.proc, 'exitcode', '?')})"}
            return None

    async def unload(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.pid is not None:
            if self.cancel is not None:
                self.cancel.set()
            try:
                self.jobs.put(None)
            except (OSError, ValueError):
                pass
            await asyncio.to_thread(p.join, 5)
            if p.is_alive():
                p.terminate()
                await asyncio.to_thread(p.join, 5)
            if p.is_alive():
                p.kill()
        self.state, self.msg, self.device = "asleep", "Model unloaded, VRAM is free", None
        self.progress = {"frac": 0.0, "msg": ""}

    async def idle_watch(self) -> None:
        while True:
            await asyncio.sleep(10)
            if (self.state == "ready" and not self.job_lock.locked()
                    and time.time() - self.last_used > self.s.idle_timeout_s):
                await self.unload()
                self.msg = f"Fell asleep after {self.s.idle_timeout_s}s idle (VRAM freed)"

    def control(self, cmd: dict) -> bool:
        """Send a command to the running job (only the debugger listens). False if nothing is running."""
        if self.ctl is None or not self.job_lock.locked():
            return False
        self.ctl.put(cmd)
        self.last_used = time.time()
        return True

    def stop(self) -> None:
        if self.cancel is not None:
            self.cancel.set()

    # ---- jobs ---------------------------------------------------------------------------------------------
    async def run(self, job: dict):
        """Async generator of events for one job. Loads the model first if needed (with progress events)."""
        if self.job_lock.locked():
            raise ModelBusy("Qwen is busy with another reply; stop it first or wait a moment.")
        async with self.job_lock:
            if self.state != "ready" or not self.proc or not self.proc.is_alive():
                pending = []

                async def on_progress(p):
                    pending.append({"t": "progress", **p})
                load = asyncio.create_task(self.ensure_loaded(on_progress))
                while not load.done():
                    await asyncio.sleep(0.25)
                    while pending:
                        yield pending.pop(0)
                while pending:
                    yield pending.pop(0)
                try:
                    load.result()
                except RuntimeError as e:
                    yield {"t": "error", "msg": str(e), "fatal": True}
                    return
                yield {"t": "loaded", "device": self.device, "slow": self.slow}
            jid = next(self.ids)
            job = {**job, "id": jid}
            self.last_used = time.time()
            self.jobs.put(job)
            try:
                while True:
                    ev = await self._next_event()
                    if ev is None:
                        continue
                    if ev["t"] == "dead":
                        await self.unload()
                        yield {"t": "error", "msg": ev["msg"], "fatal": True}
                        return
                    if ev.get("job") not in (jid, None):
                        continue  # a stale event from a cancelled job
                    if ev["t"] == "done":
                        return
                    yield ev
            except (GeneratorExit, asyncio.CancelledError):
                self.stop()  # the browser went away: don't keep generating for nobody
                raise
            finally:
                self.last_used = time.time()
