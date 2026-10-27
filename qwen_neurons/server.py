"""The web app: static frontend + JSON/SSE API."""
import asyncio
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import genetics
from .config import Settings
from .creatures import Zoo, clean
from .engine import Engine, ModelBusy
from .steering import LOOP_ZONE, WARM_ZONE

STATIC = Path(__file__).parent / "static"


class Gene(BaseModel):
    layer: int = Field(ge=0, le=31)
    neuron: int = Field(ge=0, le=9215)
    dose: float = Field(ge=-2, le=2)


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str = Field(max_length=20000)


class GenerateReq(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=60)
    rows: list[list[Gene]] = Field(min_length=1, max_length=4)
    think: bool = True
    temperature: float = Field(0.7, ge=0, le=2)
    max_new_tokens: int = Field(700, ge=1, le=2000)
    seed: int | None = None


class BreedReq(BaseModel):
    a: dict
    b: dict
    seed: int | None = None
    mutation: float = Field(0.3, ge=0, le=1)


def sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


def create_app(s: Settings | None = None) -> FastAPI:
    s = s or Settings.load()
    engine = Engine(s)
    zoo = Zoo(s.data_dir)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(engine.idle_watch())
        yield
        task.cancel()
        await engine.unload()

    app = FastAPI(title="qwen-neurons", lifespan=lifespan)
    app.state.engine, app.state.zoo = engine, zoo

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/status")
    def status():
        return {**engine.status(), "zones": {"warm": WARM_ZONE, "loop": LOOP_ZONE, "child_cap": genetics.CHILD_CAP}}

    @app.post("/api/wake")
    async def wake():
        async def stream():
            q: asyncio.Queue = asyncio.Queue()

            async def on_progress(p):
                await q.put({"t": "progress", **p})
            task = asyncio.create_task(engine.ensure_loaded(on_progress))
            while not task.done() or not q.empty():
                try:
                    yield sse(await asyncio.wait_for(q.get(), 0.3))
                except asyncio.TimeoutError:
                    pass
            try:
                task.result()
                yield sse({"t": "loaded", "device": engine.device, "slow": engine.slow})
            except RuntimeError as e:
                yield sse({"t": "error", "msg": str(e), "fatal": True})
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/sleep")
    async def sleep():
        engine.stop()
        await engine.unload()
        return engine.status()

    @app.post("/api/stop")
    def stop():
        engine.stop()
        return {"ok": True}

    @app.post("/api/generate")
    async def generate(req: GenerateReq):
        if engine.job_lock.locked():
            raise HTTPException(409, "Qwen is busy with another reply; stop it first or wait a moment.")
        job = {"kind": "generate", "messages": [m.model_dump() for m in req.messages],
               "rows": [[g.model_dump() for g in r if g.dose != 0] for r in req.rows], "think": req.think,
               "temperature": req.temperature, "max_new_tokens": req.max_new_tokens, "seed": req.seed}

        async def stream():
            try:
                async for ev in engine.run(job):
                    yield sse(ev)
            except ModelBusy as e:
                yield sse({"t": "error", "msg": str(e)})
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/creatures")
    def creatures():
        return zoo.all()

    @app.post("/api/creatures")
    def save_creature(c: dict):
        try:
            return zoo.save(c)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        except PermissionError as e:
            raise HTTPException(403, str(e))

    @app.delete("/api/creatures/{cid}")
    def delete_creature(cid: str):
        if any(b["id"] == cid for b in zoo.all() if b.get("builtin")):
            raise HTTPException(403, "built-in creatures stay in the zoo")
        if not zoo.delete(cid):
            raise HTTPException(404, "no such creature")
        return {"ok": True}

    @app.post("/api/breed")
    def breed(req: BreedReq):
        try:
            a, b = clean(req.a), clean(req.b)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        a["id"], b["id"] = req.a.get("id"), req.b.get("id")
        rng = random.Random(req.seed)
        return genetics.breed(a, b, rng, req.mutation)

    return app
