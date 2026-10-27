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

from . import discovery, genetics, scoring
from .config import Settings
from .creatures import Zoo, ancestry_node, clean
from .discovery import GenePool, SetStore
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
    use_pool: bool = True


class DiscoverReq(BaseModel):
    a: str | list[str]                      # a set id, or the replies themselves
    b: str | list[str] = "normal"
    layers: list[int] = Field(default_factory=list)
    metric: str = Field("writer", pattern="^(d|diff|resid|writer)$")
    top_k: int = Field(30, ge=1, le=200)
    per_layer_k: int = Field(5, ge=1, le=50)
    dose: float = Field(0.5, gt=0, le=2)


class Judge(BaseModel):
    kind: str = Field("model", pattern="^(model|keywords|caps)$")
    keywords: list[str] = Field(default_factory=list, max_length=20)


class EvaluateReq(BaseModel):
    genes: list[Gene] = Field(min_length=1, max_length=8)
    trait: str = Field("", max_length=200)
    judge: Judge = Field(default_factory=Judge)
    n: int = Field(20, ge=2, le=len(scoring.EVAL_QUESTIONS))
    controls: int = Field(4, ge=0, le=8)
    max_new_tokens: int = Field(120, ge=16, le=400)
    temperature: float = Field(0.7, ge=0, le=2)
    seed: int = 0


class ScreenReq(BaseModel):
    candidates: list[Gene] = Field(min_length=1, max_length=8)
    trait: str = Field("", max_length=200)
    judge: Judge = Field(default_factory=Judge)
    n: int = Field(3, ge=1, le=len(scoring.SCREEN_QUESTIONS))
    max_new_tokens: int = Field(90, ge=16, le=300)
    seed: int = 0


def sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


def create_app(s: Settings | None = None) -> FastAPI:
    s = s or Settings.load()
    engine = Engine(s)
    zoo = Zoo(s.data_dir)
    sets = SetStore(s.data_dir)
    pool = GenePool(s.data_dir)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(engine.idle_watch())
        yield
        task.cancel()
        await engine.unload()

    app = FastAPI(title="qwen-neurons", lifespan=lifespan)
    app.state.engine, app.state.zoo, app.state.sets, app.state.pool = engine, zoo, sets, pool

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

    def run_job(job: dict) -> StreamingResponse:
        if engine.job_lock.locked():
            raise HTTPException(409, "Qwen is busy with another job; stop it first or wait a moment.")

        async def stream():
            try:
                async for ev in engine.run(job):
                    yield sse(ev)
            except ModelBusy as e:
                yield sse({"t": "error", "msg": str(e)})
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/generate")
    async def generate(req: GenerateReq):
        return run_job({"kind": "generate", "messages": [m.model_dump() for m in req.messages],
                        "rows": [[g.model_dump() for g in r if g.dose != 0] for r in req.rows], "think": req.think,
                        "temperature": req.temperature, "max_new_tokens": req.max_new_tokens, "seed": req.seed})

    def texts_of(x: str | list[str], side: str) -> list[str]:
        if isinstance(x, str):
            found = sets.get(x)
            if not found:
                raise HTTPException(404, f"no prompt set {x!r}")
            return found["texts"]
        try:
            return discovery.clean_set({"name": side, "texts": x})["texts"]
        except ValueError as e:
            raise HTTPException(400, f"set {side}: {e}")

    @app.post("/api/discover")
    async def discover(req: DiscoverReq):
        a, b = texts_of(req.a, "A"), texts_of(req.b, "B")
        if any(not 0 <= L <= 31 for L in req.layers):
            raise HTTPException(422, "layers are 0-31")
        return run_job({"kind": "discover", "a": a, "b": b, "questions": discovery.NEUTRAL_QUESTIONS,
                        "layers": sorted(set(req.layers)), "metric": req.metric, "top_k": req.top_k,
                        "per_layer_k": req.per_layer_k, "dose": req.dose})

    def check_judge(judge: Judge, trait: str) -> None:
        if judge.kind == "model" and not trait.strip():
            raise HTTPException(400, "the model judge needs a trait to look for")
        if judge.kind == "keywords" and not any(k.strip() for k in judge.keywords):
            raise HTTPException(400, "the keyword rubric needs at least one keyword")

    @app.post("/api/evaluate")
    async def evaluate(req: EvaluateReq):
        genes = [g.model_dump() for g in req.genes if g.dose != 0]
        if not genes:
            raise HTTPException(400, "all doses are 0: nothing to evaluate")
        check_judge(req.judge, req.trait)
        return run_job({"kind": "evaluate", "genes": genes, "trait": req.trait.strip(), "judge": req.judge.model_dump(),
                        "questions": scoring.EVAL_QUESTIONS[:req.n], "controls": req.controls,
                        "max_new_tokens": req.max_new_tokens, "temperature": req.temperature, "seed": req.seed})

    @app.post("/api/screen")
    async def screen(req: ScreenReq):
        check_judge(req.judge, req.trait)
        return run_job({"kind": "screen", "candidates": [[g.model_dump()] for g in req.candidates],
                        "trait": req.trait.strip(), "judge": req.judge.model_dump(),
                        "questions": scoring.SCREEN_QUESTIONS[:req.n], "max_new_tokens": req.max_new_tokens,
                        "temperature": 0.7, "seed": req.seed})

    @app.get("/api/sets")
    def list_sets():
        return sets.all()

    @app.post("/api/sets")
    def save_set(body: dict):
        try:
            return sets.save(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except PermissionError as e:
            raise HTTPException(403, str(e))

    @app.delete("/api/sets/{sid}")
    def delete_set(sid: str):
        if any(b["id"] == sid for b in discovery.BUILTIN_SETS):
            raise HTTPException(403, "built-in sets stay")
        if not sets.delete(sid):
            raise HTTPException(404, "no such set")
        return {"ok": True}

    @app.get("/api/genepool")
    def genepool():
        return pool.all()

    @app.post("/api/genepool")
    def genepool_add(genes: list[dict]):
        try:
            return pool.add(genes)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/genepool/{layer}/{neuron}")
    def genepool_remove(layer: int, neuron: int):
        return pool.remove(layer, neuron)

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
        saved = {c["id"] for c in zoo.all()}
        for c, raw in ((a, req.a), (b, req.b)):
            c["id"] = raw.get("id") if raw.get("id") in saved else None  # unsaved parents live on in the snapshot
            c["builtin"] = bool(raw.get("builtin")) and c["id"] is not None
        rng = random.Random(req.seed)
        child = genetics.breed(a, b, rng, req.mutation, pool.all() if req.use_pool else None)
        child["ancestry"] = [ancestry_node(c, zoo, 5) for c in (a, b)]
        return child

    return app
