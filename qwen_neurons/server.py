"""The web app: static frontend + JSON/SSE API."""
import asyncio
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi import Path as PathArg
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agent, discovery, genetics, scoring
from .config import Settings
from .creatures import Zoo, ancestry_node, clean
from .discovery import GenePool, SetStore
from .engine import Engine, ModelBusy
from .steering import LOOP_ZONE, WARM_ZONE
from .weights import WeightLab

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
    rows: list[list[Gene]] = Field(min_length=1, max_length=6)
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


class TestReq(BaseModel):
    creatures: list[list[Gene]] = Field(min_length=1, max_length=8)
    n: int = Field(3, ge=1, le=len(scoring.TEST_QUESTIONS))
    max_new_tokens: int = Field(100, ge=16, le=300)
    seed: int = 0
    describe: bool = True
    rate: bool = True
    tame: bool = True


class ScreenReq(BaseModel):
    candidates: list[Gene] = Field(min_length=1, max_length=8)
    trait: str = Field("", max_length=200)
    judge: Judge = Field(default_factory=Judge)
    n: int = Field(3, ge=1, le=len(scoring.SCREEN_QUESTIONS))
    max_new_tokens: int = Field(90, ge=16, le=300)
    seed: int = 0


class Clamp(BaseModel):
    layer: int = Field(ge=0, le=31)
    neuron: int = Field(ge=0, le=9215)
    value: float = Field(ge=-100, le=100)


class Pin(BaseModel):
    layer: int = Field(ge=0, le=31)
    neuron: int = Field(ge=0, le=9215)


class Breakpoints(BaseModel):
    tokens: list[int] = Field(default_factory=list, max_length=50)
    text: str = Field("", max_length=40)


class DebugReq(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)
    knobs: list[Gene] = Field(default_factory=list, max_length=8)
    clamps: list[Clamp] = Field(default_factory=list, max_length=16)
    pins: list[Pin] = Field(default_factory=list, max_length=16)
    breakpoints: Breakpoints = Field(default_factory=Breakpoints)
    step: bool = False
    max_new_tokens: int = Field(160, ge=1, le=600)
    temperature: float = Field(0.7, ge=0, le=2)
    top: int = Field(6, ge=1, le=12)


class DebugCtl(BaseModel):
    cmd: str = Field(pattern="^(pause|step|continue|stop|update)$")
    knobs: list[Gene] | None = Field(None, max_length=8)
    clamps: list[Clamp] | None = Field(None, max_length=16)
    pins: list[Pin] | None = Field(None, max_length=16)
    breakpoints: Breakpoints | None = None


LayerArg = Annotated[int, PathArg(ge=0, le=31)]
NeuronArg = Annotated[int, PathArg(ge=0, le=9215)]


class NeuronRef(BaseModel):
    layer: int = Field(ge=0, le=31)
    neuron: int = Field(ge=0, le=9215)


class ProbeReq(BaseModel):
    genes: list[Gene] = Field(min_length=1, max_length=6)
    mode: str = Field("steer", pattern="^(steer|ablate)$")


class FiresTextReq(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=8)
    neurons: list[NeuronRef] = Field(min_length=1, max_length=6)


class LogprobRow(BaseModel):
    label: str = Field("", max_length=60)
    genes: list[Gene] = Field(default_factory=list, max_length=6)
    ablate: list[NeuronRef] = Field(default_factory=list, max_length=6)


class LogprobsReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    chat: bool = True
    continuations: list[str] = Field(min_length=1, max_length=8)
    rows: list[LogprobRow] = Field(min_length=1, max_length=4)


class FiresBuildReq(BaseModel):
    n_docs: int = Field(4096, ge=16, le=20000)
    max_tokens: int = Field(64, ge=16, le=256)


class AgentReq(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    max_steps: int = Field(agent.MAX_STEPS, ge=2, le=30)


def sse(ev: dict) -> str:
    return f"data: {json.dumps(ev)}\n\n"


def create_app(s: Settings | None = None) -> FastAPI:
    s = s or Settings.load()
    engine = Engine(s)
    zoo = Zoo(s.data_dir)
    sets = SetStore(s.data_dir)
    pool = GenePool(s.data_dir)
    weights = WeightLab(s.model, s.data_dir, s.idle_timeout_s)
    agent_cfg = agent.AgentConfig.from_env(s.agent_env)
    agent_run = {"cancel": False, "running": False}

    @asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(engine.idle_watch()), asyncio.create_task(weights.idle_watch())]
        yield
        for t in tasks:
            t.cancel()
        await engine.unload()
        await weights.stop()

    app = FastAPI(title="qwen-neurons", lifespan=lifespan)
    app.state.engine, app.state.zoo, app.state.sets, app.state.pool = engine, zoo, sets, pool
    app.state.weights, app.state.agent_cfg, app.state.brain = weights, agent_cfg, None

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

    @app.post("/api/test")
    async def personality_test(req: TestReq):
        if any(not [g for g in c if g.dose != 0] for c in req.creatures):
            raise HTTPException(400, "every creature needs a gene with a dose")
        return run_job({"kind": "test", "creatures": [[g.model_dump() for g in c if g.dose != 0] for c in req.creatures],
                        "questions": scoring.TEST_QUESTIONS[:req.n], "max_new_tokens": req.max_new_tokens,
                        "temperature": 0.7, "seed": req.seed, "describe": req.describe, "rate": req.rate,
                        "tame": req.tame})

    @app.post("/api/debug")
    async def debug(req: DebugReq):
        return run_job({"kind": "debug", **req.model_dump(), "messages": [m.model_dump() for m in req.messages]})

    @app.post("/api/debug/ctl")
    def debug_ctl(req: DebugCtl):
        if not engine.control(req.model_dump(exclude_none=True)):
            raise HTTPException(409, "no debugger session is running")
        return {"ok": True}

    # ---- the neuron explorer ------------------------------------------------------------------------------
    async def wcall(op: str, **kw):
        try:
            return await weights.call(op, **kw)
        except RuntimeError as e:
            raise HTTPException(409 if "yet" in str(e) else 500, str(e))

    @app.get("/api/x/status")
    def x_status():
        d = weights.dir
        parts = len(list(d.glob("lens-L*.pt"))) if d.is_dir() else 0
        index = (d / "lens-index.pt").is_file()
        return {"index": index, "index_layers": 32 if index else parts, "fires": (d / "fires.pt").is_file(),
                "building": weights.building, "weights_process": weights.alive(), "agent": agent_cfg.public()}

    @app.post("/api/x/index")
    async def x_build_index():
        if weights.building and weights.building.get("state") == "running":
            return {"started": False, "building": weights.building}
        weights.building = {"state": "running", "frac": 0.0, "msg": "starting the weights process"}

        async def build():
            try:
                async for ev in weights.stream("build_index"):
                    if ev["t"] == "progress":
                        weights.building = {"state": "running", "frac": ev["frac"], "msg": ev["msg"]}
                    elif ev["t"] == "error":
                        weights.building = {"state": "error", "frac": 0.0, "msg": ev["msg"]}
                    elif ev["t"] == "result":
                        weights.building = {"state": "done", "frac": 1.0, "msg": "the lens index is ready"}
            except Exception as e:  # noqa: BLE001
                weights.building = {"state": "error", "frac": 0.0, "msg": str(e)}
        app.state.index_task = asyncio.create_task(build())
        return {"started": True, "building": weights.building}

    @app.get("/api/x/lens/{layer}/{neuron}")
    async def x_lens(layer: LayerArg, neuron: NeuronArg):
        return await wcall("lens", layer=layer, neuron=neuron, k=16)

    @app.get("/api/x/web/{layer}/{neuron}")
    async def x_web(layer: LayerArg, neuron: NeuronArg):
        return await wcall("web", layer=layer, neuron=neuron)

    @app.get("/api/x/fires/{layer}/{neuron}")
    async def x_fires(layer: LayerArg, neuron: NeuronArg):
        return await wcall("fires", layer=layer, neuron=neuron)

    @app.get("/api/x/card/{layer}/{neuron}")
    async def x_card(layer: LayerArg, neuron: NeuronArg):
        return await wcall("card", layer=layer, neuron=neuron)

    @app.get("/api/x/search")
    async def x_search(q: str, limit: int = 30, lo: int = 3, hi: int = 26):
        if not q.strip() or len(q) > 40:
            raise HTTPException(400, "search for a word (1-40 characters)")
        lo, hi = max(0, min(lo, 31)), max(0, min(hi, 31))
        return await wcall("search", query=q, limit=max(1, min(limit, 60)), lo=min(lo, hi), hi=max(lo, hi))

    @app.get("/api/x/map")
    async def x_map(color: str = "norm"):
        if color not in ("norm", "act"):
            raise HTTPException(400, "color is norm or act")
        return await wcall("map", color=color)

    @app.post("/api/x/probe")
    async def x_probe(req: ProbeReq):
        return run_job({"kind": "probe", "mode": req.mode, "genes": [g.model_dump() for g in req.genes]})

    @app.post("/api/x/fires-text")
    async def x_fires_text(req: FiresTextReq):
        return run_job({"kind": "fires_text", "texts": [t[:1000] for t in req.texts],
                        "neurons": [n.model_dump() for n in req.neurons]})

    @app.post("/api/x/logprobs")
    async def x_logprobs(req: LogprobsReq):
        return run_job({"kind": "logprobs", **req.model_dump()})

    @app.post("/api/x/fires-build")
    async def x_fires_build(req: FiresBuildReq):
        return run_job({"kind": "fires_build", "path": str(weights.dir / "fires.pt"), "n_docs": req.n_docs,
                        "max_tokens": req.max_tokens, "k": 8, "batch": 16})

    # ---- the Lab agent ------------------------------------------------------------------------------------
    @app.get("/api/agent")
    def agent_info():
        return {**agent_cfg.public(), "running": agent_run["running"]}

    async def job_result(job: dict) -> dict:
        """Run a model job to its end (leaving early would keep engine.run holding the job lock) -> its result."""
        result = error = None
        async for ev in engine.run(job):
            if ev["t"] == "error":
                error = error or ev["msg"]
            elif ev["t"] == "result":
                result = ev
        if error or result is None:
            raise RuntimeError(error or "the model job ended without a result")
        return result

    def brain():
        if app.state.brain is not None:          # tests put a fake here
            return app.state.brain
        if agent_cfg.provider == "qwen":
            return agent.QwenProvider(job_result)
        return agent.OpenAIProvider(agent_cfg)

    @app.post("/api/agent")
    async def agent_go(req: AgentReq):
        if agent_run["running"]:
            raise HTTPException(409, "the agent is already on a case")
        if engine.job_lock.locked():
            raise HTTPException(409, "Qwen is busy with another job; stop it first or wait a moment.")
        try:
            b = brain()
        except ValueError as e:
            raise HTTPException(400, str(e))
        tools = agent.Toolbox(weights, engine.run, zoo)

        async def stream():
            agent_run.update(running=True, cancel=False)
            try:
                async for ev in agent.run_agent(req.goal, b, tools, req.max_steps):
                    yield sse(ev)
                    if agent_run["cancel"]:
                        yield sse({"t": "agent_end", "reason": "stopped"})
                        return
            except ModelBusy as e:
                yield sse({"t": "error", "msg": str(e)})
            finally:
                agent_run["running"] = False
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/agent/stop")
    def agent_stop():
        agent_run["cancel"] = True
        engine.stop()
        return {"ok": True}

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
        known = (pool.all() if req.use_pool else []) + genetics.zoo_pool(zoo.all())
        child = genetics.breed(a, b, rng, req.mutation, known)
        child["ancestry"] = [ancestry_node(c, zoo, 5) for c in (a, b)]
        return child

    return app
