from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from .config import APP_DIR, Settings
from .engine import Engine
from .retrieval import parse_document, seed_samples
from .schemas import ApprovalDecision, MessageRequest, SearchRequest, TicketStatus
from .store import Conflict, Store


def create_app(settings: Settings | None = None, model=None):
    settings = settings or Settings.load()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    secret_file = settings.data_dir / "session-secret.key"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding="ascii")

    @asynccontextmanager
    async def lifespan(app):
        store = Store(settings.data_dir / "workspace.sqlite")
        store.recover()
        engine = Engine(settings, store, model=model)
        app.state.store = store
        app.state.engine = engine
        yield
        await run_in_threadpool(engine.close)

    app = FastAPI(title="知行 · 知识与工单助手", version="0.3.0", lifespan=lifespan)
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_file.read_text(encoding="ascii"),
        session_cookie="knowledge_desk_session",
        same_site="strict",
        max_age=30 * 86400,
    )
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

    def owner(request):
        if "owner" not in request.session:
            request.session["owner"] = uuid.uuid4().hex
        return request.session["owner"]

    @app.exception_handler(Conflict)
    async def conflict_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing_error(request, error):
        return JSONResponse({"detail": str(error.args[0])}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.get("/")
    def index():
        return FileResponse(APP_DIR / "static" / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap(request: Request):
        user = owner(request)
        if not app.state.store.documents(user) and not app.state.store.sessions(user):
            seed_samples(app.state.store, user)
        return {
            "model": settings.model,
            "offline": settings.offline,
            "configured": bool(settings.api_key) or settings.offline,
            "retrieval_mode": settings.retrieval_mode
            if app.state.engine.retriever.embeddings
            else "keyword",
            "embedding_model": settings.embedding_model,
            "sessions": app.state.store.sessions(user),
            "documents": app.state.store.documents(user),
            "tickets": app.state.store.tickets(user),
        }

    @app.get("/api/health")
    def health():
        return {"status": "ready", "model": settings.model, "offline": settings.offline}

    @app.post("/api/sessions")
    def new_session(request: Request):
        return app.state.store.create_session(owner(request))

    @app.get("/api/sessions/{session_id}")
    def session(request: Request, session_id: str):
        return app.state.engine.snapshot(owner(request), session_id)

    @app.post("/api/sessions/{session_id}/messages", status_code=202)
    def message(request: Request, session_id: str, body: MessageRequest):
        user = owner(request)
        run, created = app.state.store.create_run(user, session_id, body)
        if created:
            app.state.engine.submit(run["id"], user)
        return run

    @app.get("/api/runs")
    def runs(request: Request):
        return app.state.store.runs(owner(request))

    @app.get("/api/runs/{run_id}")
    def run(request: Request, run_id: str):
        return app.state.store.run(owner(request), run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(request: Request, run_id: str, after: int = 0):
        user = owner(request)
        app.state.store.run(user, run_id)
        try:
            reconnect_cursor = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            reconnect_cursor = 0

        async def generate():
            cursor = max(after, reconnect_cursor, 0)
            ticks = 0
            while not await request.is_disconnected():
                rows = await run_in_threadpool(app.state.store.events, run_id, cursor)
                for event in rows:
                    cursor = event["id"]
                    yield f"id: {cursor}\nevent: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                current = await run_in_threadpool(app.state.store.run, user, run_id)
                if current["status"] not in ("queued", "running"):
                    # Read once more after seeing terminal status to include the final event.
                    for event in await run_in_threadpool(
                        app.state.store.events, run_id, cursor
                    ):
                        cursor = event["id"]
                        yield f"id: {cursor}\nevent: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield (
                        "event: done\ndata: "
                        + json.dumps(current, ensure_ascii=False)
                        + "\n\n"
                    )
                    break
                ticks += 1
                if ticks % 20 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/decision", status_code=202)
    def decision(request: Request, run_id: str, body: ApprovalDecision):
        user = owner(request)
        if app.state.store.claim_resume(user, run_id, body.model_dump()):
            app.state.engine.submit(run_id, user, resume=True)
        return app.state.store.run(user, run_id)

    @app.post("/api/runs/{run_id}/retry", status_code=202)
    def retry(request: Request, run_id: str):
        user = owner(request)
        if app.state.store.claim_resume(user, run_id):
            app.state.engine.submit(run_id, user, resume=True)
        return app.state.store.run(user, run_id)

    @app.post("/api/documents", status_code=201)
    async def upload(request: Request, file: UploadFile):
        user = owner(request)
        content = await file.read(5 * 1024 * 1024 + 1)
        await file.close()
        name, kind, text, chunks = await run_in_threadpool(
            parse_document, file.filename or "", content
        )
        doc, created = await run_in_threadpool(
            app.state.store.add_document, user, name, kind, text, chunks
        )
        return {"document": doc, "created": created}

    @app.get("/api/documents")
    def documents(request: Request):
        return app.state.store.documents(owner(request))

    @app.post("/api/search")
    def preview_search(request: Request, body: SearchRequest):
        query = body.query.strip()
        if not query:
            raise ValueError("请输入检索问题。")
        result = app.state.engine.retriever.search(owner(request), query, limit=3)
        return {
            **result.metadata(),
            "sources": result.sources,
            "baseline": result.baseline,
        }

    @app.delete("/api/documents/{doc_id}")
    def delete_document(request: Request, doc_id: str):
        app.state.store.delete_document(owner(request), doc_id)
        return {"deleted": True}

    @app.get("/api/sources/{chunk_id}")
    def source(request: Request, chunk_id: str):
        return app.state.store.chunk(owner(request), chunk_id)

    @app.get("/api/tickets")
    def tickets(request: Request):
        return app.state.store.tickets(owner(request))

    @app.patch("/api/tickets/{ticket_id}")
    def update_ticket(request: Request, ticket_id: str, body: TicketStatus):
        app.state.store.update_ticket(owner(request), ticket_id, body.status)
        return {"updated": True}

    return app
