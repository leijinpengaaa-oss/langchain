from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import TicketDraft


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    digest TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(owner,digest)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    heading TEXT NOT NULL, page INTEGER, text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle', current_run TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
                    request_key TEXT NOT NULL, question TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, decision TEXT, error TEXT, started_at TEXT NOT NULL,
                    finished_at TEXT, tokens INTEGER NOT NULL DEFAULT 0,
                    model_calls INTEGER NOT NULL DEFAULT 0, model_ms REAL NOT NULL DEFAULT 0,
                    UNIQUE(session_id,request_key)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id,id);
                CREATE TABLE IF NOT EXISTS vectors (
                    owner TEXT NOT NULL, namespace TEXT NOT NULL, kind TEXT NOT NULL,
                    cache_key TEXT NOT NULL, digest TEXT NOT NULL,
                    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
                    vector TEXT NOT NULL, used_at TEXT NOT NULL,
                    PRIMARY KEY(owner,namespace,kind,cache_key)
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
                    operation_id TEXT NOT NULL UNIQUE REFERENCES runs(id), title TEXT NOT NULL,
                    description TEXT NOT NULL, priority TEXT NOT NULL, category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ranking_cache (
                    owner TEXT NOT NULL, cache_key TEXT NOT NULL,
                    ratings TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(owner,cache_key)
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def recover(self):
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status='error',error='服务重启，请从检查点继续任务。' WHERE status IN ('running','queued')"
            )
            db.execute(
                "UPDATE sessions SET status='error' WHERE current_run IN (SELECT id FROM runs WHERE status='error')"
            )

    def add_document(
        self, owner: str, name: str, kind: str, text: str, chunks: list[dict]
    ):
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.connection() as db:
            existing = db.execute(
                "SELECT id FROM documents WHERE owner=? AND digest=?", (owner, digest)
            ).fetchone()
            if existing:
                return self.document(owner, existing["id"]), False
            if (
                db.execute(
                    "SELECT COUNT(*) FROM documents WHERE owner=?", (owner,)
                ).fetchone()[0]
                >= 50
            ):
                raise ValueError("本地版本最多保存50份文档，请先删除不需要的文档。")
            doc_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                (doc_id, owner, name[:160], digest, kind, now()),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                [
                    (f"{doc_id}-{i}", doc_id, c["heading"], c.get("page"), c["text"])
                    for i, c in enumerate(chunks)
                ],
            )
        return self.document(owner, doc_id), True

    def documents(self, owner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT d.id,d.name,d.kind,d.created_at,COUNT(c.id) AS chunk_count FROM documents d LEFT JOIN chunks c ON c.document_id=d.id WHERE d.owner=? GROUP BY d.id ORDER BY d.created_at DESC",
                    (owner,),
                )
            ]

    def document(self, owner, doc_id):
        result = next((d for d in self.documents(owner) if d["id"] == doc_id), None)
        if result is None:
            raise KeyError("文档不存在。")
        return result

    def delete_document(self, owner, doc_id):
        with self.connection() as db:
            if not db.execute(
                "DELETE FROM documents WHERE id=? AND owner=?", (doc_id, owner)
            ).rowcount:
                raise KeyError("文档不存在。")
            db.execute("DELETE FROM ranking_cache WHERE owner=?", (owner,))

    def chunks(self, owner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT c.*,d.name FROM chunks c JOIN documents d ON c.document_id=d.id WHERE d.owner=?",
                    (owner,),
                )
            ]

    def chunk(self, owner, chunk_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT c.*,d.name FROM chunks c JOIN documents d ON c.document_id=d.id WHERE c.id=? AND d.owner=?",
                (chunk_id, owner),
            ).fetchone()
            if row is None:
                raise KeyError("引用来源不存在或已被删除。")
            return dict(row)

    def create_session(self, owner, title="新对话"):
        session_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO sessions(id,owner,title,created_at,updated_at) VALUES (?,?,?,?,?)",
                (session_id, owner, title, now(), now()),
            )
        return self.session(owner, session_id)

    def sessions(self, owner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM sessions WHERE owner=? ORDER BY updated_at DESC",
                    (owner,),
                )
            ]

    def session(self, owner, session_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE owner=? AND id=?", (owner, session_id)
            ).fetchone()
            if row is None:
                raise KeyError("会话不存在。")
            return dict(row)

    def create_run(self, owner, session_id, request):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            session = db.execute(
                "SELECT * FROM sessions WHERE id=? AND owner=?", (session_id, owner)
            ).fetchone()
            if session is None:
                raise KeyError("会话不存在。")
            existing = db.execute(
                "SELECT * FROM runs WHERE session_id=? AND request_key=?",
                (session_id, request.request_id),
            ).fetchone()
            if existing:
                if (
                    existing["question"] != request.question
                    or existing["mode"] != request.mode
                ):
                    raise Conflict("同一请求编号不能对应不同内容。")
                return dict(existing), False
            if session["status"] in ("running", "awaiting_approval", "error"):
                raise Conflict("请先完成、审批或恢复当前任务，再发送新问题。")
            run_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO runs(id,owner,session_id,request_key,question,mode,status,started_at) VALUES (?,?,?,?,?,?,'queued',?)",
                (
                    run_id,
                    owner,
                    session_id,
                    request.request_id,
                    request.question,
                    request.mode,
                    now(),
                ),
            )
            db.execute(
                "UPDATE sessions SET status='running',current_run=?,updated_at=?,title=CASE WHEN title='新对话' THEN ? ELSE title END WHERE id=?",
                (run_id, now(), request.question[:36], session_id),
            )
        return self.run(owner, run_id), True

    def run(self, owner, run_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM runs WHERE id=? AND owner=?", (run_id, owner)
            ).fetchone()
            if row is None:
                raise KeyError("任务不存在。")
            result = dict(row)
            result["decision"] = (
                json.loads(result["decision"]) if result["decision"] else None
            )
            return result

    def runs(self, owner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,session_id,question,status,started_at,finished_at,tokens,model_calls,model_ms,error FROM runs WHERE owner=? ORDER BY started_at DESC LIMIT 100",
                    (owner,),
                )
            ]

    def set_status(self, run_id, status, error=None):
        session_status = {
            "completed": "idle",
            "cancelled": "idle",
            "error": "error",
        }.get(status, status)
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET status=?,error=?,finished_at=? WHERE id=?",
                (
                    status,
                    error,
                    now() if status in ("completed", "cancelled", "error") else None,
                    run_id,
                ),
            )
            db.execute(
                "UPDATE sessions SET status=?,updated_at=? WHERE current_run=?",
                (session_status, now(), run_id),
            )

    def claim_resume(self, owner, run_id, decision=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT r.*,s.current_run FROM runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=? AND r.owner=?",
                (run_id, owner),
            ).fetchone()
            if row is None:
                raise KeyError("任务不存在。")
            if row["current_run"] != run_id:
                raise Conflict("只能恢复会话的当前任务。")
            encoded = (
                json.dumps(decision, ensure_ascii=False, sort_keys=True)
                if decision
                else None
            )
            if decision and row["decision"]:
                if encoded != row["decision"]:
                    raise Conflict("此任务已经提交过不同的审批决定。")
                if row["status"] != "error":
                    return False
            expected = "awaiting_approval" if decision else "error"
            if row["status"] != expected:
                raise Conflict("当前任务状态不支持此操作。")
            db.execute(
                "UPDATE runs SET status='queued',error=NULL,finished_at=NULL,decision=COALESCE(?,decision) WHERE id=?",
                (encoded, run_id),
            )
            db.execute(
                "UPDATE sessions SET status='running',updated_at=? WHERE id=?",
                (now(), row["session_id"]),
            )
            return True

    def event(self, run_id, kind, payload):
        with self.connection() as db:
            db.execute(
                "INSERT INTO events(run_id,kind,payload,created_at) VALUES (?,?,?,?)",
                (run_id, kind, json.dumps(payload, ensure_ascii=False), now()),
            )

    def events(self, run_id, after=0):
        with self.connection() as db:
            return [
                {**dict(r), "payload": json.loads(r["payload"])}
                for r in db.execute(
                    "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id",
                    (run_id, after),
                )
            ]

    def usage(self, run_id, tokens, elapsed_ms):
        with self.connection() as db:
            db.execute(
                "UPDATE runs SET tokens=tokens+?,model_calls=model_calls+1,model_ms=model_ms+? WHERE id=?",
                (tokens, elapsed_ms, run_id),
            )

    def vectors(self, owner, namespace, kind):
        with self.connection() as db:
            result = {}
            for row in db.execute(
                "SELECT cache_key,digest,vector FROM vectors WHERE owner=? AND namespace=? AND kind=?",
                (owner, namespace, kind),
            ):
                try:
                    vector = json.loads(row["vector"])
                except (TypeError, ValueError):
                    continue
                result[row["cache_key"]] = {"digest": row["digest"], "vector": vector}
            return result

    def save_vectors(self, owner, namespace, kind, records):
        if kind not in {"document", "query"}:
            raise ValueError("向量缓存类型无效。")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for record in records:
                if kind == "document":
                    chunk = db.execute(
                        "SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=? AND d.owner=?",
                        (record["key"], owner),
                    ).fetchone()
                    if chunk is None:
                        continue
                db.execute(
                    "INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?,?,?)",
                    (
                        owner,
                        namespace,
                        kind,
                        record["key"],
                        record["digest"],
                        record["key"] if kind == "document" else None,
                        json.dumps(record["vector"], allow_nan=False),
                        now(),
                    ),
                )
            if kind == "query":
                db.execute(
                    "DELETE FROM vectors WHERE owner=? AND namespace=? AND kind='query' AND cache_key NOT IN (SELECT cache_key FROM vectors WHERE owner=? AND namespace=? AND kind='query' ORDER BY used_at DESC LIMIT 256)",
                    (owner, namespace, owner, namespace),
                )

    def create_ticket(self, owner, session_id, operation_id, draft: TicketDraft):
        ticket_id = "TK-" + uuid.uuid4().hex[:8].upper()
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO tickets(id,owner,session_id,operation_id,title,description,priority,category,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ticket_id,
                    owner,
                    session_id,
                    operation_id,
                    draft.title,
                    draft.description,
                    draft.priority,
                    draft.category,
                    now(),
                ),
            )
            return dict(
                db.execute(
                    "SELECT * FROM tickets WHERE operation_id=? AND owner=?",
                    (operation_id, owner),
                ).fetchone()
            )

    def cached_ranking(self, owner, key):
        with self.connection() as db:
            row = db.execute(
                "SELECT ratings FROM ranking_cache WHERE owner=? AND cache_key=?",
                (owner, key),
            ).fetchone()
            try:
                return json.loads(row[0]) if row else None
            except (TypeError, ValueError):
                return None

    def save_ranking(self, owner, key, ratings):
        with self.connection() as db:
            db.execute(
                "INSERT OR REPLACE INTO ranking_cache VALUES (?,?,?,?)",
                (owner, key, json.dumps(ratings, allow_nan=False), now()),
            )
            db.execute(
                "DELETE FROM ranking_cache WHERE owner=? AND cache_key NOT IN (SELECT cache_key FROM ranking_cache WHERE owner=? ORDER BY created_at DESC LIMIT 256)",
                (owner, owner),
            )

    def tickets(self, owner):
        with self.connection() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM tickets WHERE owner=? ORDER BY created_at DESC",
                    (owner,),
                )
            ]

    def update_ticket(self, owner, ticket_id, status):
        if status not in ("open", "in_progress", "resolved"):
            raise ValueError("工单状态无效。")
        with self.connection() as db:
            if not db.execute(
                "UPDATE tickets SET status=? WHERE id=? AND owner=?",
                (status, ticket_id, owner),
            ).rowcount:
                raise KeyError("工单不存在。")
