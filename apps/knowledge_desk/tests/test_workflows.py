from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from apps.knowledge_desk.config import Settings
from apps.knowledge_desk.engine import Engine
from apps.knowledge_desk.llm import ModelError, OfflineModel
from apps.knowledge_desk.retrieval import parse_document, seed_samples
from apps.knowledge_desk.schemas import ApprovalDecision, MessageRequest
from apps.knowledge_desk.store import Conflict, Store
from apps.knowledge_desk.web import create_app


@pytest.fixture
def workspace(tmp_path):
    settings = Settings(data_dir=tmp_path, offline=True)
    store = Store(tmp_path / "workspace.sqlite")
    seed_samples(store, "alice")
    engine = Engine(settings, store)
    yield settings, store, engine
    engine.close()


def run_question(
    store, engine, question, mode="auto", owner="alice", session=None, key="request01"
):
    session = session or store.create_session(owner)
    run, _ = store.create_run(
        owner,
        session["id"],
        MessageRequest(question=question, mode=mode, request_id=key),
    )
    engine.execute(owner, run["id"])
    return session, store.run(owner, run["id"])


def test_grounded_answer_cites_an_owned_source(workspace):
    _, store, engine = workspace
    session, run = run_question(store, engine, "星舟服务启动失败怎么排查？")
    assert run["status"] == "completed"
    message = engine.snapshot("alice", session["id"])["messages"][-1]
    assert "8317" in message["content"]
    assert message["citations"]
    source = store.chunk("alice", message["citations"][0]["id"])
    assert source["heading"] == "启动失败排查"
    with pytest.raises(KeyError):
        store.chunk("bob", source["id"])


def test_missing_evidence_refuses_to_invent(workspace):
    _, store, engine = workspace
    session, _ = run_question(store, engine, "法国首都的天气怎么样？")
    message = engine.snapshot("alice", session["id"])["messages"][-1]
    assert "没有找到足够依据" in message["content"]
    assert not message["citations"]


def test_ticket_requires_approval_and_accepts_edits(workspace):
    _, store, engine = workspace
    session, run = run_question(
        store, engine, "服务启动失败，帮我生成部署工单", "ticket"
    )
    assert run["status"] == "awaiting_approval"
    assert store.tickets("alice") == []
    draft = engine.snapshot("alice", session["id"])["pending"]
    draft["title"] = "人工修改后的启动故障"
    draft["priority"] = "high"
    decision = ApprovalDecision(action="approve", draft=draft).model_dump()
    assert store.claim_resume("alice", run["id"], decision)
    engine.execute("alice", run["id"], resume=True)
    ticket = store.tickets("alice")[0]
    assert ticket["title"] == draft["title"] and ticket["priority"] == "high"
    assert store.run("alice", run["id"])["status"] == "completed"
    assert not store.claim_resume("alice", run["id"], decision)
    assert len(store.tickets("alice")) == 1


def test_reject_creates_no_ticket(workspace):
    _, store, engine = workspace
    _, run = run_question(store, engine, "帮我创建服务启动失败工单", "ticket")
    store.claim_resume(
        "alice", run["id"], ApprovalDecision(action="reject").model_dump()
    )
    engine.execute("alice", run["id"], resume=True)
    assert store.run("alice", run["id"])["status"] == "cancelled"
    assert store.tickets("alice") == []


def test_pending_approval_survives_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, offline=True)
    store = Store(tmp_path / "workspace.sqlite")
    seed_samples(store, "alice")
    first = Engine(settings, store)
    session, run = run_question(store, first, "帮我创建启动失败工单", "ticket")
    first.close()
    second = Engine(settings, store)
    try:
        assert second.snapshot("alice", session["id"])["pending"]
        store.claim_resume(
            "alice", run["id"], ApprovalDecision(action="approve").model_dump()
        )
        second.execute("alice", run["id"], resume=True)
        assert len(store.tickets("alice")) == 1
        assert len(second.snapshot("alice", session["id"])["messages"]) == 2
    finally:
        second.close()


def test_crash_after_ticket_insert_does_not_duplicate(workspace, monkeypatch):
    _, store, engine = workspace
    _, run = run_question(store, engine, "帮我创建部署故障工单", "ticket")
    store.claim_resume(
        "alice", run["id"], ApprovalDecision(action="approve").model_dump()
    )
    original = store.create_ticket

    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected crash after committed insert")

    monkeypatch.setattr(store, "create_ticket", crash)
    engine.execute("alice", run["id"], resume=True)
    assert store.run("alice", run["id"])["status"] == "error"
    original_id = store.tickets("alice")[0]["id"]
    monkeypatch.setattr(store, "create_ticket", original)
    store.claim_resume("alice", run["id"])
    engine.execute("alice", run["id"], resume=True)
    assert store.run("alice", run["id"])["status"] == "completed"
    assert [t["id"] for t in store.tickets("alice")] == [original_id]


def test_request_idempotency_and_session_conflicts(workspace):
    _, store, engine = workspace
    session = store.create_session("alice")
    request = MessageRequest(
        question="创建启动失败工单", request_id="stable-key", mode="ticket"
    )
    run, created = store.create_run("alice", session["id"], request)
    assert created
    duplicate, created = store.create_run("alice", session["id"], request)
    assert not created and duplicate["id"] == run["id"]
    with pytest.raises(Conflict):
        store.create_run(
            "alice", session["id"], request.model_copy(update={"question": "different"})
        )
    with pytest.raises(Conflict):
        store.create_run(
            "alice",
            session["id"],
            request.model_copy(update={"request_id": "another-key"}),
        )
    engine.execute("alice", run["id"])
    assert store.session("alice", session["id"])["status"] == "awaiting_approval"


def test_sessions_and_users_are_isolated(workspace):
    _, store, engine = workspace
    first, _ = run_question(store, engine, "星舟默认端口是多少？")
    second, _ = run_question(store, engine, "工单响应时间？")
    assert first["id"] != second["id"]
    assert len(engine.snapshot("alice", second["id"])["messages"]) == 2
    with pytest.raises(KeyError):
        engine.snapshot("bob", first["id"])
    assert store.chunks("bob") == []


def test_conversation_persists_and_does_not_duplicate_messages(workspace):
    _, store, engine = workspace
    session, _ = run_question(store, engine, "服务启动失败？")
    run_question(
        store, engine, "帮我把刚才的问题生成工单", session=session, key="request02"
    )
    messages = engine.snapshot("alice", session["id"])["messages"]
    assert len(messages) == 3
    assert len({m["id"] for m in messages}) == len(messages)


def test_failure_can_resume_from_saved_node(workspace):
    _, store, engine = workspace

    class FailingModel(OfflineModel):
        def answer(self, state):
            raise ModelError("injected model failure")

    engine.model = FailingModel()
    session, run = run_question(store, engine, "星舟默认端口？")
    assert run["status"] == "error"
    engine.model = OfflineModel()
    store.claim_resume("alice", run["id"])
    engine.execute("alice", run["id"], resume=True)
    assert store.run("alice", run["id"])["status"] == "completed"
    assert len(engine.snapshot("alice", session["id"])["messages"]) == 2


def test_upload_validation_and_content_deduplication(workspace):
    _, store, _ = workspace
    content = "# 手册\n\n## 特殊参数\n测试参数为9071。".encode()
    name, kind, text, chunks = parse_document("../手册.md", content)
    assert name == "手册.md"
    doc, created = store.add_document("alice", name, kind, text, chunks)
    assert created
    duplicate, created = store.add_document(
        "alice", "另一个名字.md", kind, text, chunks
    )
    assert not created and duplicate["id"] == doc["id"]
    for filename, body in [
        ("file.exe", b"text"),
        ("file.txt", b"\xff"),
        ("empty.md", b""),
        ("big.txt", b"x" * (5 * 1024 * 1024 + 1)),
        ("invalid.pdf", b"not-pdf"),
    ]:
        with pytest.raises(ValueError):
            parse_document(filename, body)


def test_scan_pdf_returns_actionable_error():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="OCR"):
        parse_document("scan.pdf", buffer.getvalue())


def wait_for_run(client, run_id):
    for _ in range(100):
        result = client.get("/api/runs/" + run_id).json()
        if result["status"] not in ("running", "queued"):
            return result
        time.sleep(0.03)
    raise AssertionError("worker timed out")


def test_api_sse_ownership_upload_and_ticket_lifecycle(tmp_path):
    settings = Settings(data_dir=tmp_path, offline=True)
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 200
        assert len(client.get("/api/bootstrap").json()["documents"]) == 3
        session = client.post("/api/sessions").json()
        run = client.post(
            f"/api/sessions/{session['id']}/messages",
            json={
                "question": "帮我创建星舟启动故障工单",
                "mode": "ticket",
                "request_id": "api-request01",
            },
        ).json()
        assert wait_for_run(client, run["id"])["status"] == "awaiting_approval"
        events = client.get(f"/api/runs/{run['id']}/events").text
        assert (
            "event: trace" in events
            and "event: done" in events
            and "approval" in events
        )
        last_event_id = max(
            int(line.removeprefix("id: "))
            for line in events.splitlines()
            if line.startswith("id: ")
        )
        replay = client.get(
            f"/api/runs/{run['id']}/events",
            headers={"Last-Event-ID": str(last_event_id)},
        ).text
        assert "event: done" in replay and "event: trace" not in replay
        assert (
            client.post(
                f"/api/runs/{run['id']}/decision", json={"action": "approve"}
            ).status_code
            == 202
        )
        assert wait_for_run(client, run["id"])["status"] == "completed"
        ticket = client.get("/api/tickets").json()[0]
        assert (
            client.patch(
                "/api/tickets/" + ticket["id"], json={"status": "resolved"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/documents",
                files={"file": ("note.txt", "端口参数为9071".encode(), "text/plain")},
            ).status_code
            == 201
        )
        cookies = dict(client.cookies)
        client.cookies.clear()
        client.get("/api/bootstrap")
        assert client.get("/api/sessions/" + session["id"]).status_code == 404
        assert client.get("/api/runs/" + run["id"]).status_code == 404
        assert (
            client.patch(
                "/api/tickets/" + ticket["id"], json={"status": "open"}
            ).status_code
            == 404
        )
        client.cookies.update(cookies)
