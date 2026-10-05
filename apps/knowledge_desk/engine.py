from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict

from .grounding import grounded_citations
from .hybrid import HybridRetriever
from .llm import ModelError, OfflineModel, QwenModel, TemporaryModelError
from .schemas import ApprovalDecision, TicketDraft


def merge_messages(old, new):
    merged = {m["id"]: m for m in old}
    for message in new:
        merged[message["id"]] = message
    return list(merged.values())


class State(TypedDict, total=False):
    owner: str
    session_id: str
    run_id: str
    question: str
    mode: str
    messages: Annotated[list[dict], merge_messages]
    search_query: str
    intent: str
    evidence: list[dict]
    citations: list[dict]
    answer: str
    draft: dict
    decision: dict
    ticket_id: str
    retrieval: dict


class Engine:
    def __init__(self, settings, store, model=None, retriever=None):
        self.settings = settings
        self.store = store
        self.model = model or (
            OfflineModel() if settings.offline else QwenModel(settings, store)
        )
        self.retriever = retriever or HybridRetriever(settings, store)
        self.conn = sqlite3.connect(
            settings.data_dir / "checkpoints.sqlite", check_same_thread=False
        )
        self.saver = SqliteSaver(
            self.conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[])
        )
        self.graph = self.build_graph()
        self.pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="knowledge-desk"
        )
        self.locks = {}
        self.locks_guard = threading.Lock()

    def close(self):
        self.pool.shutdown(wait=True)
        self.model.close()
        self.retriever.close()
        self.conn.close()

    def event(self, state, node, message, **extra):
        self.store.event(
            state["run_id"], "node", {"node": node, "message": message, **extra}
        )

    def build_graph(self):
        def prepare(state):
            self.event(state, "prepare", "开始处理问题")
            return {
                "answer": "",
                "draft": {},
                "decision": {},
                "ticket_id": "",
                "evidence": [],
                "citations": [],
                "retrieval": {},
                "messages": [
                    {
                        "id": state["run_id"] + ":user",
                        "role": "user",
                        "content": state["question"],
                    }
                ],
            }

        def plan(state):
            self.event(state, "plan", "理解问题与会话上下文")
            result = self.model.plan(state)
            intent = result.intent if state["mode"] == "auto" else state["mode"]
            return {"search_query": result.search_query, "intent": intent}

        def retrieve(state):
            result = self.retriever.search(
                state["owner"],
                state["search_query"],
                run_id=state["run_id"],
                rerank_query=state["question"]
                + "\n上下文补全检索词："
                + state["search_query"],
            )
            sources = result.sources
            if (
                not sources
                and result.mode != "rerank"
                and state["search_query"] != state["question"]
            ):
                result = self.retriever.search(
                    state["owner"], state["question"], run_id=state["run_id"]
                )
                sources = result.sources
            if result.fallback:
                self.event(state, "retrieve_fallback", result.fallback)
            self.event(
                state,
                "retrieve",
                f"{ {'hybrid': '关键词＋语义', 'semantic': '语义', 'rerank': '千问重排', 'keyword': '关键词'}[result.mode] }检索找到{len(sources)}段资料",
                query=result.query,
                count=len(sources),
                retrieval=result.metadata(),
                sources=[
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "heading": c["heading"],
                        "score": c["score"],
                        "channels": c.get("channels", []),
                        "semantic_score": c.get("semantic_score"),
                        "rerank_grade": c.get("rerank_grade"),
                    }
                    for c in sources
                ],
            )
            return {"evidence": sources, "retrieval": result.metadata()}

        def answer(state):
            self.event(state, "answer", "根据资料生成回答")
            if not state["evidence"]:
                return {
                    "answer": "当前知识库没有找到足够依据。请上传相关文档，或补充更具体的问题。",
                    "citations": [],
                }
            result = self.model.answer(state)
            sources = grounded_citations(
                result, state["evidence"], require_quotes=not self.settings.offline
            )
            if not sources:
                return {
                    "answer": "检索到的资料没有提供足够依据，暂时无法回答这个问题。请补充相关文档。",
                    "citations": [],
                }
            return {"answer": result.answer, "citations": sources}

        def draft(state):
            self.event(state, "draft", "整理工单草稿，等待人工审核")
            result = self.model.draft(state)
            return {
                "draft": result.model_dump(),
                "citations": [
                    {
                        k: c.get(k)
                        for k in ("id", "document_id", "name", "heading", "page")
                    }
                    for c in state["evidence"][:3]
                ],
            }

        def approval(state):
            decision = interrupt(
                {
                    "run_id": state["run_id"],
                    "draft": state["draft"],
                    "citations": state["citations"],
                },
                response_schema=ApprovalDecision,
            )
            return {"decision": decision.model_dump()}

        def create_ticket(state):
            self.event(state, "create_ticket", "保存审核后的工单")
            draft_value = state["decision"].get("draft") or state["draft"]
            ticket = self.store.create_ticket(
                state["owner"],
                state["session_id"],
                state["run_id"],
                TicketDraft.model_validate(draft_value),
            )
            return {
                "ticket_id": ticket["id"],
                "draft": draft_value,
                "answer": f"工单 {ticket['id']} 已创建：{ticket['title']}。你可以在工单页查看处理状态。",
            }

        def cancel(state):
            return {
                "answer": "已取消创建工单，没有保存新的工单。",
                "ticket_id": "",
                "citations": [],
            }

        def finalize(state):
            self.event(state, "finalize", "本轮处理完成")
            message = {
                "id": state["run_id"] + ":assistant",
                "role": "assistant",
                "content": state["answer"],
                "citations": state["citations"],
                "ticket_id": state.get("ticket_id", ""),
                "run_id": state["run_id"],
            }
            return {"messages": [message]}

        graph = StateGraph(State)
        retry = RetryPolicy(
            max_attempts=2, initial_interval=1, retry_on=TemporaryModelError
        )
        nodes = {
            "prepare": prepare,
            "plan": plan,
            "retrieve": retrieve,
            "answer": answer,
            "draft": draft,
            "approval": approval,
            "create_ticket": create_ticket,
            "cancel": cancel,
            "finalize": finalize,
        }
        for name, node in nodes.items():
            graph.add_node(
                name,
                node,
                retry_policy=retry if name in ("plan", "answer", "draft") else None,
            )
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            lambda state: "draft" if state["intent"] == "ticket" else "answer",
        )
        graph.add_edge("answer", "finalize")
        graph.add_edge("draft", "approval")
        graph.add_conditional_edges(
            "approval",
            lambda state: (
                "create_ticket"
                if state["decision"]["action"] == "approve"
                else "cancel"
            ),
        )
        graph.add_edge("create_ticket", "finalize")
        graph.add_edge("cancel", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self.saver)

    def config(self, session_id):
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 20}

    def snapshot(self, owner, session_id):
        session = self.store.session(owner, session_id)
        snapshot = self.graph.get_state(self.config(session_id))
        return {
            "session": session,
            "messages": snapshot.values.get("messages", []),
            "pending": snapshot.values.get("draft") if snapshot.interrupts else None,
            "retrieval": snapshot.values.get("retrieval", {}),
            "checkpoint_count": len(
                list(self.graph.get_state_history(self.config(session_id)))
            ),
            "run": self.store.run(owner, session["current_run"])
            if session["current_run"]
            else None,
        }

    def submit(self, run_id, owner, *, resume=False):
        return self.pool.submit(self.execute, owner, run_id, resume=resume)

    def execute(self, owner, run_id, *, resume=False):
        run = self.store.run(owner, run_id)
        with self.locks_guard:
            lock = self.locks.setdefault(run["session_id"], threading.Lock())
        with lock:
            config = self.config(run["session_id"])
            self.store.set_status(run_id, "running")
            self.store.event(
                run_id, "status", {"status": "running", "message": "任务开始"}
            )
            snapshot = self.graph.get_state(config)
            initial = {
                "owner": owner,
                "session_id": run["session_id"],
                "run_id": run_id,
                "question": run["question"],
                "mode": run["mode"],
            }
            payload = initial
            if resume and snapshot.values.get("run_id") == run_id:
                payload = (
                    Command(resume=run["decision"])
                    if snapshot.interrupts and run["decision"]
                    else None
                )
            try:
                for _ in self.graph.stream(
                    payload, config, stream_mode="updates", durability="sync"
                ):
                    pass
                final = self.graph.get_state(config)
                if final.interrupts:
                    status = "awaiting_approval"
                    self.store.event(
                        run_id,
                        "approval",
                        {
                            "draft": final.values["draft"],
                            "citations": final.values["citations"],
                        },
                    )
                else:
                    status = (
                        "cancelled"
                        if final.values.get("decision", {}).get("action") == "reject"
                        else "completed"
                    )
                    self.store.event(
                        run_id,
                        "answer",
                        {
                            "answer": final.values.get("answer", ""),
                            "citations": final.values.get("citations", []),
                            "ticket_id": final.values.get("ticket_id", ""),
                        },
                    )
                self.store.set_status(run_id, status)
                self.store.event(
                    run_id,
                    "status",
                    {
                        "status": status,
                        "message": "等待审核"
                        if status == "awaiting_approval"
                        else "任务完成",
                    },
                )
            except Exception as error:  # noqa: BLE001 -- worker boundary persists failures for recovery
                message = (
                    str(error)
                    if isinstance(error, ModelError)
                    else "任务执行失败，已保留检查点，可重试。"
                )
                self.store.set_status(run_id, "error", message)
                self.store.event(run_id, "error", {"message": message})
