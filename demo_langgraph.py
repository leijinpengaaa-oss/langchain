"""Run a local LangGraph tour; use --interactive to enter your own questions."""

from __future__ import annotations

import argparse
import inspect
import io
import json
import operator
import re
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class State(TypedDict, total=False):
    question: str
    turn: int
    needs_tool: bool
    left: int
    right: int
    result: int
    approved: bool
    answer: str
    history: Annotated[list[str], operator.add]


def classify(state: State) -> dict:
    # Only recognize multiplication of two integers; never evaluate user code.
    match = re.fullmatch(
        r"(?:计算\s*)?(-?\d{1,9})\s*[*×xX]\s*(-?\d{1,9})\s*[?？]?",
        state["question"].strip(),
    )
    update = {
        "turn": state.get("turn", 0) + 1,
        "needs_tool": match is not None,
        "approved": True,
        "answer": "",
        "result": 0,
    }
    if match:
        update.update(left=int(match[1]), right=int(match[2]))
    return update


def calculate(state: State) -> dict:
    return {"result": state["left"] * state["right"]}


def review(state: State) -> dict:
    approved = interrupt(
        {
            "message": "本演示中，结果达到 100 时暂停，等待确认后再输出。",
            "question": state["question"],
            "result": state["result"],
        }
    )
    return {"approved": approved is True}


def respond(state: State) -> dict:
    if not state["needs_tool"]:
        answer = "LangGraph 把任务组织成图，支持状态管理、工具调用和暂停恢复。"
    elif not state["approved"]:
        answer = "已取消输出计算结果。"
    else:
        answer = f"{state['left']} × {state['right']} = {state['result']}"
    return {
        "answer": answer,
        "history": [f"用户：{state['question']}", f"助手：{answer}"],
    }


def build_graph():
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("calculate", calculate)
    builder.add_node("review", review)
    builder.add_node("respond", respond)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", lambda state: "calculate" if state["needs_tool"] else "respond"
    )
    builder.add_conditional_edges(
        "calculate", lambda state: "review" if state["result"] >= 100 else "respond"
    )
    builder.add_edge("review", "respond")
    builder.add_edge("respond", END)
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[]))
    return builder.compile(checkpointer=saver)


def run_turn(graph, config: dict, question: str, *, interactive: bool = False):
    print(f"\n输入：{question}")

    def show_events(payload):
        paused = False
        for event in graph.stream(payload, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    paused = True
                    for item in update:
                        print("  [暂停] " + json.dumps(item.value, ensure_ascii=False))
                else:
                    print(f"  [{node}] " + json.dumps(update, ensure_ascii=False))
        return paused

    if show_events({"question": question}):
        if interactive:
            approved = input("同意输出结果吗？[y/N] ").strip().lower() == "y"
        else:
            approved = True
            print("  [演示] 程序模拟人工选择“同意”，使用 Command(resume=True) 继续。")
        show_events(Command(resume=approved))
    snapshot = graph.get_state(config)
    print(f"输出：{snapshot.values['answer']}")
    print(f"会话记忆：已完成 {snapshot.values['turn']} 轮")


def tour():
    print(f"LangGraph {version('langgraph')} 本地运行演示")
    print(f"加载源码：{inspect.getfile(StateGraph)}")
    print("节点使用固定规则，不调用大模型，也不需要 API Key。")
    graph = build_graph()
    config = {"configurable": {"thread_id": "local-demo"}}
    for question in ("LangGraph 能做什么？", "计算 12 × 8", "计算 25 × 8"):
        run_turn(graph, config, question)
    snapshot = graph.get_state(config)
    checkpoints = list(graph.get_state_history(config))
    print(f"\n执行完成：{snapshot.values['turn']} 轮，{len(checkpoints)} 个内存检查点。")
    print(f"历史记录：{json.dumps(snapshot.values['history'], ensure_ascii=False)}")
    print("内存检查点只在当前进程中保留；跨进程持久化可改用 SQLite/Postgres。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="交互式输入问题")
    args = parser.parse_args()
    if args.interactive:
        graph = build_graph()
        config = {"configurable": {"thread_id": "interactive-demo"}}
        print("输入问题或两个整数的乘法（例如 12*8）；输入 q 退出。")
        while True:
            question = input("\n问题> ").strip()
            if question.lower() in {"q", "quit", "exit"}:
                break
            if question:
                run_turn(graph, config, question, interactive=True)
    else:
        output = io.StringIO()
        with redirect_stdout(output):
            tour()
        report = output.getvalue()
        output_path = Path(__file__).with_name("demo_langgraph_output.txt")
        output_path.write_text(report, encoding="utf-8")
        print(report, end="")
        print(f"\n运行记录已保存：{output_path}")


if __name__ == "__main__":
    main()
