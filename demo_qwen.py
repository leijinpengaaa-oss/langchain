"""Run Qwen with LangGraph: python demo_qwen.py [--interactive]."""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "你是运行在 LangGraph 上的千问助手。用中文简短回答，每次不超过80字。"
    "所有算术计算都必须调用 calculator 工具，包括简单计算。"
    "工具只支持数字、括号和加减乘除。不能执行代码、访问文件或联网搜索。"
    "用户提到刚才的结果时，结合当前会话历史理解。"
    "如果用户拒绝工具执行，就说明操作已取消，不要自行计算或重复请求工具。"
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "准确计算数字的加、减、乘、除及括号表达式。所有算术都使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "如 (12*8)*2+10"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}]


class DemoError(RuntimeError):
    """An error that can be displayed without printing credentials."""


@dataclass
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str

    @classmethod
    def read(cls, env_file: Path):
        if not env_file.is_file():
            raise DemoError(f"找不到配置文件：{env_file}。可用 --env-file 指定路径。")
        values = dotenv_values(env_file, encoding="utf-8-sig", interpolate=False)

        def get(*names):
            for name in names:
                value = values.get(name) or os.environ.get(name)
                if value and value.strip():
                    return value.strip()
            return ""

        key = get("LLM_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY")
        base = get("LLM_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "OPENAI_BASE_URL")
        model = get("LLM_MODEL", "QWEN_MODEL", "DASHSCOPE_MODEL", "OPENAI_MODEL", "MODEL_NAME")
        if not key or not base or not model:
            missing = [label for label, value in (
                ("LLM_API_KEY", key), ("LLM_BASE_URL", base), ("LLM_MODEL", model)
            ) if not value]
            raise DemoError("配置缺少：" + ", ".join(missing))
        url = urlsplit(base)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise DemoError("LLM_BASE_URL 应是 HTTPS 接口地址，不能包含凭据或查询参数。")
        if not model.lower().startswith("qwen"):
            raise DemoError("LLM_MODEL 应填写千问的模型 ID，例如 qwen-plus。")
        return cls(key, base.rstrip("/"), model)


def calculate(expression: str) -> int | float:
    """Evaluate only bounded arithmetic syntax, without eval or executable code."""
    if not isinstance(expression, str) or not 0 < len(expression) <= 160:
        raise ValueError("表达式必须是1至160个字符的字符串。")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError("表达式过于复杂。")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("仅支持数字、括号、加减乘除。")
        if abs(value) > 10**15 or not math.isfinite(value):
            raise ValueError("数值超出本演示支持的范围。")
        return value

    return visit(tree.body)


class State(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    turn: int
    calls_this_turn: int
    tokens: Annotated[int, operator.add]
    approved: bool


def build_graph(client: httpx.Client, settings: Settings, confirm_tools: bool):
    def prepare(state: State):
        return {"turn": state.get("turn", 0) + 1, "calls_this_turn": 0}

    def assistant(state: State):
        calls = state.get("calls_this_turn", 0)
        if calls >= 6:
            raise DemoError("本轮已达到6次模型调用上限，停止执行。")
        print(f"  [模型] 正在调用 {settings.model}…", flush=True)
        request = {
            "model": settings.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *state["messages"]],
            "tools": TOOLS,
            "max_tokens": 600,
            "enable_thinking": False,
        }
        try:
            response = client.post(
                settings.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + settings.api_key},
                json=request,
            )
        except httpx.RequestError as error:
            raise DemoError("模型网络请求失败：" + str(error).replace(settings.api_key, "<redacted>")) from None
        if response.is_error:
            # Do not echo response bodies: a gateway could reflect secrets in an error.
            hints = {401: "请检查 API Key 与接口地域是否匹配。", 403: "请检查模型访问权限。", 404: "请检查接口地址与模型名称。", 429: "请求受限，请检查额度或稍后重试。"}
            raise DemoError(f"模型接口返回 HTTP {response.status_code}。" + hints.get(response.status_code, "请检查模型及兼容接口配置。"))
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") == "length":
                raise DemoError("模型输出达到长度上限，请缩短问题后重试。")
            normalized = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("tool_calls"):
                normalized["tool_calls"] = message["tool_calls"]
            elif not normalized["content"]:
                raise DemoError("模型未返回正文或工具调用。请检查模型是否支持非思考模式。")
            tokens = data.get("usage", {}).get("total_tokens", 0)
        except (KeyError, IndexError, TypeError, ValueError):
            raise DemoError("模型接口未返回有效的 Chat Completions 响应。") from None
        return {"messages": [normalized], "calls_this_turn": calls + 1, "tokens": tokens}

    def review(state: State):
        calls = state["messages"][-1]["tool_calls"]
        approved = interrupt({"message": "是否允许执行模型请求的计算工具？", "tool_calls": calls})
        return {"approved": approved is True}

    def tools(state: State):
        messages = []
        for call in state["messages"][-1]["tool_calls"]:
            function = call["function"]
            if confirm_tools and not state.get("approved", False):
                output = {"error": "用户拒绝执行工具，请直接告知操作已取消。"}
            else:
                try:
                    if function["name"] != "calculator":
                        raise ValueError("未知工具；只允许 calculator。")
                    args = json.loads(function["arguments"])
                    result = calculate(args["expression"])
                    output = {"expression": args["expression"], "result": result}
                except (ValueError, SyntaxError, TypeError, KeyError, ZeroDivisionError):
                    output = {"error": "参数无效：请使用有限数字、括号、加减乘除，且不能除以零。"}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(output, ensure_ascii=False)})
        return {"messages": messages}

    graph = StateGraph(State)
    graph.add_node("prepare", prepare)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", tools)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "assistant")
    tool_target = "review" if confirm_tools else "tools"
    graph.add_conditional_edges("assistant", lambda state: tool_target if state["messages"][-1].get("tool_calls") else END)
    if confirm_tools:
        graph.add_node("review", review)
        graph.add_edge("review", "tools")
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[])))


def run_turn(graph, config, question: str, interactive: bool):
    print(f"\n你：{question}", flush=True)
    payload = {"messages": [{"role": "user", "content": question}]}
    while True:
        paused = False
        for event in graph.stream(payload, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    paused = True
                    for item in update:
                        print("  [暂停] " + json.dumps(item.value, ensure_ascii=False), flush=True)
                elif node == "assistant":
                    message = update["messages"][-1]
                    if message.get("tool_calls"):
                        for call in message["tool_calls"]:
                            print(f"  [工具请求] {call['function']['name']}({call['function']['arguments']})", flush=True)
                    else:
                        print("千问：" + message["content"], flush=True)
                elif node == "tools":
                    for message in update["messages"]:
                        print("  [工具结果] " + message["content"], flush=True)
        if not paused:
            break
        approved = input("允许执行？[y/N] ").strip().lower() == "y" if interactive else True
        if not interactive:
            print("  [演示] 模拟人工选择同意。", flush=True)
        payload = Command(resume=approved)
    snapshot = graph.get_state(config)
    print(f"  [记忆] 当前会话 {snapshot.values['turn']} 轮", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--confirm-tools", action="store_true", help="每次工具调用前暂停确认")
    args = parser.parse_args(argv)
    try:
        settings = Settings.read(args.env_file)
        print(f"模型：{settings.model}\n接口：{settings.base_url}\n配置：{args.env_file.resolve()}", flush=True)
        with httpx.Client(timeout=httpx.Timeout(60, connect=15)) as client:
            graph = build_graph(client, settings, args.confirm_tools)
            config = {"configurable": {"thread_id": "qwen-demo"}, "recursion_limit": 24}
            if args.interactive:
                print("输入自然语言问题；输入 q 退出。", flush=True)
                while True:
                    question = input("\n问题> ").strip()
                    if question.lower() in {"q", "quit", "exit"}:
                        break
                    if question:
                        run_turn(graph, config, question, True)
            else:
                for question in ("用一句话介绍你自己。", "帮我算一下十二乘八。", "把刚才的结果翻倍，再加十。"):
                    run_turn(graph, config, question, False)
                state = graph.get_state(config).values
                print(f"\n完成：{state['turn']} 轮，{len(list(graph.get_state_history(config)))} 个内存检查点，API 返回总用量 {state.get('tokens', 0)} tokens。")
        return 0
    except (DemoError, httpx.HTTPError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n会话已结束。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
