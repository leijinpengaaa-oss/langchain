from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .challenge_cases import ANSWER_CASES
from .config import APP_DIR, Settings
from .engine import Engine
from .eval_cases import CASES, LIVE_CASES
from .retrieval import search, seed_samples, tokenize
from .schemas import GroundedAnswer, MessageRequest
from .store import Store, now


def lexical_baseline(chunks, query):
    terms = set(tokenize(query))
    ranked = sorted(
        (
            (
                sum(
                    min(count, 1)
                    for term, count in Counter(tokenize(c["text"])).items()
                    if term in terms
                ),
                c,
            )
            for c in chunks
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [c for score, c in ranked[:3] if score > 0]


def retrieval_report(store):
    chunks = store.chunks("eval")
    rows = []
    for case in CASES:
        bm25 = search(chunks, case["question"], limit=3)
        baseline = lexical_baseline(chunks, case["question"])
        rows.append(
            {
                **case,
                "bm25_sources": [c["heading"] for c in bm25],
                "baseline_sources": [c["heading"] for c in baseline],
                "bm25_pass": case["heading"] in [c["heading"] for c in bm25]
                if case["heading"]
                else not bm25,
                "baseline_pass": case["heading"] in [c["heading"] for c in baseline]
                if case["heading"]
                else not baseline,
            }
        )
    positive = [r for r in rows if r["heading"]]
    negative = [r for r in rows if not r["heading"]]
    return {
        "created_at": now(),
        "scope": "自编星舟示例FAQ；离线检索评测，不代表模型回答准确率或真实企业数据效果",
        "case_count": len(rows),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "bm25_hit_at_3": sum(r["bm25_pass"] for r in positive) / len(positive),
        "baseline_hit_at_3": sum(r["baseline_pass"] for r in positive) / len(positive),
        "bm25_negative_rejection": sum(r["bm25_pass"] for r in negative)
        / len(negative),
        "cases": rows,
    }


def live_report(settings, store, limit, cases=LIVE_CASES):
    engine = Engine(settings, store)
    rows = []
    try:
        for index, case in enumerate(cases[:limit]):
            session = store.create_session("eval", "实时评测")
            run, _ = store.create_run(
                "eval",
                session["id"],
                MessageRequest(
                    question=case["question"],
                    mode="answer",
                    request_id=f"live-rag-{index}",
                ),
            )
            started = time.perf_counter()
            engine.execute("eval", run["id"])
            elapsed = time.perf_counter() - started
            current = store.run("eval", run["id"])
            if current["status"] != "completed":
                raise RuntimeError(current["error"])
            snapshot = engine.snapshot("eval", session["id"])
            message = snapshot["messages"][-1]
            citations = message.get("citations", [])
            valid = all(store.chunk("eval", c["id"]) for c in citations)
            passed = all(
                k.lower() in message["content"].lower() for k in case["keywords"]
            )
            if case.get("refusal"):
                passed = not citations and "依据" in message["content"]
            baseline_session = store.create_session("eval", "无检索对照")
            baseline_run, _ = store.create_run(
                "eval",
                baseline_session["id"],
                MessageRequest(
                    question=case["question"], request_id=f"live-baseline-{index}"
                ),
            )
            baseline = engine.model.complete(
                baseline_run["id"],
                "无检索对照",
                '你是助手。直接回答问题，不知道时说明不确定。返回 {"answer":"回答","citation_ids":[]}',
                {"question": case["question"]},
                GroundedAnswer,
            )
            store.set_status(baseline_run["id"], "completed")
            baseline_pass = (
                all(k.lower() in baseline.answer.lower() for k in case["keywords"])
                if case["keywords"]
                else None
            )
            rows.append(
                {
                    "question": case["question"],
                    "answer": message["content"],
                    "baseline_answer": baseline.answer,
                    "keyword_check_pass": passed,
                    "baseline_keyword_check_pass": baseline_pass,
                    "citation_count": len(citations),
                    "citations_resolve": bool(valid),
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens": current["tokens"],
                    "model_calls": current["model_calls"],
                    "baseline_tokens": store.run("eval", baseline_run["id"])["tokens"],
                    "retrieval": snapshot["retrieval"],
                }
            )
    finally:
        engine.close()
    durations = sorted(r["elapsed_seconds"] for r in rows)
    return {
        "created_at": now(),
        "scope": "小样本实时烟雾测试；关键词检查仅是自动代理指标，需要人工复核，不作为整体准确率",
        "model": settings.model,
        "embedding_model": settings.embedding_model,
        "retrieval_mode": settings.retrieval_mode,
        "case_count": len(rows),
        "keyword_checks_passed": sum(r["keyword_check_pass"] for r in rows),
        "p95_seconds": durations[max(0, math.ceil(len(durations) * 0.95) - 1)],
        "rag_tokens": sum(r["tokens"] for r in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="调用千问对比有检索和无检索回答"
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--challenge", action="store_true", help="评测口语/英文改写与相近但无答案的问题"
    )
    args = parser.parse_args()
    if args.challenge and not args.live:
        parser.error("--challenge 需要 --live")
    reports = APP_DIR / "reports"
    reports.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knowledge-eval-") as directory:
        settings = replace(Settings.load(), data_dir=Path(directory))
        store = Store(settings.data_dir / "workspace.sqlite")
        seed_samples(store, "eval")
        report = (
            live_report(
                settings,
                store,
                max(1, min(args.limit, len(ANSWER_CASES) if args.challenge else 4)),
                ANSWER_CASES if args.challenge else LIVE_CASES,
            )
            if args.live
            else retrieval_report(store)
        )
    path = reports / (
        "challenge-answers.json"
        if args.challenge
        else "live.json"
        if args.live
        else "retrieval.json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Report:", path)


if __name__ == "__main__":
    main()
