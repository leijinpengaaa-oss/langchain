from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

from .challenge_cases import CHALLENGE_CASES
from .config import APP_DIR, Settings
from .eval_cases import CASES
from .hybrid import HybridRetriever, cosine_ranking, digest, fuse
from .retrieval import normalized, search, seed_samples
from .store import Store, now


def dataset():
    faq = [
        {
            "id": f"faq-{i:03d}",
            "question": c["question"],
            "headings": [c["heading"]] if c["heading"] else [],
            "kind": "positive" if c["heading"] else "out_of_scope",
            "split": "faq",
            "tag": "literal",
        }
        for i, c in enumerate(CASES)
    ]
    return faq + CHALLENGE_CASES


def metrics(rows, channel):
    positive = [r for r in rows if r["kind"] == "positive"]
    negative = [r for r in rows if r["kind"] == "out_of_scope"]
    ranks = []
    for row in positive:
        ranks.append(
            next(
                (i for i, h in enumerate(row[channel], 1) if h in row["headings"]), None
            )
        )
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "hit_at_1": sum(r == 1 for r in ranks) / len(ranks) if ranks else None,
        "hit_at_3": sum(r is not None and r <= 3 for r in ranks) / len(ranks)
        if ranks
        else None,
        "mrr_at_3": sum(1 / r for r in ranks if r is not None and r <= 3) / len(ranks)
        if ranks
        else None,
        "unrelated_rejection": sum(not r[channel] for r in negative) / len(negative)
        if negative
        else None,
    }


def compare(settings, store, *, semantic=False, calibrate=False):
    retriever = HybridRetriever(settings, store)
    cases = dataset()
    chunks = store.chunks("benchmark")
    vectors = query_vectors = None
    stats = None
    started = time.perf_counter()
    try:
        if semantic:
            chunks, vectors, query_vectors, stats = retriever.prepare(
                "benchmark", [c["question"] for c in cases]
            )
        vectorization_seconds = time.perf_counter() - started
        if stats:
            stats.elapsed_ms = round(vectorization_seconds * 1000)

        def rankings(case, threshold):
            keyword = search(chunks, case["question"], expand=True, limit=12)
            semantic_rows = (
                cosine_ranking(
                    chunks,
                    vectors,
                    query_vectors[digest(normalized(case["question"]).strip())],
                    threshold,
                )
                if semantic
                else []
            )
            return {
                "bm25": [
                    c["heading"] for c in search(chunks, case["question"], limit=3)
                ],
                "expanded_bm25": [c["heading"] for c in keyword[:3]],
                "semantic": [c["heading"] for c in semantic_rows[:3]],
                "hybrid": [c["heading"] for c in fuse(keyword, semantic_rows, 3)]
                if semantic
                else [],
            }

        threshold = settings.semantic_min_score
        sweep = []
        if semantic and calibrate:
            development = [c for c in cases if c["split"] == "development"]
            for candidate in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                rows = [{**c, **rankings(c, candidate)} for c in development]
                result = metrics(rows, "hybrid")
                sweep.append(
                    {
                        "threshold": candidate,
                        "balanced_hit_rejection": (
                            result["hit_at_3"] + result["unrelated_rejection"]
                        )
                        / 2,
                        **result,
                    }
                )
            threshold = max(
                sweep,
                key=lambda c: (
                    c["balanced_hit_rejection"],
                    c["mrr_at_3"],
                    c["threshold"],
                ),
            )["threshold"]
        rows = [{**c, **rankings(c, threshold)} for c in cases]
        channels = ["bm25", "expanded_bm25"] + (
            ["semantic", "hybrid"] if semantic else []
        )
        summary = {
            split: {
                channel: metrics([r for r in rows if r["split"] == split], channel)
                for channel in channels
            }
            for split in ("faq", "development", "validation")
        }
        warm = []
        if semantic:
            retriever.settings = replace(settings, semantic_min_score=threshold)
            for case in cases[:10]:
                result = retriever.search("benchmark", case["question"])
                if result.fallback or result.embedding_calls:
                    raise RuntimeError("热缓存验证不符合预期。")
                warm.append(result.elapsed_ms)
        warm.sort()
        return {
            "created_at": now(),
            "scope": "自编132条检索评测，含固定开发/验证划分；不是独立盲测或答案准确率",
            "dataset_sha256": digest(
                json.dumps(cases, ensure_ascii=False, sort_keys=True)
            ),
            "semantic_enabled": semantic,
            "embedding_model": settings.embedding_model if semantic else None,
            "dimensions": settings.embedding_dimensions if semantic else None,
            "selected_semantic_threshold": threshold if semantic else None,
            "selection": "仅开发集选择阈值；最大化Hit@3与无关拒绝率均值，平局比较MRR再取较高阈值"
            if calibrate
            else "固定配置阈值",
            "rank_constant": 60,
            "candidate_window": 12,
            "case_count": len(rows),
            "unsupported_fact_count": sum(
                r["kind"] == "unsupported_fact" for r in rows
            ),
            "unsupported_note": "相近但缺少答案的6条只保留来源记录，不把检索到相关主题算作幻觉；拒答另做真实回答检查",
            "vectorization_seconds": round(vectorization_seconds, 3),
            "embedding_stats": stats.metadata() if stats else None,
            "warm_search_p95_ms": warm[max(0, math.ceil(0.95 * len(warm)) - 1)]
            if warm
            else None,
            "warm_note": "10次相同进程已缓存查询，包含SQLite读取和排名，不含回答生成、网络或首次索引",
            "calibration": sweep,
            "summary": summary,
            "cases": rows,
        }
    finally:
        retriever.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic", action="store_true", help="真实调用千问向量接口，保留本地向量缓存"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="仅使用开发集选择相似度阈值"
    )
    parser.add_argument(
        "--warm", action="store_true", help="验证重启后全部向量来自缓存，另存热缓存报告"
    )
    args = parser.parse_args()
    if args.calibrate and not args.semantic:
        parser.error("--calibrate 需要 --semantic")
    if args.warm and not args.semantic:
        parser.error("--warm 需要 --semantic")
    settings = replace(
        Settings.load(), retrieval_mode="hybrid" if args.semantic else "keyword"
    )
    store = Store(settings.data_dir / "retrieval-eval.sqlite")
    seed_samples(store, "benchmark")
    report = compare(settings, store, semantic=args.semantic, calibrate=args.calibrate)
    if args.warm and report["embedding_stats"]["embedding_calls"]:
        raise RuntimeError("热缓存检查未通过：仍发生了向量API调用。")
    name = (
        "hybrid-warm.json"
        if args.warm
        else "hybrid.json"
        if args.semantic
        else "retrieval-v2.json"
    )
    path = APP_DIR / "reports" / name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"cases", "calibration"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Report:", path)


if __name__ == "__main__":
    main()
