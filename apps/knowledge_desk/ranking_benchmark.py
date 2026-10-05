"""Compare unchanged retrieval baselines with structured Qwen reranking."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

from .benchmark import dataset, metrics
from .config import APP_DIR, Settings
from .hybrid import HybridRetriever, cosine_ranking, digest, fuse
from .reranking import QwenReranker
from .retrieval import normalized, search, seed_samples
from .store import Store, now


def percentile(values, p=0.95):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def compare_ranking(settings, store, *, warm=False):
    cases = dataset()
    retriever = HybridRetriever(replace(settings, retrieval_mode="hybrid"), store)
    reranker = QwenReranker(settings, store)
    rows, durations = [], []
    calls = tokens = cache_hits = 0
    try:
        started = time.perf_counter()
        chunks, vectors, queries, prep = retriever.prepare(
            "benchmark", [c["question"] for c in cases]
        )
        prep.elapsed_ms = round((time.perf_counter() - started) * 1000)
        for case in cases:
            keyword = search(chunks, case["question"], expand=True, limit=12)
            semantic = cosine_ranking(
                chunks,
                vectors,
                queries[digest(normalized(case["question"]).strip())],
                settings.semantic_min_score,
            )
            candidates = fuse(keyword, semantic, 24)
            started = time.perf_counter()
            ranked = reranker.rerank("benchmark", case["question"], candidates)
            elapsed = round((time.perf_counter() - started) * 1000)
            durations.append(elapsed)
            calls += ranked.calls
            tokens += ranked.tokens
            cache_hits += ranked.cache_hits
            rows.append(
                {
                    **case,
                    "bm25": [
                        c["heading"] for c in search(chunks, case["question"], limit=3)
                    ],
                    "expanded_bm25": [c["heading"] for c in keyword[:3]],
                    "semantic": [c["heading"] for c in semantic[:3]],
                    "hybrid": [c["heading"] for c in candidates[:3]],
                    "rerank": [c["heading"] for c in ranked.sources[:3]],
                    "rerank_grades": [c["rerank_grade"] for c in ranked.sources[:3]],
                    "rerank_calls": ranked.calls,
                    "rerank_tokens": ranked.tokens,
                    "elapsed_ms": elapsed,
                }
            )
            if len(rows) % 12 == 0:
                print(
                    f"Ranking {len(rows)}/{len(cases)}; calls={calls}; tokens={tokens}",
                    flush=True,
                )
        if warm and (calls or prep.embedding_calls):
            raise RuntimeError("热缓存验证失败：仍有模型或向量调用。")
        channels = ["bm25", "expanded_bm25", "semantic", "hybrid", "rerank"]
        summary = {
            split: {
                channel: metrics([r for r in rows if r["split"] == split], channel)
                for channel in channels
            }
            for split in ("faq", "development", "validation")
        }
        return {
            "created_at": now(),
            "scope": "原132条自编回归集；验证集已在v0.2查看，不属于新盲测。没有按验证集调参。",
            "dataset_sha256": digest(
                json.dumps(cases, ensure_ascii=False, sort_keys=True)
            ),
            "strategy": "两路各12候选，去重并全部重排（至多24）；模型只给0-3等级，保留2/3级",
            "reranker_type": "structured chat LLM; not a dedicated cross-encoder",
            "model": settings.model,
            "semantic_threshold": settings.semantic_min_score,
            "case_count": len(rows),
            "embedding_stats": prep.metadata(),
            "rerank_calls": calls,
            "rerank_tokens": tokens,
            "rerank_cache_hits": cache_hits,
            "rerank_p95_ms": percentile(durations),
            "timing_scope": "重排阶段，含缓存读写或实际模型请求；不含检索向量准备和最终回答生成",
            "summary": summary,
            "cases": rows,
        }
    finally:
        retriever.close()
        reranker.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm", action="store_true", help="检查跨进程缓存复用，必须没有API调用"
    )
    args = parser.parse_args()
    settings = Settings.load()
    store = Store(settings.data_dir / "retrieval-eval.sqlite")
    seed_samples(store, "benchmark")
    report = compare_ranking(settings, store, warm=args.warm)
    path = (
        APP_DIR
        / "reports"
        / ("ranking-v3-warm.json" if args.warm else "ranking-v3.json")
    )
    path.parent.mkdir(exist_ok=True)
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
