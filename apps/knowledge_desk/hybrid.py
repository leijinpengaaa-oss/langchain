from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from .embeddings import EmbeddingError, QwenEmbeddings, unit_vector
from .reranking import QwenReranker, RerankError
from .retrieval import normalized, search


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cosine_ranking(chunks, vectors, query_vector, minimum, limit=12):
    rows = []
    for chunk in chunks:
        score = max(
            -1.0,
            min(1.0, sum(a * b for a, b in zip(vectors[chunk["id"]], query_vector))),
        )
        if score >= minimum:
            rows.append({**chunk, "semantic_score": score})
    return sorted(rows, key=lambda c: (-c["semantic_score"], c["id"]))[:limit]


def fuse(lexical, semantic, limit=4, constant=60):
    merged = {}
    for channel, rows in (("keyword", lexical), ("semantic", semantic)):
        for rank, chunk in enumerate(rows, 1):
            record = merged.setdefault(
                chunk["id"], {**chunk, "rrf_score": 0.0, "channels": []}
            )
            record["rrf_score"] += 1.0 / (constant + rank)
            record["channels"].append(channel)
            if channel == "keyword":
                record["keyword_score"] = chunk["score"]
                record["matched_terms"] = chunk.get("matched_terms", [])
            else:
                record["semantic_score"] = round(chunk["semantic_score"], 4)
    for record in merged.values():
        record["score"] = round(record["rrf_score"], 6)
    return sorted(
        merged.values(),
        key=lambda c: (-c["rrf_score"], -c.get("semantic_score", 0), c["id"]),
    )[:limit]


@dataclass
class SearchResult:
    sources: list[dict]
    mode: str
    query: str
    elapsed_ms: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    cache_hits: int = 0
    indexed_chunks: int = 0
    rerank_calls: int = 0
    rerank_tokens: int = 0
    rerank_cache_hits: int = 0
    rerank_ms: int = 0
    fallback: str | None = None
    baseline: list[dict] = field(default_factory=list)

    def metadata(self):
        return {
            key: value
            for key, value in vars(self).items()
            if key not in {"sources", "baseline"}
        }


class HybridRetriever:
    def __init__(self, settings, store, embeddings=None, reranker=None):
        self.settings, self.store = settings, store
        self.embeddings = embeddings or (
            QwenEmbeddings(settings, store)
            if settings.retrieval_mode in {"hybrid", "semantic", "rerank"}
            and not settings.offline
            and settings.api_key
            else None
        )
        self.lock = threading.Lock()
        self.unavailable_until = 0.0
        self.reranker = reranker or (
            QwenReranker(settings, store)
            if settings.retrieval_mode == "rerank"
            and not settings.offline
            and settings.api_key
            else None
        )
        self.rerank_unavailable_until = 0.0

    def close(self):
        if self.embeddings:
            self.embeddings.close()
        if self.reranker:
            self.reranker.close()

    def cached_vectors(self, owner, kind, items, result, run_id=None):
        namespace = self.embeddings.namespace
        cached = self.store.vectors(owner, namespace, kind)
        vectors, missing = {}, []
        for key, text in items:
            fingerprint = digest(text)
            entry = cached.get(key)
            if entry and entry["digest"] == fingerprint:
                try:
                    vectors[key] = unit_vector(
                        entry["vector"], self.settings.embedding_dimensions
                    )
                    result.cache_hits += 1
                    continue
                except EmbeddingError:
                    pass
            missing.append((key, text, fingerprint))
        for offset in range(0, len(missing), 10):
            batch = missing[offset : offset + 10]
            values, tokens = self.embeddings.embed(
                [text for _, text, _ in batch], run_id=run_id
            )
            if len(values) != len(batch):
                raise EmbeddingError("向量数量不一致，已使用关键词检索。")
            values = [
                unit_vector(v, self.settings.embedding_dimensions) for v in values
            ]
            result.embedding_calls += 1
            result.embedding_tokens += tokens
            records = []
            for (key, _, fingerprint), value in zip(batch, values):
                vectors[key] = value
                records.append({"key": key, "digest": fingerprint, "vector": value})
            self.store.save_vectors(owner, namespace, kind, records)
            if kind == "document":
                result.indexed_chunks += len(batch)
        return vectors

    def prepare(self, owner, queries, *, run_id=None, stats=None):
        chunks = self.store.chunks(owner)
        result = stats or SearchResult([], "hybrid", "")
        if not self.embeddings:
            raise EmbeddingError("当前使用关键词模式。")
        with self.lock:
            vectors = self.cached_vectors(
                owner,
                "document",
                [(c["id"], c["heading"] + "\n" + c["text"]) for c in chunks],
                result,
                run_id,
            )
            query_vectors = self.cached_vectors(
                owner,
                "query",
                [
                    (digest(normalized(q).strip()), normalized(q).strip())
                    for q in queries
                ],
                result,
                run_id,
            )
        return chunks, vectors, query_vectors, result

    def search(self, owner, query, *, limit=4, run_id=None, rerank_query=None):
        started = time.perf_counter()
        chunks = self.store.chunks(owner)
        lexical = search(chunks, query, limit=12, expand=True)
        result = SearchResult(
            lexical[:limit],
            "keyword",
            rerank_query or query,
            baseline=search(chunks, query, limit=limit),
        )
        result.sources = [
            {**c, "channels": ["keyword"], "keyword_score": c["score"]}
            for c in result.sources
        ]
        if chunks and self.embeddings:
            try:
                if time.monotonic() < self.unavailable_until:
                    raise EmbeddingError("向量服务正在恢复，当前使用关键词检索。")
                chunks, vectors, query_vectors, stats = self.prepare(
                    owner, [query], run_id=run_id, stats=result
                )
                semantic = cosine_ranking(
                    chunks,
                    vectors,
                    query_vectors[digest(normalized(query).strip())],
                    self.settings.semantic_min_score,
                )
                if self.settings.retrieval_mode == "semantic":
                    result.sources = [
                        {
                            **c,
                            "score": round(c["semantic_score"], 4),
                            "channels": ["semantic"],
                        }
                        for c in semantic[:limit]
                    ]
                    result.mode = "semantic"
                else:
                    candidates = fuse(lexical, semantic, 24)
                    result.sources = candidates[:limit]
                    result.mode = "hybrid"
                    if candidates and self.reranker:
                        usage = {}
                        try:
                            if time.monotonic() < self.rerank_unavailable_until:
                                raise RerankError(
                                    "重排服务正在恢复，当前使用融合检索结果。"
                                )
                            ranked = self.reranker.rerank(
                                owner, query, candidates, run_id=run_id, usage=usage
                            )
                            result.sources = ranked.sources[:limit]
                            result.mode = "rerank"
                            result.rerank_cache_hits = ranked.cache_hits
                        except RerankError as error:
                            result.fallback = str(error)
                            self.rerank_unavailable_until = time.monotonic() + 30
                        result.rerank_calls = usage.get("calls", 0)
                        result.rerank_tokens = usage.get("tokens", 0)
                        result.rerank_ms = usage.get("elapsed_ms", 0)
                for key in (
                    "embedding_calls",
                    "embedding_tokens",
                    "cache_hits",
                    "indexed_chunks",
                ):
                    setattr(result, key, getattr(stats, key))
            except EmbeddingError as error:
                result.fallback = str(error)
                self.unavailable_until = time.monotonic() + 30
        result.elapsed_ms = round((time.perf_counter() - started) * 1000)
        return result
