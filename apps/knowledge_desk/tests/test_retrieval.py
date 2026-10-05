from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.knowledge_desk.config import Settings
from apps.knowledge_desk.embeddings import EmbeddingError, QwenEmbeddings, unit_vector
from apps.knowledge_desk.hybrid import HybridRetriever, fuse
from apps.knowledge_desk.retrieval import parse_document, search
from apps.knowledge_desk.store import Store
from apps.knowledge_desk.web import create_app


class FakeEmbeddings:
    namespace = "test-vectors-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts, *, run_id=None):
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if "停机" in text or "shutdown" in text else [0.0, 1.0]
            for text in texts
        ], len(texts)

    def close(self):
        pass


@pytest.fixture
def corpus(tmp_path):
    settings = Settings(data_dir=tmp_path, embedding_dimensions=2)
    store = Store(tmp_path / "workspace.sqlite")
    content = "## 停机\n服务退出前等待正在运行的任务。\n## 请求频次\n每分钟只接受60次请求。".encode()
    name, kind, text, chunks = parse_document("guide.md", content)
    doc, _ = store.add_document("alice", name, kind, text, chunks)
    return settings, store, doc


def test_semantic_search_cache_survives_restart_and_is_owner_scoped(corpus):
    settings, store, _ = corpus
    provider = FakeEmbeddings()
    retriever = HybridRetriever(settings, store, provider)
    query = "let the process finish before shutdown"
    assert not search(store.chunks("alice"), query)
    first = retriever.search("alice", query)
    assert first.mode == "hybrid" and first.sources[0]["heading"] == "停机"
    assert first.sources[0]["channels"] == ["semantic"]
    assert first.indexed_chunks == 2 and len(provider.calls) == 2
    second = retriever.search("alice", query)
    assert second.embedding_calls == 0 and second.cache_hits == 3
    assert not retriever.search("bob", query).sources
    assert not store.vectors("bob", provider.namespace, "document")
    replacement = FakeEmbeddings()
    restored = HybridRetriever(settings, store, replacement).search("alice", query)
    assert restored.sources[0]["id"] == first.sources[0]["id"]
    assert replacement.calls == []


def test_deleted_document_cascades_vector_cache(corpus):
    settings, store, doc = corpus
    retriever = HybridRetriever(settings, store, FakeEmbeddings())
    retriever.search("alice", "shutdown")
    assert len(store.vectors("alice", retriever.embeddings.namespace, "document")) == 2
    store.delete_document("alice", doc["id"])
    assert not store.vectors("alice", retriever.embeddings.namespace, "document")
    assert not retriever.search("alice", "shutdown").sources


def test_provider_failure_falls_back_and_does_not_fake_hybrid(corpus):
    settings, store, _ = corpus

    class Unavailable(FakeEmbeddings):
        def embed(self, texts, *, run_id=None):
            self.calls.append(texts)
            raise EmbeddingError("向量服务不可用，已使用关键词检索。")

    provider = Unavailable()
    retriever = HybridRetriever(settings, store, provider)
    result = retriever.search("alice", "停机")
    assert result.mode == "keyword" and result.fallback
    assert result.sources[0]["heading"] == "停机"
    assert result.sources[0]["channels"] == ["keyword"]
    retriever.search("alice", "停机")
    assert len(provider.calls) == 1


def test_cache_namespace_change_reindexes_and_corruption_repairs(corpus):
    settings, store, _ = corpus
    first = HybridRetriever(settings, store, FakeEmbeddings())
    first.search("alice", "shutdown")
    with store.connection() as db:
        db.execute("UPDATE vectors SET vector='[NaN,0]' WHERE kind='document'")
    assert first.search("alice", "shutdown").indexed_chunks == 2
    second_provider = FakeEmbeddings()
    second_provider.namespace = "different-model-or-dimension"
    result = HybridRetriever(settings, store, second_provider).search(
        "alice", "shutdown"
    )
    assert result.indexed_chunks == 2 and result.embedding_calls == 2


def test_rrf_merges_by_rank_not_incomparable_scores():
    lexical = [{"id": "a", "score": 99}, {"id": "b", "score": 2}]
    semantic = [{"id": "b", "semantic_score": 0.9}, {"id": "c", "semantic_score": 0.8}]
    result = fuse(lexical, semantic)
    assert result[0]["id"] == "b"
    assert result[0]["channels"] == ["keyword", "semantic"]
    assert len({r["id"] for r in result}) == 3


@pytest.mark.parametrize(
    "value", [[0, 0], [float("nan"), 0], [float("inf"), 0], [True, 0], [1], "bad"]
)
def test_invalid_vectors_are_rejected(value):
    with pytest.raises(EmbeddingError):
        unit_vector(value, 2)


def test_embedding_protocol_order_usage_and_safe_error(corpus):
    settings, store, _ = corpus
    settings = replace(settings, api_key="test-secret")
    provider = QwenEmbeddings(settings, store)

    def respond(request):
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 2]},
                    {"index": 0, "embedding": [2, 0]},
                ],
                "usage": {"total_tokens": 12},
            },
        )

    provider.client.close()
    provider.client = httpx.Client(transport=httpx.MockTransport(respond))
    vectors, tokens = provider.embed(["a", "b"])
    assert vectors == [[1, 0], [0, 1]] and tokens == 12
    provider.client.close()
    provider.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                403, text="test-secret internal provider message"
            )
        )
    )
    with pytest.raises(EmbeddingError) as error:
        provider.embed(["a"])
    assert "test-secret" not in str(error.value) and "403" in str(error.value)
    provider.close()


def test_query_cache_is_bounded_and_foreign_chunks_not_saved(corpus):
    _, store, _ = corpus
    records = [{"key": str(i), "digest": str(i), "vector": [1, 0]} for i in range(260)]
    store.save_vectors("alice", "namespace", "query", records)
    assert len(store.vectors("alice", "namespace", "query")) == 256
    chunk = store.chunks("alice")[0]
    store.save_vectors(
        "bob",
        "namespace",
        "document",
        [{"key": chunk["id"], "digest": "x", "vector": [1, 0]}],
    )
    assert not store.vectors("bob", "namespace", "document")


def test_search_preview_validates_query_and_returns_only_own_sources(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, offline=True))) as client:
        client.get("/api/bootstrap")
        result = client.post("/api/search", json={"query": "星舟下线如何收尾？"}).json()
        assert result["mode"] == "keyword" and result["sources"]
        assert any(c["heading"] == "优雅停机" for c in result["sources"])
        assert client.post("/api/search", json={"query": " "}).status_code == 400
        assert client.post("/api/search", json={"query": "x" * 2001}).status_code == 422
        client.cookies.clear()
        assert not client.post("/api/search", json={"query": "星舟"}).json()["sources"]
