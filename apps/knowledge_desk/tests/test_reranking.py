from dataclasses import replace

import pytest

from apps.knowledge_desk.answer_quality import rule_checks, summarize
from apps.knowledge_desk.config import Settings
from apps.knowledge_desk.grounding import grounded_citations
from apps.knowledge_desk.hybrid import HybridRetriever
from apps.knowledge_desk.llm import ModelError
from apps.knowledge_desk.reranking import QwenReranker, validate_ratings
from apps.knowledge_desk.retrieval import parse_document
from apps.knowledge_desk.schemas import GroundedAnswer, RankingDecision
from apps.knowledge_desk.store import Store


class RankingModel:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def close(self):
        pass

    def complete(self, run_id, purpose, instruction, payload, schema, **kwargs):
        self.calls += 1
        kwargs["usage"].update(calls=1, tokens=71, elapsed_ms=4)
        if self.fail:
            raise ModelError("模型未返回符合约定的数据，可从检查点重试。")
        return RankingDecision(
            ratings=[
                {"index": d["index"], "grade": 3 if "等待" in d["text"] else 1}
                for d in payload["documents"]
            ]
        )


class Vectors:
    namespace = "vectors-test"

    def embed(self, texts, **kwargs):
        return [[1.0, 0.0] for t in texts], 4

    def close(self):
        pass


@pytest.fixture
def ranking_corpus(tmp_path):
    settings = Settings(
        data_dir=tmp_path, embedding_dimensions=2, retrieval_mode="rerank"
    )
    store = Store(tmp_path / "workspace.sqlite")
    doc, _ = store.add_document(
        "alice",
        *parse_document(
            "test.md",
            "## 停机\n服务退出前需要等待正在运行的任务，最多等待45秒。\n## 频次\n每分钟可以调用60次请求。".encode(),
        ),
    )
    model = RankingModel()
    reranker = QwenReranker(settings, store, model)
    return settings, store, doc, model, reranker


def test_rerank_filters_topic_only_and_cache_survives_restart(ranking_corpus):
    settings, store, _, model, reranker = ranking_corpus
    chunks = store.chunks("alice")
    first = reranker.rerank("alice", "how to shutdown", chunks)
    assert [c["heading"] for c in first.sources] == ["停机"]
    assert first.calls == 1 and first.tokens == 71 and first.cache_hits == 0
    second = QwenReranker(settings, store, model).rerank(
        "alice", "how to shutdown", chunks
    )
    assert second.cache_hits == 1 and second.calls == 0 and model.calls == 1
    assert not store.cached_ranking("bob", next_cache_key(store))


def next_cache_key(store):
    with store.connection() as db:
        return db.execute("SELECT cache_key FROM ranking_cache LIMIT 1").fetchone()[0]


def test_rerank_cache_changes_with_content_and_model_and_delete(ranking_corpus):
    settings, store, doc, model, reranker = ranking_corpus
    chunks = store.chunks("alice")
    reranker.rerank("alice", "query", chunks)
    changed = [{**c, "text": c["text"] + "版本变化"} for c in chunks]
    reranker.rerank("alice", "query", changed)
    QwenReranker(replace(settings, model="other-model"), store, model).rerank(
        "alice", "query", chunks
    )
    assert model.calls == 3
    with store.connection() as db:
        db.execute("UPDATE ranking_cache SET ratings='bad-json'")
    assert reranker.rerank("alice", "query", chunks).calls == 1
    store.delete_document("alice", doc["id"])
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM ranking_cache").fetchone()[0] == 0


@pytest.mark.parametrize(
    "ratings",
    [
        [{"index": 0, "grade": 3}],
        [{"index": 0, "grade": 3}, {"index": 0, "grade": 2}],
        [{"index": 0, "grade": 3}, {"index": 9, "grade": 2}],
        [{"index": 0, "grade": True}, {"index": 1, "grade": 2}],
        [{"index": "0", "grade": 3}, {"index": 1, "grade": 2}],
    ],
)
def test_untrusted_ranking_cannot_omit_duplicate_or_invent_candidates(ratings):
    with pytest.raises(ValueError):
        validate_ratings(ratings, 2)


def test_rerank_failure_returns_hybrid_and_records_consumed_tokens(ranking_corpus):
    settings, store, _, model, reranker = ranking_corpus
    model.fail = True
    retriever = HybridRetriever(settings, store, Vectors(), reranker)
    first = retriever.search("alice", "停机")
    assert first.mode == "hybrid" and first.sources and first.fallback
    assert first.rerank_calls == 1 and first.rerank_tokens == 71
    second = retriever.search("alice", "停机")
    assert second.mode == "hybrid" and second.rerank_calls == 0 and model.calls == 1


def test_empty_rerank_result_does_not_fake_a_provider_failure(ranking_corpus):
    settings, store, _, _, _ = ranking_corpus

    class TopicOnly(RankingModel):
        def complete(self, run_id, purpose, instruction, payload, schema, **kwargs):
            return RankingDecision(
                ratings=[
                    {"index": d["index"], "grade": 1} for d in payload["documents"]
                ]
            )

    retriever = HybridRetriever(
        settings, store, Vectors(), QwenReranker(settings, store, TopicOnly())
    )
    result = retriever.search("alice", "备份密码是多少")
    assert result.mode == "rerank" and not result.sources and result.fallback is None


def test_semantic_mode_is_a_separate_baseline(ranking_corpus):
    settings, store, _, _, _ = ranking_corpus
    result = HybridRetriever(
        replace(settings, retrieval_mode="semantic"), store, Vectors()
    ).search("alice", "let the process exit")
    assert result.mode == "semantic" and result.sources
    assert all(c["channels"] == ["semantic"] for c in result.sources)


SOURCE = {
    "id": "a",
    "text": "星舟数据库每天02:30执行一次全量备份，保留14天。",
    "heading": "备份",
}


def test_exact_excerpt_is_preserved_in_citations():
    result = GroundedAnswer(
        answer="每天02:30备份，保留14天。",
        citation_ids=["a"],
        supporting_quotes=[
            {"source_id": "a", "quote": "每天02：30执行一次全量备份，保留14天。"}
        ],
    )
    citations = grounded_citations(result, [SOURCE])
    assert citations[0]["quotes"] == [result.supporting_quotes[0].quote]


@pytest.mark.parametrize(
    "ids,quotes",
    [
        (["foreign"], []),
        (["a"], []),
        (["a"], [{"source_id": "a", "quote": "保留期是30天，密码是123456"}]),
        (["a"], [{"source_id": "a", "quote": "         "}]),
        ([], [{"source_id": "a", "quote": SOURCE["text"]}]),
    ],
)
def test_invalid_or_unattributed_excerpts_are_blocked(ids, quotes):
    with pytest.raises(ModelError):
        grounded_citations(
            GroundedAnswer(answer="回答", citation_ids=ids, supporting_quotes=quotes),
            [SOURCE],
        )


def test_refusal_and_positive_answer_checks_have_different_denominators():
    negative = {"kind": "unsupported", "headings": [], "required_patterns": []}
    result = rule_checks(negative, "知识库没有提供足够依据。", [], [])
    assert result["refusal_rule_pass"] and result["facts_pattern_match"] is None
    positive = {
        "kind": "answerable",
        "headings": ["备份"],
        "required_patterns": [r"14天"],
    }
    result = rule_checks(
        positive,
        "保留30天",
        [{"id": "a", "heading": "备份", "quotes": [SOURCE["text"]]}],
        [SOURCE],
    )
    assert result["excerpts_resolve"] and not result["facts_pattern_match"]
    wrongly_refused = {
        "kind": "answerable",
        "citations": [],
        "elapsed_seconds": 1,
        "judgment": {"citations_support_answer": True},
    }
    assert summarize([wrongly_refused])["judge_supported_answers"] == 0


def test_ranking_cache_has_a_per_owner_bound(ranking_corpus):
    _, store, _, _, _ = ranking_corpus
    for i in range(258):
        store.save_ranking("alice", str(i), [{"index": 0, "grade": 3}])
    store.save_ranking("bob", "separate", [{"index": 0, "grade": 2}])
    with store.connection() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM ranking_cache WHERE owner='alice'"
            ).fetchone()[0]
            == 256
        )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM ranking_cache WHERE owner='bob'"
            ).fetchone()[0]
            == 1
        )
