import re
import unicodedata

from .llm import ModelError


def compact(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def grounded_citations(answer, evidence, *, require_quotes=True):
    by_id = {c["id"]: c for c in evidence}
    ids = set(answer.citation_ids)
    if ids - by_id.keys():
        raise ModelError("模型引用了未检索到的资料，已阻止输出，请重试。")
    quotes = {}
    for item in answer.supporting_quotes:
        if (
            item.source_id not in ids
            or len(compact(item.quote)) < 8
            or compact(item.quote) not in compact(by_id[item.source_id]["text"])
        ):
            raise ModelError("模型提供的摘录无法在引用原文中核验，已阻止输出，请重试。")
        quotes.setdefault(item.source_id, []).append(item.quote)
    if require_quotes and ids - quotes.keys():
        raise ModelError("回答缺少可核验的原文摘录，已阻止输出，请重试。")
    return [
        {
            **{k: c.get(k) for k in ("id", "document_id", "name", "heading", "page")},
            "quotes": quotes.get(c["id"], []),
        }
        for c in evidence
        if c["id"] in ids
    ]
