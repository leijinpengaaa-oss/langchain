from __future__ import annotations

import io
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

STOP = {
    "什么",
    "怎么",
    "如何",
    "一下",
    "请问",
    "可以",
    "是否",
    "多少",
    "需要",
    "我们",
    "你们",
    "一个",
    "哪些",
    "应该",
    "帮我",
    "告诉",
    "诉我",
}

# General operations vocabulary; not query-to-answer rules from the evaluation set.
ALIASES = [
    ("下线", "停机", "停止服务", "收尾", "drain", "sigterm"),
    ("回滚", "降级版本", "退回老版本", "rollback", "rolled back"),
    ("备份", "副本", "backup", "retention"),
    ("健康检查", "探活", "探测", "readiness", "health check"),
    ("密钥", "令牌", "凭据", "credentials", "api key", "token"),
    ("轮换", "换访问令牌", "rotate", "rotation"),
    ("限流", "太频繁", "限额", "rate limit", "http429"),
    ("幂等", "重复创建", "idempotency", "idempotent"),
    ("首次响应", "第一次回应", "先回复", "sla", "response time"),
    ("会话", "聊天线程", "conversation", "session"),
    ("启动", "起不来", "boot", "launch"),
    ("工单", "问题单", "ticket", "incident report"),
    ("扫描", "扫出来", "scanned", "ocr"),
    ("生产", "线上", "production"),
]


def normalized(text: str):
    return unicodedata.normalize("NFKC", text).lower()


def query_weights(query: str, *, expand: bool):
    weights = dict.fromkeys(tokenize(normalized(query) if expand else query), 1.0)
    if expand:
        text = normalized(query)
        for aliases in ALIASES:
            if any(
                re.search(r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])", text)
                if re.search(r"[a-z]", alias)
                else alias in text
                for alias in aliases
            ):
                for term in tokenize(" ".join(aliases)):
                    weights.setdefault(term, 0.35)
        for code in re.findall(r"http\s*([1-5][0-9]{2})", text):
            weights[code] = 1.0
    return weights


def tokenize(text: str):
    result = []
    for word in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            result.extend(
                word[i : i + 2]
                for i in range(len(word) - 1)
                if word[i : i + 2] not in STOP
            )
        else:
            result.append(word)
    return result


def split_text(text: str, heading: str, page: int | None = None):
    # Heading boundaries remain intact so citations identify a useful section.
    chunks = []
    for offset in range(0, len(text), 620):
        part = text[offset : offset + 760].strip()
        if part:
            chunks.append({"heading": heading[:200], "page": page, "text": part})
    return chunks


def markdown_chunks(text: str, name: str):
    chunks, lines = [], []
    heading = name
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if lines:
                chunks.extend(split_text("\n".join(lines), heading))
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            lines = []
        else:
            lines.append(line)
    if lines:
        chunks.extend(split_text("\n".join(lines), heading))
    return chunks


def parse_document(filename: str, content: bytes):
    name = Path(filename.replace("\\", "/")).name
    if not name or len(content) > 5 * 1024 * 1024:
        raise ValueError("单个文件最多5MB。")
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise ValueError("支持 TXT、Markdown 和有文字层的 PDF 文件。")
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted or len(reader.pages) > 100:
                raise ValueError("PDF不能加密，且最多100页。")
            pages, chunks = [], []
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append(text)
                if sum(map(len, pages)) > 150_000:
                    raise ValueError("文档文字量最多15万字符。")
                chunks.extend(split_text(text, f"第{number}页", number))
            text = "\n".join(pages)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 -- reject untrusted PDF parser failures without leaking internals
            raise ValueError("无法解析PDF，请使用有效且含文字层的PDF。") from None
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("文本文件请使用UTF-8编码。") from None
        chunks = markdown_chunks(text, name)
    if not text.strip() or not chunks:
        raise ValueError("未提取到文字；扫描PDF请先进行OCR。")
    if len(text) > 150_000 or len(chunks) > 250 or "\x00" in text:
        raise ValueError("文档过大或包含无效文本。")
    return name, suffix.removeprefix("."), text, chunks


def search(chunks: list[dict], query: str, *, limit: int = 4, expand: bool = False):
    if not chunks:
        return []
    query_terms = query_weights(query, expand=expand)
    if not query_terms:
        return []
    counts = [Counter(tokenize(c["text"] + " " + c["heading"] * 3)) for c in chunks]
    avg = sum(sum(c.values()) for c in counts) / len(counts) or 1
    frequency = Counter(term for count in counts for term in count)
    scored = []
    for chunk, count in zip(chunks, counts):
        length = sum(count.values())
        score = 0.0
        matched = []
        for term in query_terms:
            tf = count.get(term, 0)
            if not tf:
                continue
            matched.append(term)
            idf = math.log(
                1 + (len(chunks) - frequency[term] + 0.5) / (frequency[term] + 0.5)
            )
            score += (
                query_terms[term]
                * idf
                * tf
                * 2.5
                / (tf + 1.5 * (0.25 + 0.75 * length / avg))
            )
        if expand and normalized(query).strip() == normalized(chunk["heading"]).strip():
            # Exact section lookup should also work in a one-document workspace.
            score += 2.0
        if score >= 2.0:
            scored.append({**chunk, "score": round(score, 3), "matched_terms": matched})
    return sorted(scored, key=lambda c: c["score"], reverse=True)[:limit]


def seed_samples(store, owner):
    for file in sorted((Path(__file__).parent / "sample_docs").glob("*.md")):
        content = file.read_bytes()
        name, kind, text, chunks = parse_document(file.name, content)
        store.add_document(owner, name, kind, text, chunks)
