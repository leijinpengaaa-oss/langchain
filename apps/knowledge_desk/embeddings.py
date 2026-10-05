from __future__ import annotations

import hashlib
import math
import time

import httpx


class EmbeddingError(RuntimeError):
    pass


def unit_vector(value, dimensions):
    if not isinstance(value, list) or len(value) != dimensions:
        raise EmbeddingError("向量接口返回维度不匹配，已使用关键词检索。")
    try:
        valid = all(type(n) in (float, int) and math.isfinite(n) for n in value)
        norm = math.hypot(*value) if valid else 0
    except (OverflowError, TypeError):
        norm = 0
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("向量接口返回无效向量，已使用关键词检索。")
    return [n / norm for n in value]


class QwenEmbeddings:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.namespace = hashlib.sha256(
            f"{settings.base_url}|{settings.embedding_model}|{settings.embedding_dimensions}|heading-text-v1".encode()
        ).hexdigest()
        self.client = httpx.Client(timeout=httpx.Timeout(30, connect=5))

    def close(self):
        self.client.close()

    def embed(self, texts, *, run_id=None):
        if not self.settings.api_key:
            raise EmbeddingError("未配置向量接口密钥，已使用关键词检索。")
        started = time.perf_counter()
        try:
            response = self.client.post(
                self.settings.base_url + "/embeddings",
                headers={"Authorization": "Bearer " + self.settings.api_key},
                json={
                    "model": self.settings.embedding_model,
                    "input": texts,
                    "dimensions": self.settings.embedding_dimensions,
                    "encoding_format": "float",
                },
            )
        except httpx.RequestError:
            raise EmbeddingError("向量服务连接失败，已使用关键词检索。") from None
        if response.is_error:
            raise EmbeddingError(
                f"向量服务暂不可用（HTTP {response.status_code}），已使用关键词检索。"
            )
        try:
            body = response.json()
            tokens = int(body.get("usage", {}).get("total_tokens", 0))
            elapsed = (time.perf_counter() - started) * 1000
            if run_id:
                self.store.usage(run_id, tokens, elapsed)
                self.store.event(
                    run_id,
                    "usage",
                    {
                        "purpose": "文本向量化",
                        "tokens": tokens,
                        "elapsed_ms": round(elapsed),
                        "model": self.settings.embedding_model,
                    },
                )
            data = body["data"]
            if len(data) != len(texts) or {item["index"] for item in data} != set(
                range(len(texts))
            ):
                raise ValueError("incomplete response")
            vectors = [
                unit_vector(item["embedding"], self.settings.embedding_dimensions)
                for item in sorted(data, key=lambda item: item["index"])
            ]
            return vectors, tokens
        except (KeyError, TypeError, ValueError):
            raise EmbeddingError("向量接口返回格式无效，已使用关键词检索。") from None
