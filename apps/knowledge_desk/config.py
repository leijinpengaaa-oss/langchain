from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    data_dir: Path = APP_DIR / "data"
    offline: bool = False
    retrieval_mode: str = "rerank"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 256
    semantic_min_score: float = 0.45

    @classmethod
    def load(cls, env_file: Path | None = None, *, offline: bool = False):
        values = dotenv_values(
            env_file or ROOT / ".env", encoding="utf-8-sig", interpolate=False
        )

        def get(names, default=""):
            return next(
                (
                    str(values.get(n) or os.environ.get(n)).strip()
                    for n in names
                    if values.get(n) or os.environ.get(n)
                ),
                default,
            )

        key = get(
            ["DASHSCOPE_API_KEY", "LLM_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"]
        )
        base = get(["QWEN_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL"], cls.base_url)
        model = get(["QWEN_MODEL", "LLM_MODEL", "OPENAI_MODEL"], cls.model)
        url = urlsplit(base)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("模型接口必须是无凭据、无查询参数的 HTTPS 地址。")
        mode = get(["RETRIEVAL_MODE"], cls.retrieval_mode)
        dimensions = int(get(["QWEN_EMBEDDING_DIMENSIONS"], "256"))
        minimum = float(get(["SEMANTIC_MIN_SCORE"], "0.45"))
        if mode not in {"keyword", "semantic", "hybrid", "rerank"}:
            raise ValueError(
                "RETRIEVAL_MODE 必须是 keyword、semantic、hybrid 或 rerank。"
            )
        if dimensions not in {64, 128, 256, 512, 768, 1024, 1536, 2048}:
            raise ValueError("向量维度配置无效。")
        if not 0 <= minimum <= 1:
            raise ValueError("SEMANTIC_MIN_SCORE 必须在0到1之间。")
        return cls(
            api_key=key,
            base_url=base.rstrip("/"),
            model=model,
            offline=offline,
            retrieval_mode=mode,
            embedding_model=get(["QWEN_EMBEDDING_MODEL"], "text-embedding-v4"),
            embedding_dimensions=dimensions,
            semantic_min_score=minimum,
        )
