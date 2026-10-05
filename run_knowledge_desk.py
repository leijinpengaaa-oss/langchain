"""Start the local knowledge desk with the repository's existing Qwen configuration."""

import argparse

import uvicorn

from apps.knowledge_desk.config import Settings
from apps.knowledge_desk.web import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--offline", action="store_true", help="明确标识的离线演示模式，不调用千问"
    )
    args = parser.parse_args()
    uvicorn.run(
        create_app(Settings.load(offline=args.offline)),
        host="127.0.0.1",
        port=args.port,
    )
