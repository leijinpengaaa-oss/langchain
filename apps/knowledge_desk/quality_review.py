"""Render paired evidence and import human ratings without making API requests."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from .config import APP_DIR
from .hybrid import digest
from .store import now

SCORE_KEYS = ("answer_correct", "citations_support_answer", "refusal_correct")


def answers_fingerprint(report):
    return digest(
        json.dumps(
            [
                {k: row.get(k) for k in ("id", "mode", "answer", "citations", "error")}
                for row in report["cases"]
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def review_template(report):
    return {
        "dataset_sha256": report["dataset_sha256"],
        "answers_sha256": answers_fingerprint(report),
        "reviewer": "",
        "instructions": "先阅读HTML中问题、预期事实、答案及引用原文，再填写三个布尔评分和备注。资料支持的额外事实不因超出预期事实而自动判错；同时检查时序和问题场景。所有null须改为true/false后才可导入。",
        "reviews": [
            {"id": r["id"], "mode": r["mode"], **dict.fromkeys(SCORE_KEYS), "notes": ""}
            for r in report["cases"]
        ],
    }


def import_review(report, review):
    if review.get("dataset_sha256") != report["dataset_sha256"] or review.get(
        "answers_sha256"
    ) != answers_fingerprint(report):
        raise ValueError("评分文件与本次数据或答案不一致，不能导入。")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("请填写复核人员。")
    expected = {(r["id"], r["mode"]) for r in report["cases"]}
    entries = review.get("reviews", [])
    if not isinstance(entries, list) or any(not isinstance(r, dict) for r in entries):
        raise ValueError("评分记录格式无效。")
    if any(
        not isinstance(r.get("id"), str) or not isinstance(r.get("mode"), str)
        for r in entries
    ):
        raise ValueError("评分的问题ID及模式必须是字符串。")
    keys = [(r.get("id"), r.get("mode")) for r in entries]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("评分必须覆盖每个问题及模式一次，不得重复或遗漏。")
    if any(type(r.get(k)) is not bool for r in entries for k in SCORE_KEYS):
        raise ValueError("所有评分必须是true或false，不能保留null或字符串。")
    return {
        "created_at": now(),
        "source": "用户导入的人工复核评分；程序只验证格式与答案版本，不验证复核人员身份",
        **review,
    }


def render_review(report):
    def card(row, label):
        if row is None:
            return f"<article><h3>{label}</h3><p>尚未评测</p></article>"
        judgment = row.get("judgment", {})
        status = " · ".join(
            f"{name}：{'通过' if judgment.get(key) else '待复核'}"
            for name, key in (
                ("答案", "answer_correct"),
                ("引用", "citations_support_answer"),
            )
        )
        sources = "".join(
            f"<details><summary>{escape(c['name'])} · {escape(c['heading'])}</summary><blockquote>{escape(c['text'])}</blockquote></details>"
            for c in row.get("cited_sources", [])
        )
        return f"<article><h3>{label}</h3><p class='status'>模型辅助评审 · {status}</p><p class='answer'>{escape(row.get('answer', row.get('error', '')))}</p><p class='reason'>{escape(judgment.get('explanation', ''))}</p>{sources}<small>{row.get('tokens', 0)} tokens · {row.get('elapsed_seconds', 0)} 秒</small></article>"

    questions = dict.fromkeys(row["id"] for row in report["cases"])
    paired = {(r["id"], r["mode"]): r for r in report["cases"]}
    sections = []
    for key in questions:
        case = next(r for r in report["cases"] if r["id"] == key)
        facts = "；".join(case["expected_facts"])
        sections.append(
            f"<section><h2>{escape(case['question'])}</h2><p>预期：{escape(facts)}</p><div class='pair'>{card(paired.get((key, 'hybrid')), 'RRF 对照')}{card(paired.get((key, 'rerank')), '千问重排')}</div></section>"
        )
    return (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>知行 · 答案质量逐题复核</title><style>
body{margin:0;background:#f4f6f3;color:#23392f;font-family:system-ui,sans-serif;line-height:1.7}main{max-width:1150px;margin:auto;padding:32px}h1{font-size:28px}h2{font-size:19px}section{margin:28px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{background:white;padding:20px;border:1px solid #dce4dd;border-radius:12px}.answer,blockquote{white-space:pre-wrap;overflow-wrap:anywhere}.reason,small{color:#647168;font-size:13px}.status{color:#936014;font-size:13px}summary{cursor:pointer}blockquote{border-left:3px solid #aac5b4;margin:10px 0;padding-left:12px}@media(max-width:700px){main{padding:16px}.pair{grid-template-columns:1fr}}
</style><main><h1>答案质量 · 逐题对照与复核</h1><p>24条自编问题；两组采用相同答案提示和摘录校验。模型评审由同一千问模型完成，存在偏差，不是人工盲审。展开引用原文后独立判断事实、场景和时序；原文中有依据的扩展信息不应仅因未出现在预期事实中而判错。</p><p>填写 human-review-template-v3.json 后，可用 quality_review --import-review 导入独立评分；原始模型评审保留。</p>"""
        + "".join(sections)
        + "</main></html>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-review", type=Path)
    args = parser.parse_args()
    directory = APP_DIR / "reports"
    report = json.loads(
        (directory / "answer-quality-v3.json").read_text(encoding="utf-8")
    )
    if args.import_review:
        reviewed = import_review(
            report, json.loads(args.import_review.read_text(encoding="utf-8"))
        )
        path = directory / "human-review-v3.json"
        path.write_text(
            json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        path = directory / "answer-quality-v3.html"
        path.write_text(render_review(report), encoding="utf-8")
        template = directory / "human-review-template-v3.json"
        if not template.exists():
            template.write_text(
                json.dumps(review_template(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            existing = json.loads(template.read_text(encoding="utf-8"))
            if existing.get("answers_sha256") != answers_fingerprint(report):
                # Keep a user's existing scoring template intact after a new run.
                template = directory / (
                    "human-review-template-"
                    + answers_fingerprint(report)[:12]
                    + ".json"
                )
                template.write_text(
                    json.dumps(review_template(report), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        print("Template:", template)
    print("Report:", path)


if __name__ == "__main__":
    main()
