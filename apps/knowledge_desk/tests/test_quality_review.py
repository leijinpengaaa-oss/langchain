import copy

import pytest

from apps.knowledge_desk.quality_review import (
    import_review,
    render_review,
    review_template,
)


@pytest.fixture
def review_report():
    return {
        "dataset_sha256": "test-dataset",
        "cases": [
            {
                "id": "one",
                "mode": "rerank",
                "answer": "保留14天",
                "question": "<script>alert(1)</script>",
                "expected_facts": ["14天"],
                "citations": [],
                "cited_sources": [],
            }
        ],
    }


def test_unfilled_template_is_not_presented_as_human_review(review_report):
    template = review_template(review_report)
    template["reviewer"] = "测试复核人员"
    with pytest.raises(ValueError, match="true或false"):
        import_review(review_report, template)


def test_human_review_is_bound_to_the_exact_answers(review_report):
    template = review_template(review_report)
    template["reviewer"] = "测试复核人员"
    template["reviews"][0].update(
        answer_correct=True, citations_support_answer=True, refusal_correct=True
    )
    assert import_review(review_report, template)["reviews"][0]["answer_correct"]
    changed = copy.deepcopy(review_report)
    changed["cases"][0]["answer"] = "保留30天"
    with pytest.raises(ValueError, match="答案不一致"):
        import_review(changed, template)
    template["reviews"] *= 2
    with pytest.raises(ValueError, match="不得重复"):
        import_review(review_report, template)


def test_review_html_escapes_untrusted_question_text(review_report):
    html = render_review(review_report)
    assert "<script>" not in html and "&lt;script&gt;" in html
