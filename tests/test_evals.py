import json

import pytest

from evals.evaluate_answers import load_questions, retrieval_metrics, summarize


def test_load_questions_checks_schema_and_no_answer_labels(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Pergunta sem resposta",
                "expected_status": "insufficient",
                "relevant_chunks": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_questions(path)[0]["expected_status"] == "insufficient"

    path.write_text(
        json.dumps(
            {
                "question": "Pergunta sem resposta",
                "expected_status": "insufficient",
                "relevant_chunks": [
                    {"source_id": "a", "chunk_number": 1, "grade": 2}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="não deve ter"):
        load_questions(path)


def test_retrieval_metrics_reward_relevant_chunks_near_the_top():
    labels = [
        {"source_id": "a", "chunk_number": 1, "grade": 2},
        {"source_id": "b", "chunk_number": 2, "grade": 1},
    ]
    at_top = retrieval_metrics(
        [
            {"metadata": {"source_id": "a", "chunk_number": 1}},
            {"metadata": {"source_id": "b", "chunk_number": 2}},
        ],
        labels,
    )
    at_bottom = retrieval_metrics(
        [
            {"metadata": {"source_id": "x", "chunk_number": 1}},
            {"metadata": {"source_id": "a", "chunk_number": 1}},
        ],
        labels,
    )

    assert at_top["recall_at_5"] == 1.0
    assert at_top["ndcg_at_5"] > at_bottom["ndcg_at_5"]


def test_summary_reports_false_accepts_for_unanswerable_questions():
    metrics = summarize(
        [
            {
                "expected_status": "insufficient",
                "predicted_status": "partial",
                "retrieval_metrics": {},
            },
            {
                "expected_status": "sufficient",
                "predicted_status": "sufficient",
                "retrieval_metrics": {},
            },
        ]
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["false_accept_rate_on_unanswerable"] == 1.0
