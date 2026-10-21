"""Evaluate answerability decisions and retrieval ranking on labeled questions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

import pipeline
import storage


STATUSES = {"sufficient", "partial", "insufficient"}


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON inválido na linha {line_number}: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"A linha {line_number} deve conter um objeto JSON.")
        question = str(item.get("question", "")).strip()
        expected = item.get("expected_status")
        labels = item.get("relevant_chunks", [])
        if not question:
            raise ValueError(f"A pergunta está vazia na linha {line_number}.")
        if expected not in STATUSES:
            raise ValueError(
                f"expected_status inválido na linha {line_number}; use {sorted(STATUSES)}."
            )
        if not isinstance(labels, list):
            raise ValueError(f"relevant_chunks deve ser uma lista na linha {line_number}.")
        for label in labels:
            if (
                not isinstance(label, dict)
                or not label.get("source_id")
                or "chunk_number" not in label
                or label.get("grade") not in (1, 2)
            ):
                raise ValueError(
                    f"Cada rótulo da linha {line_number} precisa de source_id, "
                    "chunk_number e grade 1 ou 2."
                )
        if expected == "insufficient" and labels:
            raise ValueError(
                f"A pergunta sem resposta da linha {line_number} não deve ter "
                "trechos relevantes rotulados."
            )
        if expected != "insufficient" and not labels:
            raise ValueError(
                f"A pergunta respondível ou parcial da linha {line_number} precisa "
                "ter ao menos um trecho relevante rotulado."
            )
        questions.append(
            {"question": question, "expected_status": expected, "relevant_chunks": labels}
        )
    if not questions:
        raise ValueError("O arquivo não contém perguntas para avaliar.")
    return questions


def _chunk_key(metadata: dict) -> tuple[str, str] | None:
    source_id = metadata.get("source_id")
    chunk_number = metadata.get("chunk_number")
    if source_id is None or chunk_number is None:
        return None
    return str(source_id), str(chunk_number)


def retrieval_metrics(matches: list[dict], labels: list[dict], k: int = 5) -> dict:
    if not labels:
        return {}
    relevance = {
        (str(item["source_id"]), str(item["chunk_number"])): int(item["grade"])
        for item in labels
    }
    retrieved = [
        _chunk_key(match.get("metadata") or {}) for match in matches[:k]
    ]
    retrieved_relevant = [key for key in retrieved if key in relevance]
    recall = len(set(retrieved_relevant)) / len(relevance)
    dcg = sum(
        (2 ** relevance[key] - 1) / math.log2(rank + 2)
        for rank, key in enumerate(retrieved)
        if key in relevance
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 2)
        for rank, grade in enumerate(ideal_grades)
    )
    return {"recall_at_5": recall, "ndcg_at_5": dcg / idcg if idcg else 0.0}


def summarize(results: list[dict]) -> dict:
    confusion = {status: Counter() for status in sorted(STATUSES)}
    for result in results:
        confusion[result["expected_status"]][result["predicted_status"]] += 1

    count = len(results)
    accuracy = sum(
        result["expected_status"] == result["predicted_status"] for result in results
    ) / count
    no_answer = [
        result for result in results if result["expected_status"] == "insufficient"
    ]
    false_accept_rate = (
        sum(item["predicted_status"] != "insufficient" for item in no_answer)
        / len(no_answer)
        if no_answer
        else None
    )
    retrieval = [
        result["retrieval_metrics"]
        for result in results
        if result["retrieval_metrics"]
    ]
    return {
        "question_count": count,
        "accuracy": accuracy,
        "false_accept_rate_on_unanswerable": false_accept_rate,
        "mean_recall_at_5": (
            sum(item["recall_at_5"] for item in retrieval) / len(retrieval)
            if retrieval
            else None
        ),
        "mean_ndcg_at_5": (
            sum(item["ndcg_at_5"] for item in retrieval) / len(retrieval)
            if retrieval
            else None
        ),
        "confusion_matrix": {
            expected: dict(confusion[expected]) for expected in sorted(STATUSES)
        },
    }


def main() -> int:
    load_dotenv(APP_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="Nome ou ID da coleção local.")
    parser.add_argument("--questions", type=Path, required=True, help="Arquivo JSONL rotulado.")
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "").strip(),
        help="Opcional; ex.: BAAI/bge-reranker-v2-m3. Omita para medir a busca atual.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Salva os resultados detalhados em JSON para comparação.",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    chroma_dir = APP_DIR / "data" / "chroma"
    storage.initialize_chroma(chroma_dir)
    collections = storage.list_collections(chroma_dir)
    collection = next(
        (
            item
            for item in collections
            if item["id"] == args.collection or item["name"].casefold() == args.collection.casefold()
        ),
        None,
    )
    if collection is None:
        available = ", ".join(item["name"] for item in collections) or "nenhuma"
        parser.error(f"Coleção não encontrada. Coleções disponíveis: {available}")

    embedding_model = collection["embedding_model"]
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b")
    vectorstore = pipeline.open_chroma_store(
        str(chroma_dir),
        collection["id"],
        OllamaEmbeddings(model=embedding_model, base_url=base_url),
    )
    bm25_index = pipeline.build_bm25_index(
        storage.list_searchable_chunks(chroma_dir, collection["id"])
    )

    reranker_factory = None
    if args.reranker_model:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise SystemExit(
                "Para avaliar o reranker, instale pip install -r requirements-reranker.txt."
            ) from error
        reranker = CrossEncoder(args.reranker_model, max_length=512)
        reranker_factory = lambda: reranker

    llm_factory = lambda: ChatOllama(
        model=chat_model, base_url=base_url, temperature=0.0, num_ctx=8192
    )
    results = []
    for number, item in enumerate(questions, 1):
        decision, matches, error = pipeline.answer_from_materials(
            question=item["question"],
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            llm_factory=llm_factory,
            reranker_factory=reranker_factory,
        )
        if error:
            raise RuntimeError(
                f"Falha ao avaliar a pergunta {number}: {error}"
            ) from error
        result = {
            "question": item["question"],
            "expected_status": item["expected_status"],
            "predicted_status": decision.status,
            "answer": decision.answer,
            "retrieved_chunks": [
                {
                    "source_id": (match.get("metadata") or {}).get("source_id"),
                    "chunk_number": (match.get("metadata") or {}).get("chunk_number"),
                    "filename": match.get("filename"),
                }
                for match in matches
            ],
            "retrieval_metrics": retrieval_metrics(
                matches, item["relevant_chunks"]
            ),
        }
        results.append(result)
        print(
            f"[{number}/{len(questions)}] esperado={result['expected_status']} "
            f"previsto={result['predicted_status']} | {item['question']}"
        )

    report = {
        "collection": collection["name"],
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "reranker_model": args.reranker_model or None,
        "metrics": summarize(results),
        "results": results,
    }
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Relatório detalhado: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
