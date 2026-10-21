import json
from io import BytesIO

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from reportlab.pdfgen import canvas

import pipeline


def make_pdf(*page_texts: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for page_text in page_texts:
        pdf.drawString(72, 720, page_text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_extract_pdf_text_and_page_metadata():
    kind, language, content, documents = pipeline.extract_file(
        "aula.pdf", make_pdf("Conteúdo da primeira página")
    )

    assert kind == "PDF"
    assert language == "Texto extraído"
    assert "Conteúdo da primeira página" in content
    assert documents[0].metadata["page_number"] == 1


def test_pdf_page_limit_is_enforced():
    pdf_bytes = make_pdf("Página um", "Página dois")

    with pytest.raises(ValueError, match="limite atual é 1"):
        pipeline.extract_file("aula.pdf", pdf_bytes, max_pdf_pages=1)


def test_split_documents_keeps_page_metadata_and_chunk_limit():
    source = Document(
        page_content="conceito importante " * 200,
        metadata={"page_number": 3},
    )

    chunks = pipeline.split_documents([source], "biologia.pdf")

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 1400 for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 3 for chunk in chunks)
    assert all(chunk.metadata["source_name"] == "biologia.pdf" for chunk in chunks)


def test_retrieve_chunks_returns_the_nearest_results():
    class FakeVectorStore:
        def similarity_search(self, **kwargs):
            assert kwargs["k"] == 20
            assert kwargs["filter"] == {"record_type": "chunk"}
            return [
                Document(page_content="primeiro trecho", metadata={"filename": "a.txt"}),
                Document(page_content="segundo trecho", metadata={"filename": "b.txt"}),
            ]

    matches = pipeline.retrieve_chunks(FakeVectorStore(), "pergunta", k=5)

    assert len(matches) == 2
    assert matches[0]["filename"] == "a.txt"
    assert "score" not in matches[0]


def test_bm25_finds_exact_lexical_terms():
    index = pipeline.build_bm25_index(
        [
            {
                "id": "chunk:a:1",
                "content": "A fotossíntese transforma luz em energia com clorofila.",
                "metadata": {"filename": "biologia.txt"},
            },
            {
                "id": "chunk:b:1",
                "content": "A respiração celular libera energia da glicose.",
                "metadata": {"filename": "célula.txt"},
            },
        ]
    )

    results = index.search("fotossíntese clorofila", limit=2)

    assert results[0]["id"] == "chunk:a:1"
    assert results[0]["score"] > 0


def test_rrf_fuses_bm25_and_chroma_rankings():
    chunk_a = Document(
        page_content="quantum plasma especial",
        metadata={"source_id": "a", "chunk_number": 1, "filename": "a.txt"},
    )
    chunk_b = Document(
        page_content="conteúdo encontrado semanticamente",
        metadata={"source_id": "b", "chunk_number": 1, "filename": "b.txt"},
    )

    class FakeVectorStore:
        def similarity_search(self, **kwargs):
            assert kwargs["k"] == 20
            return [chunk_b, chunk_a]

    bm25_index = pipeline.build_bm25_index(
        [
            {
                "id": "chunk:a:1",
                "content": chunk_a.page_content,
                "metadata": chunk_a.metadata,
            },
            {
                "id": "chunk:c:1",
                "content": "quantum plasma estudo",
                "metadata": {"source_id": "c", "chunk_number": 1, "filename": "c.txt"},
            },
        ]
    )

    matches = pipeline.retrieve_chunks(
        FakeVectorStore(), "quantum plasma", k=2, bm25_index=bm25_index
    )

    assert matches[0]["filename"] == "a.txt"
    assert {match["filename"] for match in matches} == {"a.txt", "b.txt"}


def test_out_of_topic_question_skips_llm_when_retrieval_is_empty():
    class EmptyVectorStore:
        def similarity_search(self, **kwargs):
            return []

    index = pipeline.build_bm25_index(
        [
            {
                "id": "chunk:biology:1",
                "content": "A fotossíntese ocorre nas folhas das plantas.",
                "metadata": {"filename": "biologia.txt"},
            }
        ]
    )

    def unexpected_llm_factory():
        raise AssertionError("O LLM não deve ser carregado sem candidatos.")

    decision, matches, error = pipeline.answer_from_materials(
        question="Como funciona um buraco negro?",
        vectorstore=EmptyVectorStore(),
        bm25_index=index,
        llm_factory=unexpected_llm_factory,
    )

    assert error is None
    assert matches == []
    assert decision.status == "insufficient"
    assert decision.answer == pipeline.NO_EVIDENCE_RESPONSE


def test_irrelevant_retrieved_result_is_refused_by_evidence_assessment():
    calls = []
    fake_llm = RunnableLambda(
        lambda _prompt: calls.append("called")
        or json.dumps(
            {"status": "insufficient", "answer": "", "missing": ""}
        )
    )
    class FakeVectorStore:
        def similarity_search(self, **kwargs):
            assert kwargs["filter"] == {"record_type": "chunk"}
            return [
                Document(
                    page_content="A fotossíntese usa luz para produzir glicose.",
                    metadata={
                        "source_id": "biology",
                        "chunk_number": 1,
                        "filename": "biologia.txt",
                    },
                )
            ]

    decision, matches, error = pipeline.answer_from_materials(
        question="Qual é a capital da Noruega?",
        vectorstore=FakeVectorStore(),
        bm25_index=None,
        llm_factory=lambda: fake_llm,
    )

    assert error is None
    assert len(matches) == 1
    assert calls == ["called"]
    assert decision.status == "insufficient"
    assert decision.answer == pipeline.NO_EVIDENCE_RESPONSE


def test_partially_relevant_result_is_marked_and_limited():
    fake_llm = RunnableLambda(
        lambda _prompt: json.dumps(
            {
                "status": "partial",
                "answer": "O material informa que a fotossíntese produz glicose [F1].",
                "missing": "não explica a etapa química solicitada",
            },
            ensure_ascii=False,
        )
    )
    match = {
        "filename": "biologia.txt",
        "metadata": {"source_id": "biology", "chunk_number": 1},
        "content": "A fotossíntese produz glicose.",
    }

    decision = pipeline.answer_question(
        "Como a fotossíntese produz glicose e quais são as etapas químicas?",
        [match],
        fake_llm,
    )

    assert decision.status == "partial"
    assert "[F1]" in decision.answer
    assert "não explica a etapa química solicitada" in decision.answer


def test_broad_question_gets_a_partial_answer_from_an_overview_chunk():
    prompts = []

    def respond(prompt):
        prompts.append(prompt.to_string().casefold())
        return json.dumps(
            {
                "status": "partial",
                "answer": (
                    "As placas de trânsito comunicam orientações por símbolos, cores e "
                    "formatos, e são organizadas em famílias [F1]."
                ),
                "missing": "o trecho não detalha todas as famílias",
            },
            ensure_ascii=False,
        )

    fake_llm = RunnableLambda(respond)
    match = {
        "filename": "Módulo 1 Placas, Cores e Caminhos.pdf",
        "metadata": {"source_id": "traffic", "chunk_number": 1, "page_number": 3},
        "content": (
            "As placas usam símbolos universais, cores estratégicas e formas distintas. "
            "As famílias de placas têm características próprias e uma missão específica "
            "no trânsito."
        ),
    }

    decision = pipeline.answer_question(
        "me fale sobre as placas do transito", [match], fake_llm
    )

    assert decision.status == "partial"
    assert "símbolos, cores e formatos" in decision.answer
    assert "[F1]" in decision.answer
    assert "não detalha todas as famílias" in decision.answer
    assert "não recuse apenas porque a pergunta é ampla" in prompts[0]


def test_answerable_result_keeps_model_citations():
    fake_llm = RunnableLambda(
        lambda _prompt: json.dumps(
            {
                "status": "sufficient",
                "answer": "A fotossíntese converte energia luminosa em energia química [F1].",
                "missing": "",
            },
            ensure_ascii=False,
        )
    )
    match = {
        "filename": "biologia.pdf",
        "metadata": {"source_id": "biology", "chunk_number": 1, "page_number": 4},
        "content": "A fotossíntese converte energia luminosa em energia química.",
    }

    decision = pipeline.answer_question(
        "O que a fotossíntese converte?", [match], fake_llm
    )

    assert decision.status == "sufficient"
    assert decision.answer.endswith("[F1].")


@pytest.mark.parametrize("citation", ["", "[F9]", "[F0]"])
def test_supported_answer_is_not_refused_for_missing_or_invalid_citation(citation):
    fake_llm = RunnableLambda(
        lambda _prompt: json.dumps(
            {
                "status": "sufficient",
                "answer": (
                    "A fotossíntese converte energia luminosa em energia química. "
                    f"{citation}"
                ).strip(),
                "missing": "",
            },
            ensure_ascii=False,
        )
    )
    match = {
        "filename": "biologia.pdf",
        "metadata": {"source_id": "biology", "chunk_number": 1},
        "content": "A fotossíntese converte energia luminosa em energia química.",
    }

    decision = pipeline.answer_question("O que ela converte?", [match], fake_llm)

    assert decision.status == "sufficient"
    assert "A fotossíntese converte energia luminosa em energia química." in decision.answer
    assert "[F9]" not in decision.answer
    assert "[F0]" not in decision.answer


def test_retrieval_failure_skips_llm_and_returns_error():
    class BrokenVectorStore:
        def similarity_search(self, **kwargs):
            raise RuntimeError("chroma indisponível")

    def unexpected_llm_factory():
        raise AssertionError("O LLM não deve ser chamado após falha na busca.")

    decision, matches, error = pipeline.answer_from_materials(
        question="Pergunta",
        vectorstore=BrokenVectorStore(),
        bm25_index=None,
        llm_factory=unexpected_llm_factory,
    )

    assert isinstance(error, RuntimeError)
    assert matches == []
    assert decision.answer == pipeline.RETRIEVAL_ERROR_RESPONSE


def test_local_reranker_compares_question_to_each_candidate():
    class FakeReranker:
        def predict(self, pairs, **kwargs):
            assert pairs == [
                ("pergunta", "trecho fraco"),
                ("pergunta", "trecho que responde à pergunta"),
            ]
            assert kwargs["batch_size"] == 8
            return [0.1, 0.9]

    candidates = [
        {"filename": "fraco.txt", "content": "trecho fraco", "metadata": {}},
        {
            "filename": "forte.txt",
            "content": "trecho que responde à pergunta",
            "metadata": {},
        },
    ]

    reranked = pipeline.rerank_candidates("pergunta", candidates, FakeReranker())

    assert reranked[0]["filename"] == "forte.txt"
    assert reranked[0]["reranker_score"] == 0.9
