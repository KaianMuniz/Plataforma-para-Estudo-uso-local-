import pytest
from langchain_core.documents import Document

import storage


def test_create_collection_duplicate_hash_and_delete_source(tmp_path):
    chroma_dir = tmp_path / "chroma"
    collection_id = storage.create_collection(
        chroma_dir, "Biologia", "test-embeddings"
    )
    batch_sizes = []

    def fake_embeddings(texts):
        batch_sizes.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]

    chunks = [
        Document(page_content=f"trecho {number}", metadata={"page_number": 1})
        for number in range(5)
    ]
    source_id = storage.save_source(
        chroma_dir,
        collection_id=collection_id,
        filename="aula.txt",
        sha256="same-content-hash",
        kind="Texto",
        language="Texto extraído",
        content="texto integral",
        chunks=chunks,
        embedding_function=fake_embeddings,
        batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    assert storage.find_source_by_hash(
        chroma_dir, collection_id, "same-content-hash"
    ) == {"id": source_id, "filename": "aula.txt"}
    assert storage.list_sources(chroma_dir, collection_id)[0]["chunk_count"] == 5
    searchable_chunks = storage.list_searchable_chunks(chroma_dir, collection_id)
    assert len(searchable_chunks) == 5
    assert searchable_chunks[0]["id"].startswith(f"chunk:{source_id}:")

    storage.delete_source(chroma_dir, collection_id, source_id)

    assert storage.find_source_by_hash(
        chroma_dir, collection_id, "same-content-hash"
    ) is None
    assert storage.list_sources(chroma_dir, collection_id) == []


def test_collection_names_are_unique_case_insensitively(tmp_path):
    chroma_dir = tmp_path / "chroma"
    storage.create_collection(chroma_dir, "História", "test-embeddings")

    with pytest.raises(ValueError, match="Já existe uma coleção"):
        storage.create_collection(chroma_dir, "HISTÓRIA", "test-embeddings")


def test_delete_collection_removes_vectors_and_source_files(tmp_path):
    chroma_dir = tmp_path / "chroma"
    collection_id = storage.create_collection(
        chroma_dir, "Química", "test-embeddings"
    )
    storage.save_source(
        chroma_dir,
        collection_id=collection_id,
        filename="aula.txt",
        sha256="collection-delete-hash",
        kind="Texto",
        language="Texto extraído",
        content="texto integral",
        chunks=[Document(page_content="trecho")],
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    source_directory = tmp_path / "sources" / collection_id
    assert source_directory.exists()

    assert storage.delete_collection(chroma_dir, collection_id) is True

    assert storage.get_collection(chroma_dir, collection_id) is None
    assert not source_directory.exists()
    assert storage.list_collections(chroma_dir) == []


def test_failed_embedding_batch_removes_partial_source(tmp_path):
    chroma_dir = tmp_path / "chroma"
    collection_id = storage.create_collection(
        chroma_dir, "Física", "test-embeddings"
    )

    def invalid_embeddings(_texts):
        return []

    with pytest.raises(ValueError, match="um embedding para cada trecho"):
        storage.save_source(
            chroma_dir,
            collection_id=collection_id,
            filename="falha.txt",
            sha256="failed-hash",
            kind="Texto",
            language="Texto extraído",
            content="texto integral",
            chunks=[Document(page_content="trecho")],
            embedding_function=invalid_embeddings,
        )

    assert storage.find_source_by_hash(chroma_dir, collection_id, "failed-hash") is None
    assert storage.list_sources(chroma_dir, collection_id) == []
