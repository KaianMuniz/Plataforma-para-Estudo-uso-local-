"""Persistent Chroma storage for study collections, source files, and text chunks."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

import chromadb
from langchain_core.documents import Document


APP_ID = "local_study_library"
COLLECTION_PREFIX = "study_"
_SAFE_ID_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


@lru_cache(maxsize=4)
def _client(persist_directory: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_directory)


def _get_client(persist_directory: str | Path) -> chromadb.ClientAPI:
    path = Path(persist_directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return _client(str(path))


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    names = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def _get_collection(client: chromadb.ClientAPI, collection_id: str):
    if collection_id not in _collection_names(client):
        return None
    collection = client.get_collection(name=collection_id)
    if (collection.metadata or {}).get("app_id") != APP_ID:
        return None
    return collection


def _source_directory(persist_directory: str | Path, collection_id: str) -> Path:
    if (
        not collection_id
        or collection_id in {".", ".."}
        or any(character not in _SAFE_ID_CHARACTERS for character in collection_id)
    ):
        raise ValueError("Identificador de coleção inválido.")
    return Path(persist_directory).resolve().parent / "sources" / collection_id


def _source_paths(
    persist_directory: str | Path, collection_id: str, source_id: str
) -> tuple[Path, Path]:
    if (
        not source_id
        or source_id in {".", ".."}
        or any(character not in _SAFE_ID_CHARACTERS for character in source_id)
    ):
        raise ValueError("Identificador de arquivo inválido.")
    source_directory = _source_directory(persist_directory, collection_id)
    return (
        source_directory / f"{source_id}.json",
        source_directory / f"{source_id}.txt",
    )


def _write_source_record(
    persist_directory: str | Path,
    collection_id: str,
    source_id: str,
    metadata: dict,
    content: str,
) -> None:
    metadata_path, content_path = _source_paths(
        persist_directory, collection_id, source_id
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_content = content_path.with_suffix(".txt.tmp")
    try:
        temporary_content.write_text(content, encoding="utf-8")
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_content.replace(content_path)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
        temporary_content.unlink(missing_ok=True)


def _read_source_records(
    persist_directory: str | Path, collection_id: str
) -> list[dict]:
    directory = _source_directory(persist_directory, collection_id)
    if not directory.exists():
        return []
    records = []
    for metadata_path in sorted(directory.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Não consegui ler os metadados de {metadata_path.name}."
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadados inválidos em {metadata_path.name}.")
        metadata.setdefault("source_id", metadata_path.stem)
        metadata["_content_path"] = metadata_path.with_suffix(".txt")
        records.append(metadata)
    return records


def _remove_source_record(
    persist_directory: str | Path, collection_id: str, source_id: str
) -> None:
    metadata_path, content_path = _source_paths(
        persist_directory, collection_id, source_id
    )
    metadata_path.unlink(missing_ok=True)
    content_path.unlink(missing_ok=True)


def _migrate_embedded_source_records(
    persist_directory: str | Path, client: chromadb.ClientAPI
) -> None:
    """Move older whole-file records out of searchable Chroma collections."""
    for collection_id in _collection_names(client):
        collection = _get_collection(client, collection_id)
        if collection is None:
            continue
        legacy = collection.get(
            where={"record_type": "source"},
            include=["documents", "metadatas"],
        )
        migrated_ids = []
        for record_id, document, metadata in zip(
            legacy["ids"], legacy["documents"] or [], legacy["metadatas"] or []
        ):
            metadata = dict(metadata or {})
            source_id = str(
                metadata.get("source_id", record_id.removeprefix("source:"))
            )
            metadata["source_id"] = source_id
            metadata_path, content_path = _source_paths(
                persist_directory, collection_id, source_id
            )
            if not metadata_path.exists() or not content_path.exists():
                _write_source_record(
                    persist_directory,
                    collection_id,
                    source_id,
                    metadata,
                    document or "",
                )
            migrated_ids.append(record_id)
        if migrated_ids:
            collection.delete(ids=migrated_ids)


@lru_cache(maxsize=4)
def _initialize_chroma_once(persist_directory: str) -> None:
    client = _get_client(persist_directory)
    _migrate_embedded_source_records(persist_directory, client)


def initialize_chroma(persist_directory: str | Path) -> None:
    """Initialize Chroma and remove old whole-file vectors after preserving content."""
    _initialize_chroma_once(str(Path(persist_directory).resolve()))


def list_collections(persist_directory: str | Path) -> list[dict]:
    client = _get_client(persist_directory)
    collections = []
    for name in sorted(_collection_names(client)):
        if not name.startswith(COLLECTION_PREFIX):
            continue
        collection = client.get_collection(name=name)
        metadata = collection.metadata or {}
        if metadata.get("app_id") != APP_ID:
            continue
        collection_id = collection.name
        chunks = collection.get(
            where={"record_type": "chunk"}, include=["metadatas"]
        )
        collections.append(
            {
                "id": collection_id,
                "name": metadata.get("display_name", collection_id),
                "embedding_model": metadata.get("embedding_model", ""),
                "revision": int(metadata.get("revision", 0)),
                "created_at": metadata.get("created_at", ""),
                "source_count": len(
                    _read_source_records(persist_directory, collection_id)
                ),
                "chunk_count": len(chunks["ids"]),
            }
        )
    return sorted(collections, key=lambda item: item["name"].casefold())


def create_collection(
    persist_directory: str | Path, name: str, embedding_model: str
) -> str:
    display_name = name.strip()
    if not display_name:
        raise ValueError("Informe um nome para a coleção.")
    if not embedding_model.strip():
        raise ValueError("Informe um modelo de embeddings.")
    if any(
        item["name"].casefold() == display_name.casefold()
        for item in list_collections(persist_directory)
    ):
        raise ValueError("Já existe uma coleção com esse nome.")

    client = _get_client(persist_directory)
    collection_id = f"{COLLECTION_PREFIX}{uuid4().hex}"
    client.create_collection(
        name=collection_id,
        metadata={
            "app_id": APP_ID,
            "display_name": display_name,
            "embedding_model": embedding_model.strip(),
            "revision": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        configuration={"hnsw": {"space": "cosine"}},
    )
    return collection_id


def get_collection(persist_directory: str | Path, collection_id: str) -> dict | None:
    collection = _get_collection(_get_client(persist_directory), str(collection_id))
    if collection is None:
        return None
    metadata = collection.metadata or {}
    return {
        "id": collection.name,
        "name": metadata.get("display_name", collection.name),
        "embedding_model": metadata.get("embedding_model", ""),
        "revision": int(metadata.get("revision", 0)),
        "created_at": metadata.get("created_at", ""),
    }


def list_searchable_chunks(
    persist_directory: str | Path, collection_id: str
) -> list[dict]:
    """Return all chunk records for building a collection's lexical index."""
    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        return []

    result = collection.get(
        where={"record_type": "chunk"},
        include=["documents", "metadatas"],
    )
    chunks = []
    for record_id, content, metadata in zip(
        result["ids"], result["documents"] or [], result["metadatas"] or []
    ):
        chunks.append(
            {
                "id": record_id,
                "content": content or "",
                "metadata": dict(metadata or {}),
            }
        )
    return sorted(
        chunks,
        key=lambda item: (
            str(item["metadata"].get("source_id", "")),
            int(item["metadata"].get("chunk_number", 0)),
        ),
    )


def delete_collection(persist_directory: str | Path, collection_id: str) -> bool:
    """Delete one app-owned collection and its local source text/metadata."""
    client = _get_client(persist_directory)
    collection = _get_collection(client, str(collection_id))
    if collection is None:
        return False

    source_directory = _source_directory(persist_directory, collection.name)
    client.delete_collection(name=collection.name)
    if source_directory.exists():
        shutil.rmtree(source_directory)
    return True


def _increment_revision(collection) -> None:
    metadata = dict(collection.metadata or {})
    metadata["revision"] = int(metadata.get("revision", 0)) + 1
    collection.modify(metadata=metadata)


def find_source_by_hash(
    persist_directory: str | Path, collection_id: str, sha256: str
) -> dict | None:
    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        return None
    for metadata in _read_source_records(persist_directory, collection_id):
        if metadata.get("sha256") == sha256:
            return {
                "id": metadata["source_id"],
                "filename": metadata.get("filename", ""),
            }
    return None


def _scalar_metadata(metadata: dict) -> dict:
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def save_source(
    persist_directory: str | Path,
    *,
    collection_id: str,
    filename: str,
    sha256: str,
    kind: str,
    language: str,
    content: str,
    chunks: list[Document],
    embedding_function: Callable[[list[str]], list[list[float]]],
    batch_size: int = 16,
) -> str:
    if not chunks:
        raise ValueError("O arquivo não contém trechos para indexar.")
    if batch_size < 1:
        raise ValueError("O tamanho do lote de embeddings deve ser positivo.")

    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        raise ValueError("A coleção selecionada não existe no banco Chroma.")

    source_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "source_id": source_id,
        "filename": filename,
        "source_name": filename,
        "sha256": sha256,
        "kind": kind,
        "language": language,
        "created_at": created_at,
    }

    _write_source_record(
        persist_directory,
        collection_id,
        source_id,
        common_metadata,
        content,
    )
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [document.page_content for document in batch]
            vectors = embedding_function(documents)
            if len(vectors) != len(batch):
                raise ValueError(
                    "O modelo não retornou um embedding para cada trecho do lote."
                )
            ids = []
            metadatas = []
            for offset, document in enumerate(batch):
                number = start + offset + 1
                chunk_metadata = _scalar_metadata(dict(document.metadata))
                chunk_metadata.update(common_metadata)
                chunk_metadata.update(
                    {"record_type": "chunk", "chunk_number": number}
                )
                ids.append(f"chunk:{source_id}:{number}")
                metadatas.append(chunk_metadata)
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=vectors,
            )
    except Exception:
        collection.delete(where={"source_id": source_id})
        _remove_source_record(persist_directory, collection_id, source_id)
        raise

    _increment_revision(collection)
    return source_id


def list_sources(persist_directory: str | Path, collection_id: str) -> list[dict]:
    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        return []
    chunk_rows = collection.get(
        where={"record_type": "chunk"}, include=["metadatas"]
    )
    chunk_counts: dict[str, int] = {}
    for metadata in chunk_rows["metadatas"] or []:
        source_id = str((metadata or {}).get("source_id", ""))
        chunk_counts[source_id] = chunk_counts.get(source_id, 0) + 1

    sources = []
    for metadata in _read_source_records(persist_directory, collection_id):
        source_id = str(metadata["source_id"])
        sources.append(
            {
                "id": source_id,
                "filename": metadata.get("filename", "Arquivo sem nome"),
                "kind": metadata.get("kind", "Texto"),
                "language": metadata.get("language", "Texto extraído"),
                "created_at": metadata.get("created_at", ""),
                "chunk_count": chunk_counts.get(source_id, 0),
            }
        )
    return sorted(sources, key=lambda item: (item["created_at"], item["id"]))


def list_collection_content(
    persist_directory: str | Path, collection_id: str
) -> list[dict]:
    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        return []
    materials = []
    for metadata in _read_source_records(persist_directory, collection_id):
        content_path = Path(metadata.pop("_content_path"))
        materials.append(
            {
                "filename": metadata.get("filename", "Arquivo sem nome"),
                "kind": metadata.get("kind", "Texto"),
                "content": content_path.read_text(encoding="utf-8"),
                "created_at": metadata.get("created_at", ""),
            }
        )
    return sorted(materials, key=lambda item: (item["created_at"], item["filename"].casefold()))


def delete_source(
    persist_directory: str | Path, collection_id: str, source_id: str
) -> None:
    collection = _get_collection(_get_client(persist_directory), collection_id)
    if collection is None:
        return
    collection.delete(where={"source_id": str(source_id)})
    _remove_source_record(persist_directory, collection_id, str(source_id))
    _increment_revision(collection)
