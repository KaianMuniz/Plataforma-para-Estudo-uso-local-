from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

import pipeline
import storage


load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
MAX_FILE_SIZE_MB = 25
MAX_TOTAL_UPLOAD_SIZE_MB = 50
storage.initialize_chroma(CHROMA_DIR)

st.set_page_config(
    page_title="Biblioteca de Estudos (uso local)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_collection_vectorstore(
    persist_directory: str,
    collection_id: str,
    embedding_model: str,
    base_url: str,
):
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    return pipeline.open_chroma_store(persist_directory, collection_id, embeddings)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_collection_bm25_index(
    persist_directory: str,
    collection_id: str,
    revision: int,
):
    chunks = storage.list_searchable_chunks(persist_directory, collection_id)
    return pipeline.build_bm25_index(chunks)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_local_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


def make_llm(model_name: str, base_url: str):
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
        num_ctx=8192,
    )


def friendly_error(error: Exception) -> str:
    message = str(error).lower()
    if "sentence_transformers" in message or "sentence-transformers" in message:
        return "Para usar o reranker, instale as dependências opcionais com pip install -r requirements-reranker.txt."
    if any(
        text in message
        for text in ("connection refused", "failed to connect", "connection error", "localhost:11434")
    ):
        return "Não consegui conectar ao Ollama."
    if "model" in message and ("not found" in message or "pull" in message):
        return "Modelo não encontrado no Ollama. Rode ollama pull com o nome indicado na barra lateral."
    return f"Ocorreu um erro ({type(error).__name__}): {error}"


def create_collection_from_form() -> None:
    name = str(st.session_state.get("new_collection_name", "")).strip()
    embedding_model = str(
        st.session_state.get("new_collection_embedding", "")
    ).strip()
    if not name or not embedding_model:
        st.session_state.collection_creation_error = (
            "Informe o nome da coleção e o modelo de embeddings na barra lateral."
        )
        return

    try:
        collection_id = storage.create_collection(
            CHROMA_DIR, name, embedding_model
        )
    except ValueError as error:
        st.session_state.collection_creation_error = str(error)
        return

    st.session_state.active_collection_id = collection_id
    st.session_state.collection_selector_id = collection_id
    st.session_state.collection_creation_error = None
    st.session_state.chat_history = []


if "active_collection_id" not in st.session_state:
    st.session_state.active_collection_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}


with st.sidebar:
    st.markdown("## ⚙️ Modelos locais")
    chat_model = st.text_input(
        "Modelo para respostas e resumos",
        value=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b"),
        help="Use um modelo instalado no Ollama. Confira os nomes com ollama list.",
    ).strip()
    new_collection_embedding = st.text_input(
        "Embeddings para novas coleções",
        value=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        help=(
            "Só define o modelo de coleções criadas daqui em diante. Cada coleção "
            "existente continua usando o modelo salvo quando foi criada."
        ),
        key="new_collection_embedding",
    ).strip()
    ollama_url = st.text_input(
        "Endereço do Ollama",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ).strip().rstrip("/")
    reranker_model = st.text_input(
        "Reranker local (opcional)",
        value=os.getenv("RERANKER_MODEL", ""),
        placeholder="BAAI/bge-reranker-v2-m3",
        help=(
            "Reordena os candidatos comparando diretamente pergunta e trecho. "
            "Deixe vazio para usar apenas Chroma + BM25 + RRF. O modelo é baixado "
            "na primeira utilização."
        ),
    ).strip()

    st.markdown("---")
    st.markdown("**Modelos utilizados (baixar no PowerShell)**")
    st.code("ollama pull qwen3:1.7b\nollama pull qwen3-embedding:0.6b", language="powershell")


st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      .hero {padding: 1.35rem 1.55rem; border-radius: 18px; background: linear-gradient(120deg, #101827, #19354a); color: #f7fafc; margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .86;}
    </style>
    <div class="hero">
      <h1>📚 Biblioteca de Estudos (Uso Local) 📚</h1>
      <p>Organize materiais por matéria, pergunte sobre vários arquivos e confira as referencias de cada resposta.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Coleções de estudo")
create_column, select_column = st.columns([1, 1.25])
with create_column:
    with st.form("create_collection_form", clear_on_submit=True):
        new_collection_name = st.text_input(
            "Criar coleção", placeholder="Ex.: Inglês", key="new_collection_name"
        )
        st.form_submit_button(
            "Criar coleção",
            type="primary",
            use_container_width=True,
            on_click=create_collection_from_form,
        )
        if st.session_state.get("collection_creation_error"):
            st.error(st.session_state.collection_creation_error)

with select_column:
    collections = storage.list_collections(CHROMA_DIR)
    active_collection = None
    if collections:
        collection_ids = [item["id"] for item in collections]
        if st.session_state.active_collection_id not in collection_ids:
            st.session_state.active_collection_id = collection_ids[0]
        if st.session_state.get("collection_selector_id") not in collection_ids:
            st.session_state.collection_selector_id = st.session_state.active_collection_id

        selected_id = st.selectbox(
            "Coleção ativa",
            options=collection_ids,
            index=collection_ids.index(st.session_state.active_collection_id),
            format_func=lambda item_id: next(
                item["name"] for item in collections if item["id"] == item_id
            ),
            key="collection_selector_id",
        )
        if selected_id != st.session_state.active_collection_id:
            st.session_state.active_collection_id = selected_id
            st.session_state.chat_history = []
        active_collection = storage.get_collection(CHROMA_DIR, selected_id)
        if active_collection:
            st.caption(f"Embeddings desta coleção: {active_collection['embedding_model']}")
            with st.expander("Excluir coleção ativa"):
                st.warning(
                    "Isso apaga a coleção, os vetores e todos os textos armazenados nela."
                )
                confirm_collection_delete = st.checkbox(
                    f"Confirmo excluir '{active_collection['name']}' e todos os seus dados",
                    key=f"confirm_delete_collection_{active_collection['id']}",
                )
                if st.button(
                    "Excluir coleção e seus dados",
                    key=f"delete_collection_{active_collection['id']}",
                    disabled=not confirm_collection_delete,
                ):
                    deleted_collection_id = active_collection["id"]
                    storage.delete_collection(CHROMA_DIR, deleted_collection_id)
                    get_collection_vectorstore.clear()
                    get_collection_bm25_index.clear()
                    remaining_collections = [
                        item
                        for item in storage.list_collections(CHROMA_DIR)
                        if item["id"] != deleted_collection_id
                    ]
                    st.session_state.active_collection_id = (
                        remaining_collections[0]["id"] if remaining_collections else None
                    )
                    st.session_state.chat_history = []
                    st.session_state.summaries.pop(deleted_collection_id, None)
                    st.rerun()
    else:
        st.caption("Crie uma coleção para começar a montar sua biblioteca de materiais.")

if not active_collection:
    st.write("Crie uma matéria, como Biologia, História ou IA, para começar.")
    st.stop()


st.subheader(active_collection["name"])
st.caption(
    f"{active_collection['revision']} atualização(ões) · coleção persistida localmente · "
    f"modelo de embeddings: {active_collection['embedding_model']}"
)

uploaded_files = st.file_uploader(
    "Adicione arquivos de texto a esta coleção",
    type=pipeline.SUPPORTED_EXTENSIONS,
    accept_multiple_files=True,
    help=(
        "Aceita TXT, MD, SRT, VTT e PDF com texto selecionável. "
        "PDFs digitalizados sem camada de texto não são aceitos. "
        f"Limite: {MAX_FILE_SIZE_MB} MB por arquivo e "
        f"{MAX_TOTAL_UPLOAD_SIZE_MB} MB por envio."
    ),
)

upload_size_error = None
if uploaded_files:
    oversized_files = [
        file.name
        for file in uploaded_files
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024
    ]
    total_upload_size = sum(file.size for file in uploaded_files)
    if oversized_files:
        upload_size_error = (
            f"Cada arquivo pode ter até {MAX_FILE_SIZE_MB} MB. Arquivo(s) acima do "
            f"limite: {', '.join(oversized_files)}."
        )
    elif total_upload_size > MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024:
        upload_size_error = (
            f"O envio total pode ter até {MAX_TOTAL_UPLOAD_SIZE_MB} MB."
        )
    if upload_size_error:
        st.warning(upload_size_error)

process_clicked = st.button(
    "Adicionar à coleção e indexar",
    type="primary",
    use_container_width=True,
    disabled=not uploaded_files or upload_size_error is not None,
)

if process_clicked and uploaded_files and not upload_size_error:
    if not chat_model or not active_collection["embedding_model"] or not ollama_url:
        st.error("Preencha os modelos e o endereço do Ollama na barra lateral.")
    else:
        embedder = OllamaEmbeddings(
            model=active_collection["embedding_model"], base_url=ollama_url
        )
        progress = st.progress(0, text="Preparando arquivos...")
        added = 0
        skipped = 0
        failures = []

        for file_number, uploaded_file in enumerate(uploaded_files, start=1):
            digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
            existing = storage.find_source_by_hash(
                CHROMA_DIR, active_collection["id"], digest
            )
            if existing:
                skipped += 1
                progress.progress(
                    file_number / len(uploaded_files),
                    text=f"{uploaded_file.name} já está nesta coleção.",
                )
                continue

            try:
                raw_bytes = uploaded_file.getvalue()
                progress.progress(
                    (file_number - 1) / len(uploaded_files),
                    text=f"Lendo {uploaded_file.name}...",
                )
                kind, language, content, source_documents = pipeline.extract_file(
                    uploaded_file.name,
                    raw_bytes,
                )
                chunks = pipeline.split_documents(source_documents, uploaded_file.name)
                if not chunks:
                    raise ValueError("Não encontrei conteúdo textual para indexar.")
                storage.save_source(
                    CHROMA_DIR,
                    collection_id=active_collection["id"],
                    filename=uploaded_file.name,
                    sha256=digest,
                    kind=kind,
                    language=language,
                    content=content,
                    chunks=chunks,
                    embedding_function=embedder.embed_documents,
                    batch_size=16,
                )
                added += 1
            except Exception as error:
                failures.append(f"{uploaded_file.name}: {friendly_error(error)}")
            progress.progress(
                file_number / len(uploaded_files),
                text=f"Processados {file_number} de {len(uploaded_files)} arquivos.",
            )

        progress.empty()
        if added:
            get_collection_vectorstore.clear()
            get_collection_bm25_index.clear()
            st.success(f"{added} arquivo(s) adicionados à coleção.")
            st.session_state.summaries.pop(active_collection["id"], None)
        if skipped:
            st.info(f"{skipped} arquivo(s) já estavam indexados e foram ignorados.")
        for failure in failures:
            st.error(failure)


active_collection = storage.get_collection(CHROMA_DIR, active_collection["id"])
sources = storage.list_sources(CHROMA_DIR, active_collection["id"])
vectorstore = get_collection_vectorstore(
    str(CHROMA_DIR),
    active_collection["id"],
    active_collection["embedding_model"],
    ollama_url,
)
bm25_index = get_collection_bm25_index(
    str(CHROMA_DIR), active_collection["id"], active_collection["revision"]
)

if sources:
    stat_a, stat_b, stat_c = st.columns(3)
    stat_a.metric("Arquivos", len(sources))
    stat_b.metric("Trechos pesquisáveis", sum(item["chunk_count"] for item in sources))
    stat_c.metric("Banco vetorial", "Chroma local")

tab_questions, tab_summary, tab_files = st.tabs(
    ["Perguntar", "Resumo da coleção", "Arquivos"]
)

with tab_questions:
    st.markdown("### Pergunte sobre os materiais")
    if not sources:
        st.info("Adicione um ou mais arquivos acima para começar a fazer perguntas.")
    else:
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "Sua pergunta",
                placeholder="Compare o que os arquivos explicam sobre...",
            )
            ask_clicked = st.form_submit_button(
                "Perguntar à coleção", type="primary", use_container_width=True
            )

        if ask_clicked and question.strip():
            try:
                with st.spinner("Pensando..."):
                    conversation_history = list(st.session_state.chat_history)
                    decision, matches, question_error = pipeline.answer_from_materials(
                        question=question.strip(),
                        vectorstore=vectorstore,
                        bm25_index=bm25_index,
                        llm_factory=lambda: make_llm(chat_model, ollama_url),
                        chat_history=conversation_history,
                        reranker_factory=(
                            (lambda: get_local_reranker(reranker_model))
                            if reranker_model
                            else None
                        ),
                    )
                    if question_error:
                        st.error(friendly_error(question_error))
                    else:
                        st.session_state.chat_history.append(
                            {
                                "question": question.strip(),
                                "answer": decision.answer,
                                "evidence_status": decision.status,
                                "matches": matches,
                            }
                        )
            except Exception as error:
                st.error(friendly_error(error))

        for exchange in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(exchange["question"])
            with st.chat_message("assistant"):
                st.markdown(exchange["answer"])
                if (
                    exchange.get("evidence_status") == "insufficient"
                    and exchange.get("matches")
                ):
                    st.caption(
                        "A busca encontrou trechos abaixo; a recusa ocorreu na "
                        "etapa de avaliação das evidências."
                    )
                for number, match in enumerate(exchange["matches"], start=1):
                    metadata = match["metadata"]
                    location = []
                    if "page_number" in metadata:
                        location.append(f"página {metadata['page_number']}")
                    source_title = f"F{number} · {match['filename']}"
                    if location:
                        source_title += " · " + ", ".join(location)
                    with st.expander(source_title):
                        st.caption(match["content"])

with tab_summary:
    st.markdown("### Resuma todos os arquivos desta coleção")
    if not sources:
        st.info("Adicione arquivos para gerar um resumo da coleção.")
    else:
        if st.button("Gerar resumo da coleção", type="primary"):
            try:
                materials = storage.list_collection_content(
                    CHROMA_DIR, active_collection["id"]
                )
                combined_content = "\n\n".join(
                    f"===== {item['filename']} ({item['kind']}) =====\n{item['content']}"
                    for item in materials
                )
                progress = st.progress(0, text="Lendo os materiais...")
                summary = pipeline.build_summary(
                    combined_content,
                    make_llm(chat_model, ollama_url),
                    progress_callback=lambda fraction: progress.progress(
                        fraction,
                        text=f"Resumindo a coleção ({int(fraction * 100)}%)...",
                    ),
                )
                st.session_state.summaries[active_collection["id"]] = summary
                progress.empty()
            except Exception as error:
                st.error(friendly_error(error))
        saved_summary = st.session_state.summaries.get(active_collection["id"])
        if saved_summary:
            st.markdown(saved_summary)
        else:
            st.caption("O resumo considera todos os materiais atualmente indexados na coleção.")

with tab_files:
    st.markdown("### Materiais nesta coleção")
    if not sources:
        st.info("A coleção ainda não tem arquivos.")
    else:
        st.dataframe(
            [
                {
                    "Arquivo": item["filename"],
                    "Tipo": item["kind"],
                    "Idioma": item["language"],
                    "Trechos": item["chunk_count"],
                }
                for item in sources
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Remover um arquivo da coleção"):
            source_ids = [item["id"] for item in sources]
            selected_source_id = st.selectbox(
                "Arquivo",
                options=source_ids,
                format_func=lambda source_id: next(
                    item["filename"] for item in sources if item["id"] == source_id
                ),
            )
            confirm_remove = st.checkbox("Confirmo a remoção do arquivo e dos seus trechos indexados.")
            if st.button("Remover arquivo", disabled=not confirm_remove):
                storage.delete_source(
                    CHROMA_DIR, active_collection["id"], selected_source_id
                )
                get_collection_vectorstore.clear()
                get_collection_bm25_index.clear()
                st.session_state.chat_history = []
                st.session_state.summaries.pop(active_collection["id"], None)
                st.success("Arquivo removido da coleção.")
                st.rerun()

