# src/retrievers/vectorstore.py
from __future__ import annotations

import os
import pathlib
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_INDEX_DIR = pathlib.Path(os.getenv("FAISS_INDEX_DIR", "data/index.faiss"))
DEFAULT_BACKEND = "faiss"
SUPPORTED_BACKENDS = {"faiss", "pinecone"}


def vectorstore_backend() -> str:
    """Return the configured backend without importing optional cloud SDKs."""
    backend = os.getenv("VECTORSTORE_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported VECTORSTORE_BACKEND={backend!r}; use {supported}.")
    return backend


def _pinecone_config() -> tuple[str, str]:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    namespace = os.getenv("PINECONE_NAMESPACE", "hlm-documents").strip()

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is required when VECTORSTORE_BACKEND=pinecone."
        )
    if not index_name:
        raise RuntimeError(
            "PINECONE_INDEX_NAME is required when VECTORSTORE_BACKEND=pinecone."
        )
    if not namespace:
        raise RuntimeError("PINECONE_NAMESPACE must not be empty.")

    return index_name, namespace


def build_documents_from_folder(folder: str) -> List[Document]:
    """Scan a folder for text-like files and wrap them as Documents."""
    docs: List[Document] = []
    path = pathlib.Path(folder)

    if not path.exists():
        return docs

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {".md", ".txt", ".json"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"path": str(file_path)},
                )
            )
    return docs


def _embeddings(embedding_model: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=embedding_model)


def save_index(
    docs: Iterable[Document],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    out_dir: pathlib.Path = DEFAULT_INDEX_DIR,
    backend: str | None = None,
) -> None:
    """Persist documents to FAISS locally or Pinecone when explicitly selected."""
    selected_backend = backend or vectorstore_backend()
    embeddings = _embeddings(embedding_model)
    documents = list(docs)

    if selected_backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        index_name, namespace = _pinecone_config()
        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=index_name,
            namespace=namespace,
        )
        return

    vector_store = FAISS.from_documents(documents, embeddings)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(out_dir))


def load_index(
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    index_dir: pathlib.Path = DEFAULT_INDEX_DIR,
    backend: str | None = None,
):
    """Load the configured local FAISS or hosted Pinecone vector store."""
    selected_backend = backend or vectorstore_backend()
    embeddings = _embeddings(embedding_model)

    if selected_backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        index_name, namespace = _pinecone_config()
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace,
        )

    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
