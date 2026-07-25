import pytest

from src.retrievers.vectorstore import vectorstore_backend


def test_vectorstore_defaults_to_offline_faiss(monkeypatch):
    monkeypatch.delenv("VECTORSTORE_BACKEND", raising=False)

    assert vectorstore_backend() == "faiss"


def test_vectorstore_accepts_pinecone(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "pinecone")

    assert vectorstore_backend() == "pinecone"


def test_vectorstore_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("VECTORSTORE_BACKEND", "weaviate")

    with pytest.raises(ValueError, match="Unsupported VECTORSTORE_BACKEND"):
        vectorstore_backend()
