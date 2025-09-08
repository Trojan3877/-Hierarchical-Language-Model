# src/retrievers/vectorstore.py
from __future__ import annotations
import os, pathlib
from typing import List, Iterable
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMB = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = pathlib.Path("data/index.faiss")

def build_documents_from_folder(folder: str) -> List[Document]:
    """
    Scan a folder for text-like files and wrap them as LangChain Documents.
    """
    docs: List[Document] = []
    p = pathlib.Path(folder)
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".md", ".txt", ".json"}:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            meta = {"path": str(fp)}
            docs.append(Document(page_content=text, metadata=meta))
    return docs

def save_index(docs: Iterable[Document], embedding_model: str = DEFAULT_EMB, out_dir: pathlib.Path = INDEX_DIR):
    """
    Save a FAISS vector index from a list of documents.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = FAISS.from_documents(list(docs), embeddings)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))

def load_index(embedding_model: str = DEFAULT_EMB, index_dir: pathlib.Path = INDEX_DIR) -> FAISS:
    """
    Load an existing FAISS vector index.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
