# src/ingest.py
import argparse

from src.retrievers.vectorstore import (
    build_documents_from_folder,
    save_index,
    vectorstore_backend,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a local FAISS or hosted Pinecone index from documents."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["data", "docs"],
        help="Folders to scan for documents",
    )
    args = parser.parse_args()

    all_docs = []
    for folder in args.folders:
        all_docs.extend(build_documents_from_folder(folder))

    if not all_docs:
        print("No documents found to index.")
        return

    backend = vectorstore_backend()
    save_index(all_docs)
    if backend == "pinecone":
        print(
            f"Indexed {len(all_docs)} documents into Pinecone "
            "using PINECONE_INDEX_NAME/PINECONE_NAMESPACE."
        )
    else:
        print(f"Indexed {len(all_docs)} documents into data/index.faiss.")


if __name__ == "__main__":
    main()
