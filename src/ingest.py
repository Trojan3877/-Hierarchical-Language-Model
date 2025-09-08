# src/ingest.py
import argparse
from src.retrievers.vectorstore import build_documents_from_folder, save_index

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from local documents.")
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
        print("⚠️ No documents found to index.")
        return

    save_index(all_docs)
    print(f"✅ Indexed {len(all_docs)} docs into data/index.faiss")

if __name__ == "__main__":
    main()
