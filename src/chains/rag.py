# src/chains/rag.py
from __future__ import annotations
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.retrievers.vectorstore import load_index
from src.providers.hf_model import HFGenerator

def hlm_fuse(context_chunks: List[str]) -> str:
    """
    Combine retrieved chunks into a hierarchical-style context.
    Replace/extend this stub with your actual HLM encoders later.
    """
    joined = "\n\n".join(context_chunks[:8])  # cap at 8 chunks
    return f"== Hierarchical Context Start ==\n{joined}\n== Hierarchical Context End =="

def build_chain(model_name: str):
    retriever = load_index().as_retriever(search_kwargs={"k": 6})
    generator = HFGenerator(model_name=model_name)

    def format_prompt(inputs):
        question = inputs["question"]
        docs: List[Document] = inputs["context"]
        ctx_chunks = [d.page_content for d in docs]
        hlm_ctx = hlm_fuse(ctx_chunks)
        prompt = (
            "Use the hierarchical context to answer the question.\n\n"
            f"{hlm_ctx}\n\nQuestion: {question}\nAnswer:"
        )
        return {"prompt": prompt}

    def call_llm(inputs):
        return {"answer": generator.generate(inputs["prompt"])}

    chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | RunnableLambda(format_prompt)
        | RunnableLambda(call_llm)
    )
    return chain
