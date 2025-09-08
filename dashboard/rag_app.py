# dashboard/rag_app.py
import os
import streamlit as st
from src.chains.rag import build_chain

st.set_page_config(page_title="HLM + HF + LangChain RAG", layout="wide")

model = os.getenv("GENERATION_MODEL", "google/flan-t5-base")
chain = build_chain(model)

st.title("HLM + Hugging Face + LangChain (RAG)")

q = st.text_area(
    "Ask something about your project/data:",
    height=120,
    placeholder="e.g., Summarize the HLM architecture..."
)

if st.button("Run"):
    if q.strip():
        with st.spinner("Thinking..."):
            out = chain.invoke(q.strip())
        st.subheader("Answer")
        st.write(out["answer"])
