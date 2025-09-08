# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from src.chains.rag import build_chain

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")

app = FastAPI(title="HLM + Hugging Face + LangChain API")

class QueryIn(BaseModel):
    question: str

class QueryOut(BaseModel):
    answer: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/rag/query", response_model=QueryOut)
def rag_query(q: QueryIn):
    chain = build_chain(GENERATION_MODEL)
    out = chain.invoke(q.question)
    return QueryOut(answer=out["answer"])
