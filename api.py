import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from index_store import load_faiss_index
from rag_pipeline import build_rag_chain


class AskRequest(BaseModel):
    question: str


app = FastAPI(title="Local RAG API", version="0.1.0")


_vector_store = None
_chain = None


@app.on_event("startup")
def startup_event() -> None:
    global _vector_store, _chain
    store_dir = os.getenv("FAISS_DIR", ".faiss_index")
    _vector_store = load_faiss_index(store_dir)
    if _vector_store is None:
        # Delay failure to first request with clear error
        _chain = None
    else:
        _chain = build_rag_chain(_vector_store)


@app.post("/ask")
def ask(req: AskRequest):
    global _vector_store, _chain
    if _chain is None:
        raise HTTPException(status_code=400, detail="FAISS index not found. Build it first with 'python rag.py build <files>'")
    answer = _chain.invoke({"question": req.question})
    return {"answer": answer}


