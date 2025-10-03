import os
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from logger import get_logger

log = get_logger("index_store")


def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    name = model_name or os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=name)


def build_faiss_index(chunks: List[Document], faiss_dir: str) -> FAISS:
    log.info(f"Construction d'un nouvel index FAISS dans: {faiss_dir} (chunks={len(chunks)})")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    Path(faiss_dir).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(faiss_dir)
    return vector_store


def load_faiss_index(faiss_dir: str) -> Optional[FAISS]:
    if not Path(faiss_dir).exists():
        return None
    embeddings = get_embeddings()
    log.info(f"Chargement index FAISS depuis: {faiss_dir}")
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


def upsert_faiss_index(chunks: List[Document], faiss_dir: str) -> FAISS:
    """Append new documents to an existing index if present, otherwise create it.

    This enables incremental indexing across multiple build runs.
    """
    log.info(f"Mise à jour incrémentale de l'index (chunks={len(chunks)})")
    existing = load_faiss_index(faiss_dir)
    if existing is None:
        return build_faiss_index(chunks, faiss_dir)
    existing.add_documents(chunks)
    Path(faiss_dir).mkdir(parents=True, exist_ok=True)
    existing.save_local(faiss_dir)
    log.info("Index mis à jour et sauvegardé")
    return existing


