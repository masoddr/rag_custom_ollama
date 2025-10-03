from typing import Iterable, List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import get_logger

log = get_logger("ingestion")


def _load_single_path(file_path: str) -> List[Document]:
    """Load a single file path into a list of Documents.

    Supports TXT and PDF. Falls back to treating unknown extensions as TXT.
    """
    lower_path = file_path.lower()
    if lower_path.endswith(".pdf"):
        log.info(f"Chargement PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()
    else:
        # Handles .txt and any other text-like files
        log.info(f"Chargement texte: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()


def load_documents(paths: Iterable[str]) -> List[Document]:
    """Load a collection of file paths into Documents.

    - paths: iterable of file paths
    """
    documents: List[Document] = []
    for path in paths:
        documents.extend(_load_single_path(path))
    log.info(f"Documents chargés: {len(documents)}")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:
    """Split documents into chunks for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info(f"Chunks générés: {len(chunks)} (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


