import os
from typing import Optional
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS

from index_store import load_faiss_index
from logger import get_logger

log = get_logger("rag_pipeline")


def get_ollama_model(model_name: Optional[str] = None) -> ChatOllama:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = model_name or os.getenv("OLLAMA_MODEL", "llama3")
    # Control answer length and context via environment
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))  # max new tokens
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))         # context window tokens
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    return ChatOllama(
        base_url=base_url,
        model=model,
        num_predict=num_predict,
        num_ctx=num_ctx,
        temperature=temperature,
    )


def build_rag_chain(vector_store: FAISS) -> Runnable:
    k = int(os.getenv("RAG_K", "6"))
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    system_prompt = (
        "Tu es un assistant utile. Réponds de façon concise en français, "
        "en t'appuyant uniquement sur le contexte fourni. Si l'information n'est pas présente, "
        "dis que tu ne sais pas."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContexte:\n{context}"),
        ("human", "Question: {question}"),
    ])

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    model = get_ollama_model()
    parser = StrOutputParser()

    chain: Runnable = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )
    return chain


def answer_question(question: str, faiss_dir: Optional[str] = None) -> str:
    store_dir = faiss_dir or os.getenv("FAISS_DIR", ".faiss_index")
    vector_store = load_faiss_index(store_dir)
    if vector_store is None:
        raise RuntimeError(
            f"FAISS index not found at '{store_dir}'. Build it first with 'python rag.py build ...'."
        )
    chain = build_rag_chain(vector_store)
    log.info(f"Question: {question}")
    answer = chain.invoke({"question": question})
    log.info("Réponse générée")
    return answer


