import os
from pathlib import Path

import streamlit as st

from ingestion import load_documents, split_documents
from index_store import build_faiss_index, load_faiss_index, upsert_faiss_index
from rag_pipeline import answer_question


st.set_page_config(page_title="Local RAG", page_icon="📚", layout="centered")

st.title("📚 Assistant RAG local")
st.caption("Upload des fichiers, construction d'index, questions via Ollama")

faiss_dir = st.sidebar.text_input("Dossier d'index FAISS", os.getenv("FAISS_DIR", ".faiss_index"))
if st.sidebar.button("📦 Recharger l'index"):
    with st.spinner("Chargement de l'index..."):
        st.session_state["store_loaded"] = load_faiss_index(faiss_dir) is not None
    if st.session_state.get("store_loaded"):
        st.toast("Index chargé", icon="✅")
    else:
        st.toast("Aucun index trouvé à ce chemin", icon="⚠️")


st.header("1) Ajouter des documents (TXT/PDF)")
uploaded_files = st.file_uploader("Déposez vos fichiers ici", type=["txt", "pdf"], accept_multiple_files=True)

if "uploaded_dir" not in st.session_state:
    st.session_state["uploaded_dir"] = str(Path(".uploads").absolute())
Path(st.session_state["uploaded_dir"]).mkdir(parents=True, exist_ok=True)

saved_paths: list[str] = []
if uploaded_files:
    for uf in uploaded_files:
        save_path = Path(st.session_state["uploaded_dir"]) / uf.name
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(str(save_path))
    st.success(f"Fichiers enregistrés: {', '.join(Path(p).name for p in saved_paths)}")

if st.button("🧱 Construire / Mettre à jour l'index"):
    if not uploaded_files and not Path(faiss_dir).exists():
        st.warning("Ajoutez d'abord des fichiers ou assurez-vous qu'un index existe.")
    else:
        if saved_paths:
            with st.status("Construction de l'index en cours...", expanded=True) as status:
                status.update(label="Lecture des documents", state="running")
                docs = load_documents(saved_paths)
                status.update(label="Découpage en morceaux", state="running")
                chunks = split_documents(docs)
                status.update(label="Génération des embeddings et écriture FAISS (incrémental)", state="running")
                upsert_faiss_index(chunks, faiss_dir)
                st.session_state["store_loaded"] = True
                status.update(label=f"✅ Index construit dans: {faiss_dir}", state="complete")
                st.balloons()
        else:
            st.info("Aucun nouveau fichier. Utilisation de l'index existant si présent.")


st.header("2) Poser une question")
question = st.text_input("Votre question")
if st.button("🤖 Répondre"):
    if not question.strip():
        st.warning("Entrez une question d'abord.")
    else:
        with st.spinner("Génération de la réponse..."):
            try:
                ans = answer_question(question, faiss_dir)
                st.markdown("### Réponse")
                st.write(ans)
            except Exception as e:
                st.error(str(e))


