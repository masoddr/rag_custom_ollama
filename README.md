# Minimal Local RAG with Ollama (Python 3.12)

This repo provides a simple, local RAG setup using FAISS + HuggingFace embeddings and an Ollama-backed LLM, with both a CLI and a FastAPI server.

## Mode d'emploi pour collègues (3 étapes)

1) Construire l'index à partir de vos documents (TXT/PDF)
```bash
python rag.py build /chemin/vers/doc1.txt /chemin/vers/doc2.pdf
```
2) Poser une question en CLI
```bash
python rag.py ask "Ma question ?"
```
3) Optionnel: lancer l'API et interroger en HTTP
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
curl -s -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"question":"Ma question ?"}'
```

## Quickstart

### 1) Create env and install deps

```bash
# optional: conda env as per request
conda create -n rag python=3.12 -y && conda activate rag
pip install -r requirements.txt
cp .env.example .env
```

### 2) Build the index

```bash
# Place your documents anywhere (TXT or PDF)
python rag.py build path/to/doc1.txt path/to/doc2.pdf
```

### 3) Ask a question via CLI

```bash
python rag.py ask "Quelle est la mission du projet ?"
```

### 4) Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# then query
curl -s -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Résume le contenu du corpus"}' | jq
```

## Docker Compose (bonus)

```bash
docker compose up --build
# In another terminal, build the index inside the container volume mapping:
docker compose exec api python rag.py build /app/README.md
# Ask through API
curl -s -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Que fournit ce projet ?"}' | jq
```

## Interface Streamlit (optionnelle)

```bash
pip install -r requirements.txt  # si pas déjà fait
streamlit run ui_streamlit.py --server.address 0.0.0.0 --server.port 7860
```

- Onglet: uploadez des fichiers TXT/PDF, cliquez sur "Construire / Mettre à jour l'index", puis posez une question.
- Les fichiers uploadés sont sauvegardés dans `.uploads/` par défaut.

## Structure

- `ingestion.py`: lit les fichiers TXT/PDF et les découpe en petits morceaux (chunks) utilisables pour la recherche.
- `index_store.py`: crée l'index FAISS à partir des chunks et le recharge depuis le disque; configure le modèle d'embeddings.
- `rag_pipeline.py`: assemble la recherche (retriever) et le modèle Ollama pour générer une réponse basée sur les passages trouvés.
- `api.py`: petit serveur FastAPI qui expose `POST /ask` pour poser une question via HTTP.
- `rag.py`: outil en ligne de commande pour construire l'index (`build`) et poser une question (`ask`).
- `Dockerfile`, `docker-compose.yml`: exécution en conteneurs (service `ollama` + service `api`).
- `.env.example`: variables d'environnement par défaut (modèle d'embed, modèle Ollama, dossier d'index).

## Config

Environment variables (can be set in `.env`):
- `EMBEDDINGS_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3`)
- `FAISS_DIR` (default: `.faiss_index`)

## Notes
- Ensure the Ollama model is pulled: `ollama run llama3` once, or rely on container to pull.
- If index is missing, the API returns 400 with a clear message. Build first via CLI.
- PDFs are parsed with `pypdf` via `PyPDFLoader`.

