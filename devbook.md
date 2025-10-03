Développe pour moi une architecture complète de RAG documentaire en Python, adaptée à un LLM hébergé localement avec **Ollama**.  
Voici les contraintes et besoins :

1. **Langages et libs** :
   - Utiliser Python 3.12+
   - Utiliser `langchain` ou `llama-index` pour orchestrer le RAG
   - Utiliser `sentence-transformers` pour générer les embeddings (par ex. `all-MiniLM-L6-v2`)
   - Utiliser un vector store simple et local (`faiss`) pour commencer
   - Connecter le tout avec mon LLM local via l’API Ollama

2. **Fonctionnalités attendues** :
   - Une fonction pour charger et parser différents formats de documents (au moins TXT et PDF)
   - Découpage en chunks avec un text splitter
   - Indexation dans FAISS avec embeddings HuggingFace
   - Création d’un pipeline RAG (retrieval + Ollama) pour répondre aux questions
   - Exemple de script minimal `rag.py` qui charge les documents, construit l’index, et permet de poser une question

3. **API locale** :
   - Fournir une API FastAPI `POST /ask` qui prend une question JSON et renvoie une réponse générée par le pipeline RAG
   - Exemple d’appel avec `curl` à documenter

4. **Dockerisation (bonus)** :
   - Fournir un `docker-compose.yml` avec :
     - un service `ollama` (pour le LLM local)
     - un service `api` (FastAPI avec ton code RAG et dépendances Python)
   - Documentation rapide pour lancer et interroger le RAG local

5. **Qualité** :
   - Code bien structuré et commenté
   - Organisation en modules (par ex. `ingestion.py`, `rag.py`, `api.py`)
   - Instructions README.md pour installer et utiliser

But final : je veux pouvoir lancer une API locale en 1 commande (`docker-compose up` ou `uvicorn api:app`) et interroger mon assistant RAG en local, avec Ollama comme moteur LLM et mes propres documents comme base de connaissances.

