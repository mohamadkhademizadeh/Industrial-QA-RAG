# Industrial-QA-RAG

A lightweight **RAG (Retrieval-Augmented Generation)** system for industrial SOPs, manuals, and PDFs.
- **Ingest PDFs** → chunk → embed (**SentenceTransformers**) → **FAISS** vector store
- **Chat app** in **Streamlit** (upload docs, ask questions, cite sources)
- Pluggable **LLM providers**: OpenAI-compatible (via `openai` lib) **or** local **Ollama**
- Optional **FastAPI** backend for programmatic `/chat`

> Designed for a clean portfolio demo: simple, documented, and easy to run locally.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Ingest: turn PDFs into a FAISS index
python scripts/ingest.py --input_dir data/pdfs --index_dir vectorstore --model all-MiniLM-L6-v2

# 2) Launch the chat UI
streamlit run app/chat_app.py
```

### LLM configuration
Set one of the following:
- **OpenAI-compatible** (OpenAI, Azure, etc.): set `OPENAI_API_KEY`. Optionally `OPENAI_BASE_URL` and `OPENAI_MODEL` in `configs/llm.yaml`.
- **Ollama** local: install Ollama, `ollama run llama3` and set `LLM_PROVIDER=ollama` (and `OLLAMA_MODEL` in config).

```bash
# Example (OpenAI)
export OPENAI_API_KEY=sk-...

# Example (Ollama local)
export LLM_PROVIDER=ollama
```

---

## Repo Layout

```
Industrial-QA-RAG/
├── app/
│   └── chat_app.py               # Streamlit chat
├── api/
│   └── server.py                 # Optional FastAPI service /chat
├── configs/
│   └── llm.yaml                  # LLM provider + parameters
├── data/
│   └── pdfs/                     # put your PDFs here
├── scripts/
│   └── ingest.py                 # build FAISS index from PDFs
├── utils/
│   ├── pdf_loader.py             # extract text from PDFs
│   ├── chunking.py               # simple text splitter
│   ├── embeddings.py             # sentence-transformers wrapper
│   ├── store.py                  # save/load FAISS + metadata
│   ├── retriever.py              # retrieve top-k chunks
│   ├── llm_providers.py          # OpenAI & Ollama adapters
│   └── rag_pipeline.py           # retrieval + prompt assembly
├── vectorstore/                  # index.faiss + meta.json (after ingest)
├── tests/
│   └── test_chunk.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Roadmap
- [ ] Add citations UI with exact page numbers and highlight snippets
- [ ] Add multi-query expansion (query reformulation)
- [ ] Add PDF OCR fallback for scanned docs (pytesseract)

