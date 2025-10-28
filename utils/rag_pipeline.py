import yaml, os
from .embeddings import STEmbeddings
from .retriever import retrieve

def format_context(chunks, max_chars=6000):
    collected = []
    total = 0
    for c in chunks:
        snip = f"[p.{c['page']}] {c['text'].strip()}"
        if total + len(snip) > max_chars:
            break
        collected.append(snip)
        total += len(snip)
    return "\n\n".join(collected)

SYSTEM_PROMPT = "You are a helpful industrial QA assistant. Answer using the provided CONTEXT. If unknown, say you don't know. Cite page numbers like [p.X]."

def build_messages(question, chunks, max_chars=6000):
    ctx = format_context(chunks, max_chars)
    user = f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer concisely and cite sources."
    return [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": user}
    ]

def retrieve_chunks(query, index_dir, embed_model='all-MiniLM-L6-v2', top_k=5):
    emb = STEmbeddings(embed_model)
    q = emb.encode([query])[0]
    hits = retrieve(q, index_dir, top_k=top_k)
    return hits, emb
