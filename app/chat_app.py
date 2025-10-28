import streamlit as st
import os, yaml, json
from utils.rag_pipeline import retrieve_chunks, build_messages
from utils.llm_providers import chat as llm_chat

st.set_page_config(page_title="Industrial QA — RAG", layout="wide")
st.title("📚 Industrial QA Assistant — RAG")

with open('configs/llm.yaml','r') as f:
    CFG = yaml.safe_load(f)

index_dir = st.sidebar.text_input("Vector index dir", "vectorstore")
top_k = st.sidebar.number_input("Top-K", 1, 10, CFG['retrieval'].get('top_k', 5))
max_ctx = st.sidebar.number_input("Max context chars", 1000, 20000, CFG['retrieval'].get('max_context_chars', 6000))

provider = st.sidebar.selectbox("LLM Provider", ["openai","ollama"], index=0 if CFG['provider']=="openai" else 1)
question = st.text_input("Ask a question about your PDFs")

uploaded = st.file_uploader("Upload PDFs (optional, updates the index)", type=['pdf'], accept_multiple_files=True)
if uploaded:
    import tempfile, shutil, subprocess
    tmpdir = tempfile.mkdtemp()
    os.makedirs("data/pdfs", exist_ok=True)
    for f in uploaded:
        p = os.path.join("data/pdfs", f.name)
        with open(p, "wb") as out:
            out.write(f.read())
    st.success("Saved PDFs to data/pdfs. Now run ingestion:")
    st.code("python scripts/ingest.py --input_dir data/pdfs --index_dir vectorstore")

if st.button("Search & Answer") and question:
    try:
        hits, _ = retrieve_chunks(question, index_dir, top_k=int(top_k))
        messages = build_messages(question, hits, max_chars=int(max_ctx))
        CFG['provider'] = provider  # override
        answer = llm_chat(CFG['provider'], messages, CFG)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Sources")
        for h in hits:
            st.write(f"- {h['source']} [p.{h['page']}] (score: {h['score']:.3f})")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Ingest your PDFs first, then type a question and click **Search & Answer**.")
