import argparse, os, glob, json
from utils.pdf_loader import load_pdf_text
from utils.chunking import split_chunks
from utils.embeddings import STEmbeddings
from utils.store import save_index

def ingest(input_dir, index_dir, model='all-MiniLM-L6-v2', chunk=900, overlap=150):
    files = []
    for ext in ('*.pdf',):
        files += glob.glob(os.path.join(input_dir, ext))
    if not files:
        raise SystemExit("No PDFs found in input_dir")
    all_texts = []
    meta = []
    for p in files:
        pages = load_pdf_text(p)
        chunks = split_chunks(pages, chunk_size=chunk, overlap=overlap)
        for ch in chunks:
            all_texts.append(ch['text'])
            meta.append({'source': os.path.basename(p), 'page': ch['page'], 'text': ch['text']})
    emb = STEmbeddings(model)
    vecs = emb.encode(all_texts)
    save_index(index_dir, vecs, meta)
    print(f"Ingested {len(all_texts)} chunks from {len(files)} PDFs into {index_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/pdfs")
    ap.add_argument("--index_dir", default="vectorstore")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()
    ingest(args.input_dir, args.index_dir, args.model, args.chunk, args.overlap)
