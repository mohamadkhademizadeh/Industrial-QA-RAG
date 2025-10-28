import faiss, json, os, numpy as np

def save_index(index_dir, vectors, metadatas):
    os.makedirs(index_dir, exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)
    index.add(vectors)
    faiss.write_index(index, os.path.join(index_dir, 'index.faiss'))
    with open(os.path.join(index_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

def load_index(index_dir):
    idx_path = os.path.join(index_dir, 'index.faiss')
    meta_path = os.path.join(index_dir, 'meta.json')
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index not found. Run ingestion first.")
    index = faiss.read_index(idx_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    return index, metas
