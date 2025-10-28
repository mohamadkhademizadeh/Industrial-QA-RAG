import numpy as np
from .store import load_index

def retrieve(query_emb, index_dir, top_k=5):
    index, metas = load_index(index_dir)
    D, I = index.search(query_emb[np.newaxis, :], top_k)
    # return list of (score, metadata)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        m['score'] = float(score)
        results.append(m)
    return results
