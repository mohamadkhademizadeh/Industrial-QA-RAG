from typing import List, Dict

def split_chunks(pages: List[Dict], chunk_size=900, overlap=150) -> List[Dict]:
    chunks = []
    for p in pages:
        text = (p['text'] or '').strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end]
            chunks.append({'page': p['page'], 'text': chunk})
            start = max(end - overlap, end)
    return chunks
