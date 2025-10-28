from utils.chunking import split_chunks

def test_split_chunks_basic():
    pages = [{'page':1,'text':'abcdef'*200}]
    ch = split_chunks(pages, chunk_size=100, overlap=10)
    assert len(ch) > 0
    assert ch[0]['page'] == 1
