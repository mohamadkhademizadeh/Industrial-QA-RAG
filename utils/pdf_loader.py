from pypdf import PdfReader

def load_pdf_text(path: str):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(dict(page=i+1, text=text))
    return pages
