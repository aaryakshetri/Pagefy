from pypdf import PdfReader


def extract_text(file_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages).strip()
