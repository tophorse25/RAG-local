from pypdf import PdfReader
import tiktoken
from pathlib import Path

PDF_PATH = "samplenvidia.pdf"  # TODO: change to your real pdf file

def extract_pdf_text_basic(path: str) -> list[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = txt.strip()
        if txt:
            pages.append(txt)
    return pages

def main():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path.resolve()}")
        return

    pages = extract_pdf_text_basic(PDF_PATH)
    print(f"Got {len(pages)} pages with text")
    if pages:
        print("First 300 chars:\n")
        print(pages[0][:300])

if __name__ == "__main__":
    main()
