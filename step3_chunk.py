from pypdf import PdfReader
import tiktoken
from pathlib import Path
from typing import List, Dict

PDF_PATH = "samplenvidia.pdf"   # <-- change to your real pdf
ENCODING_NAME = "cl100k_base"

# 1. token counter
enc = tiktoken.get_encoding(ENCODING_NAME)
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# 2. basic pdf → list[str]
def extract_pdf_text_basic(path: str) -> List[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = txt.strip()
        if txt:
            pages.append(txt)
    return pages

# 3. blocks → chunks (~400 tokens)
def split_long_text_by_tokens(text: str, max_tokens: int):
    """Split one long text into multiple pieces, each <= max_tokens."""
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i+max_tokens]
        sub_text = enc.decode(sub_tokens)
        chunks.append(sub_text)
    return chunks

def chunk_text_blocks(blocks, max_tokens=400, min_tokens=50):
    chunks = []
    current_text = ""
    current_tokens = 0

    for block in blocks:
        block = (block or "").strip()
        if not block:
            continue

        block_tokens = count_tokens(block)

        # if a single block is already too big, split it first
        if block_tokens > max_tokens:
            # flush current buffer first
            if current_text and current_tokens >= min_tokens:
                chunks.append({"text": current_text.strip(),
                               "n_tokens": current_tokens})
                current_text, current_tokens = "", 0

            # split this big block into smaller ones
            small_parts = split_long_text_by_tokens(block, max_tokens)
            for part in small_parts:
                part_tokens = count_tokens(part)
                if part_tokens >= min_tokens:
                    chunks.append({"text": part.strip(),
                                   "n_tokens": part_tokens})
            continue

        # normal case: try to append
        if current_tokens + block_tokens > max_tokens:
            if current_tokens >= min_tokens:
                chunks.append({"text": current_text.strip(),
                               "n_tokens": current_tokens})
            current_text = block
            current_tokens = block_tokens
        else:
            current_text = (current_text + "\n\n" + block) if current_text else block
            current_tokens += block_tokens

    # flush tail
    if current_text and current_tokens >= min_tokens:
        chunks.append({"text": current_text.strip(),
                       "n_tokens": current_tokens})

    return chunks


def main():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

    blocks = extract_pdf_text_basic(PDF_PATH)
    print(f"[INFO] extracted {len(blocks)} page-blocks")

    chunks = chunk_text_blocks(blocks, max_tokens=400)
    print(f"[INFO] final chunks: {len(chunks)}")

    # show a preview
    for i, ch in enumerate(chunks[:3]):
        print("=" * 40)
        print(f"chunk {i} | tokens={ch['n_tokens']}")
        print(ch["text"][:250], "...")
    print("=" * 40)

if __name__ == "__main__":
    main()
