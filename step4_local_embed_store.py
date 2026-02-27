from pathlib import Path
from typing import List, Dict
import uuid

from pypdf import PdfReader
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer

# ============== config ==============
PDF_PATH = "samplenvidia.pdf"   # change to your pdf
ENCODING_NAME = "cl100k_base"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks_local"
MAX_TOKENS_PER_CHUNK = 400
MIN_TOKENS_PER_CHUNK = 50

# pick a small, good-enough local model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============== setup ==============
enc = tiktoken.get_encoding(ENCODING_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# ---------- helpers ----------
def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def split_long_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    tokens = enc.encode(text)
    parts = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        parts.append(enc.decode(sub_tokens))
    return parts


def extract_pdf_text_basic(path: str) -> List[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        txt = txt.strip()
        if txt:
            pages.append(txt)
    return pages


def chunk_text_blocks(blocks: List[str],
                      max_tokens=400,
                      min_tokens=50) -> List[Dict]:
    chunks = []
    cur_text = ""
    cur_tokens = 0

    for block in blocks:
        block = (block or "").strip()
        if not block:
            continue

        bt = count_tokens(block)

        # if a single block is too large, split it
        if bt > max_tokens:
            # flush current
            if cur_text and cur_tokens >= min_tokens:
                chunks.append({"text": cur_text.strip(),
                               "n_tokens": cur_tokens})
                cur_text, cur_tokens = "", 0

            small_parts = split_long_text_by_tokens(block, max_tokens)
            for part in small_parts:
                pt = count_tokens(part)
                if pt >= min_tokens:
                    chunks.append({"text": part.strip(),
                                   "n_tokens": pt})
            continue

        # normal case
        if cur_tokens + bt > max_tokens:
            if cur_tokens >= min_tokens:
                chunks.append({"text": cur_text.strip(),
                               "n_tokens": cur_tokens})
            cur_text = block
            cur_tokens = bt
        else:
            cur_text = (cur_text + "\n\n" + block) if cur_text else block
            cur_tokens += bt

    # flush tail
    if cur_text and cur_tokens >= min_tokens:
        chunks.append({"text": cur_text.strip(),
                       "n_tokens": cur_tokens})

    return chunks


def main():
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(pdf_file)

    print("[1] Extracting PDF...")
    blocks = extract_pdf_text_basic(str(pdf_file))
    print(f"    got {len(blocks)} blocks")

    print("[2] Chunking...")
    chunks = chunk_text_blocks(
        blocks,
        max_tokens=MAX_TOKENS_PER_CHUNK,
        min_tokens=MIN_TOKENS_PER_CHUNK,
    )
    print(f"    got {len(chunks)} chunks")

    print("[3] Embedding locally with SentenceTransformer...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True).tolist()
    assert len(embeddings) == len(texts)

    print("[4] Storing to Chroma...")
    client_db = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client_db.get_or_create_collection(COLLECTION_NAME)

    ids = [str(uuid.uuid4()) for _ in texts]
    metadatas = [{"n_tokens": c["n_tokens"]} for c in chunks]

    col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[done] stored {len(texts)} chunks in collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
