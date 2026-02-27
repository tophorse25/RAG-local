#!/usr/bin/env python3
"""
ingest_10q.py
- Recursively crawl ./docs for PDFs and .txt
- Extract text
- Chunk it
- Embed chunks
- Store embeddings in Chroma persistent DB under collection 'nvidia_10q'

Usage:
    (rag-env) python ingest_10q.py --rebuild
    (rag-env) python ingest_10q.py
"""

import argparse
import os
from pathlib import Path
import uuid
import re

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ----------------------------------
# config
# ----------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
DOC_DIR = ROOT_DIR / "docs"
PERSIST_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "nvidia_10q"

EMBED_MODEL = "all-MiniLM-L6-v2"  # fits CPU


# ----------------------------------
# helpers
# ----------------------------------

def clean_text(t: str) -> str:
    # collapse too many newlines / spaces
    t = t.replace("\x00", " ")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_all_source_files(doc_root: Path):
    """
    Recursively find .pdf / .txt (case-insensitive).
    Returns list[Path]
    """
    exts = (".pdf", ".PDF", ".txt", ".TXT")
    files = [p for p in doc_root.rglob("*") if p.suffix in exts and p.is_file()]
    files = sorted(files)
    return files


def extract_text_from_pdf(path: Path) -> str:
    out_pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            out_pages.append(txt)
    return "\n\n".join(out_pages)


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def chunk_text(text: str, chunk_size=800, chunk_overlap=200):
    """
    Sliding window splitter. Returns list[str] chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        # overlap step
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= len(words):
            break
    return chunks


def build_client(persist_dir: Path):
    """
    Create a Chroma client using the *new* API style.
    This avoids the deprecated 'duckdb+parquet' ctor you hit.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client


def recreate_collection(client, collection_name: str, rebuild: bool):
    """
    If --rebuild: delete old and create new empty.
    Else: return existing or create if missing.
    """
    existing_names = [c.name for c in client.list_collections()]
    if rebuild and collection_name in existing_names:
        client.delete_collection(name=collection_name)
        existing_names.remove(collection_name)

    if collection_name in existing_names:
        col = client.get_collection(name=collection_name)
    else:
        col = client.create_collection(name=collection_name)
    return col


def ingest_docs_into_collection(col, embedder, files):
    """
    For each file:
      - extract text
      - clean
      - chunk
      - embed each chunk
      - add to Chroma
    We'll generate UUID ids for each chunk.
    """

    total_chunks = 0

    for f in files:
        print(f"[ingest] {f}")
        if f.suffix.lower() == ".pdf":
            raw = extract_text_from_pdf(f)
        else:
            raw = extract_text_from_txt(f)

        raw = clean_text(raw)
        if not raw:
            print(f"  -> (skip empty)")
            continue

        chunks = chunk_text(raw, chunk_size=800, chunk_overlap=200)
        if not chunks:
            print(f"  -> (no chunks)")
            continue

        embeddings = embedder.encode(chunks).tolist()

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{
            "source_file": str(f.relative_to(DOC_DIR)),
            "chunk_index": idx,
        } for idx in range(len(chunks))]

        # add to Chroma
        col.add(
            ids=ids,
            metadatas=metadatas,
            documents=chunks,
            embeddings=embeddings,
        )

        print(f"  -> stored {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"[done] total chunks stored: {total_chunks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate the collection from scratch",
    )
    args = parser.parse_args()

    print(f"[scan] root: {DOC_DIR}")
    files = load_all_source_files(DOC_DIR)
    for f in files[:10]:
        print(f"[scan] found {f}")
    if not files:
        print("No text extracted. Add PDFs or .txt files anywhere under ./docs/")
        return

    print("[chroma] init client...")
    client = build_client(PERSIST_DIR)

    print(f"[chroma] prepare collection '{COLLECTION_NAME}' rebuild={args.rebuild}")
    col = recreate_collection(client, COLLECTION_NAME, args.rebuild)

    print("[embed] loading SentenceTransformer...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("[ingest] start...")
    ingest_docs_into_collection(col, embedder, files)

    print("[ok] ingestion complete.")


if __name__ == "__main__":
    main()
