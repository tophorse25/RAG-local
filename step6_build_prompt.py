# step6_build_prompt.py
from __future__ import annotations

import argparse
import re
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


# ----------------------------
# CONFIG
# ----------------------------
COLLECTION_NAME = "pdf_chunks_local"
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_TOP_K = 12          # semantic results
KEYWORD_MAX = 4              # max keyword chunks to force include
FINAL_MAX_CHUNKS = 10        # total chunks in prompt (keep prompt size reasonable)


# ----------------------------
# Helpers
# ----------------------------
def _dedup_keep_order(chunks: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in chunks:
        key = c.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _keyword_hits(all_docs: List[str]) -> List[str]:
    """
    Deterministic keyword-based retrieval for cover page / header lines.
    This guarantees we pull the cover page chunk if it's in the DB.
    """
    needles = [
        "For the quarterly period ended",
        "FORM 10-Q",
        "QUARTERLY REPORT",
        "Commission file number",
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    ]

    hits: List[Tuple[int, str]] = []
    for i, d in enumerate(all_docs):
        dl = (d or "").lower()
        score = 0
        for n in needles:
            if n.lower() in dl:
                score += 1
        if score > 0:
            hits.append((score, d))

    # Sort by score desc, keep strongest cover-like chunks
    hits.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in hits[:KEYWORD_MAX]]


def retrieve_context(question: str) -> List[str]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=COLLECTION_NAME)

    # Pull all docs once (collection is small enough for a demo; 199 chunks in your run)
    all_docs = col.get(include=["documents"])["documents"]

    # 1) Keyword-first (guaranteed cover page if present)
    kw = _keyword_hits(all_docs)

    # 2) Semantic retrieval
    embedder = SentenceTransformer(EMBED_MODEL)

    semantic_query = (
        "For the quarterly period ended\n"
        "quarterly period ended\n"
        "Form 10-Q cover page\n"
        "period covered by this report\n"
        "report cover period\n"
        + question
    )
    q_emb = embedder.encode(semantic_query).tolist()

    res = col.query(query_embeddings=[q_emb], n_results=SEMANTIC_TOP_K)
    sem_docs = res.get("documents", [[]])[0]

    # Merge: keyword hits first, then semantic
    merged = _dedup_keep_order(kw + sem_docs)

    # Cap total chunks to keep prompt size sane
    return merged[:FINAL_MAX_CHUNKS]


def build_prompt(question: str) -> str:
    chunks = retrieve_context(question)
    context_text = "\n\n---\n\n".join(chunks)

    prompt = f"""
You are an assistant for answering questions about NVIDIA's Form 10-Q.

Use ONLY the following context.

Rules:
1) Do NOT infer or guess. Do NOT construct date ranges unless an explicit range appears verbatim in the context.
2) Your answer must be either:
   - a short direct quote copied from the context, OR
   - "Not found in the provided context."
3) If multiple dates appear, do NOT combine them.
4) Keep the answer to ONE line.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:
""".strip()

    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="What period does this report cover?")
    args = parser.parse_args()

    p = build_prompt(args.question)
    print("\n--- PROMPT ---\n")
    print(p)
    print("\n--- END PROMPT ---\n")


if __name__ == "__main__":
    main()