# step5_query_local_smart.py
import re
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks_local"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_or_create_collection(COLLECTION_NAME)

MONTH_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", re.I)

HEADER_PATTERNS = [
    "form 10-q",
    "form 10-k",
    "for the quarterly period ended",
    "for the six months ended",
    "nvidia corporation",
    "united states",
    "securities and exchange commission",
]

def embed_query(q: str):
    return embed_model.encode([q], convert_to_numpy=True).tolist()[0]

def looks_like_header(text: str) -> int:
    t = text.lower()
    score = 0
    for p in HEADER_PATTERNS:
        if p in t:
            score += 4
    if MONTH_RE.search(text):
        score += 2
    # short front-page style text gets a bit more
    if len(text) < 2000:
        score += 1
    return score

def keyword_score(text: str, question: str) -> int:
    q_words = re.findall(r"\w+", question.lower())
    t = text.lower()
    return sum(1 for w in q_words if w in t)

def search(q: str, top_k: int = 5):
    q_emb = embed_query(q)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=50,   # get more to re-rank
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]

    candidates = []
    for doc, dist, meta in zip(docs, dists, metas):
        hs = looks_like_header(doc)
        ks = keyword_score(doc, q)
        # higher is better
        final_score = hs * 10 + ks * 2 - dist
        candidates.append({
            "text": doc,
            "distance": dist,
            "n_tokens": meta.get("n_tokens") if meta else None,
            "header_score": hs,
            "kw_score": ks,
            "final_score": final_score,
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]

if __name__ == "__main__":
    q = "What period does this report cover?"
    hits = search(q, top_k=5)
    for i, h in enumerate(hits):
        print("=" * 60)
        print(f"Hit {i} | header={h['header_score']} | kw={h['kw_score']} | dist={h['distance']:.4f}")
        print(h["text"][:400], "...")
