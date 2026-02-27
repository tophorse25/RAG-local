# step6_build_prompt.py
import textwrap
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks_local"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_or_create_collection(COLLECTION_NAME)

def embed_query(q: str):
    return embed_model.encode([q], convert_to_numpy=True).tolist()[0]

def retrieve(question: str, top_k: int = 8):
    q_emb = embed_query(question)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=50,  # get a lot, we'll filter
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    dists = res["distances"][0]

    # keep only “front-page / real content” and drop exhibits
    good = []
    for doc, dist in zip(docs, dists):
        low = doc.lower()
        if "exhibit 3" in low or "certification" in low:
            continue
        good.append((doc, dist))
    good.sort(key=lambda x: x[1])
    return good[:top_k]

def build_prompt(question: str):
    ctx = retrieve(question)
    ctx_text = "\n\n---\n\n".join(c[0] for c in ctx)
    prompt = f"""You are an assistant for answering questions about NVIDIA's Form 10-Q.

Use ONLY the following context. If the answer is not there, say so.

CONTEXT:
{ctx_text}

QUESTION: {question}

ANSWER:"""
    return textwrap.dedent(prompt)

if __name__ == "__main__":
    q = "What period does this report cover?"
    p = build_prompt(q)
    print(p)
