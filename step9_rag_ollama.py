# step9_rag_ollama.py
# Lightweight PDF RAG with optional extractive answers (verbatim quotes + [Page X])
# Requires: pip install pypdf llama-cpp-python

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from pypdf import PdfReader
except Exception as e:
    print("[Error] pypdf is required. Install with: pip install pypdf", file=sys.stderr)
    raise

try:
    from llama_cpp import Llama
except Exception as e:
    print("[Error] llama-cpp-python is required. Install with: pip install llama-cpp-python", file=sys.stderr)
    raise


# ----------------------------
# Data structures & utilities
# ----------------------------

@dataclass
class Chunk:
    page: int
    text: str
    score: float = 0.0


_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+(?=[A-Z\(])')  # naive sentence splitter for filings


def clean_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS.sub(" ", s)
    return s.strip()


def sent_tokenize(text: str) -> List[str]:
    text = clean_text(text)
    # handle pathological long pages with almost no punctuation
    if len(text) > 2000 and text.count(".") < 2:
        # slice into pseudo-sentences of ~300-500 chars
        step = 400
        return [text[i:i+step].strip() for i in range(0, len(text), step)]
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z]+", s.lower())


def uniq_stable(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------
# PDF loading & page/chunk creation
# ---------------------------------

def load_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(clean_text(txt))
    return pages


def coalesce_short_pages(pages: List[str], min_chars_per_page: int) -> List[Tuple[int, str]]:
    """
    Return list of (page_number, text). If a page is too short, merge forward until threshold.
    Page number is the first page number in the merged block (1-indexed).
    """
    out = []
    i = 0
    n = len(pages)
    while i < n:
        acc = pages[i]
        start_page = i + 1
        j = i + 1
        while len(acc) < min_chars_per_page and j < n:
            # merge the next page
            acc = (acc + " " + pages[j]).strip()
            j += 1
        out.append((start_page, acc))
        i = j
    return out


def make_chunks(pages_merged: List[Tuple[int, str]], chunk_chars: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, text in pages_merged:
        if not text:
            continue
        if len(text) <= chunk_chars:
            chunks.append(Chunk(page=page_num, text=text))
            continue
        step = max(1, chunk_chars - chunk_overlap)
        start = 0
        L = len(text)
        while start < L:
            end = min(L, start + chunk_chars)
            seg = text[start:end]
            # expand to sentence boundary if possible
            if end < L:
                # try not to cut mid-sentence
                k = seg.rfind(". ")
                if k >= int(0.6 * len(seg)):
                    seg = seg[:k+1]
                    end = start + len(seg)
            seg = seg.strip()
            if seg:
                chunks.append(Chunk(page=page_num, text=seg))
            if end >= L:
                break
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if start >= L:
                break
    return chunks


# -----------------------
# Scoring / Reranking
# -----------------------

def build_lexicon(section: Optional[str]) -> set:
    base = set("""
        revenue sales grew growth increase decrease segment segments compute networking data center gaming
        professional visualization automotive driver demand customer customers gross margin shipment backlog
        risk uncertain uncertainty export control regulation inventory supply shortage compliance license
        restriction geopolitical china legal proceeding tariffs sanction sustainability climate nac
    """.split())
    if section:
        for w in tokenize_words(section):
            base.add(w)
    return base


def score_chunk(chunk: Chunk, query: str, section: Optional[str], lex: set) -> float:
    s = chunk.text.lower()
    q = query.lower() if query else ""
    tokens = tokenize_words(s)
    # lexical hits
    hits = sum(1 for t in tokens if t in lex)
    # query word overlap
    q_hits = sum(1 for w in tokenize_words(q) if w in s)
    # section emphasis
    sec_bonus = 0
    if section:
        for w in tokenize_words(section):
            if w in s:
                sec_bonus += 3
    # penalize if chunk looks like tables/numbers only
    digit_ratio = sum(ch.isdigit() for ch in chunk.text) / max(1, len(chunk.text))
    penalty = 1.0 if digit_ratio > 0.25 else 0.0
    return hits + q_hits + sec_bonus - penalty


def rerank_chunks(all_chunks: List[Chunk], query: str, section: Optional[str], k: int) -> List[Chunk]:
    lex = build_lexicon(section)
    for c in all_chunks:
        c.score = score_chunk(c, query, section, lex)
    # Sort by score desc, then by shorter text (denser) as tie-breaker
    all_chunks.sort(key=lambda x: (x.score, -len(x.text)), reverse=True)
    chosen = all_chunks[:max(1, k)]
    return chosen


# -----------------------
# Extractive answering
# -----------------------

def keep_relevant(sent: str, query: str, section_hint: str) -> bool:
    q = query.lower()
    s = sent.lower()
    lex = set([
        "revenue","sales","grew","increase","decrease","segment","data center","gaming",
        "professional visualization","automotive","driver","demand","customers","gross",
        "risk","uncertain","export","control","regulation","inventory","supply","shortage",
        "compliance","license","restriction","geopolitical","china","legal","proceeding",
        "nac","tariff","sanction","climate","sustainability"
    ])
    tokens = tokenize_words(s)
    hits = sum(1 for t in tokens if t in lex)
    q_hits = sum(1 for w in tokenize_words(q) if w in s)
    sec_ok = (section_hint and section_hint.lower() in s)
    return (hits + q_hits + (1 if sec_ok else 0)) >= 2


def extract_snippets(chosen_chunks: List[Chunk], query: str, section_hint: str, max_quotes: int) -> List[Tuple[int, str]]:
    seen = set()
    results: List[Tuple[int, str]] = []
    for c in chosen_chunks:
        for sent in sent_tokenize(c.text):
            if keep_relevant(sent, query, section_hint):
                key = (c.page, sent.lower())
                if key in seen:
                    continue
                seen.add(key)
                if 30 <= len(sent) <= 600:
                    results.append((c.page, sent))
                    if len(results) >= max_quotes:
                        return results
    return results


def build_extractive_answer(snippets: List[Tuple[int, str]]) -> str:
    if not snippets:
        return "No directly relevant sentences were found in the retrieved pages."
    lines = []
    for pg, sent in snippets:
        # ensure single-line
        s1 = clean_text(sent)
        lines.append(f"[Page {pg}] {s1}")
    return "\n".join(lines)


# -----------------------
# Generative answering
# -----------------------

def build_context_block(chosen_chunks: List[Chunk], limit_chars: int = 8000) -> str:
    """Concatenate chosen chunks with page tags, truncated to limit_chars."""
    parts = []
    total = 0
    for c in chosen_chunks:
        tag = f"[Page {c.page}]"
        piece = f"{tag} {c.text}\n"
        if total + len(piece) > limit_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n".join(parts)


def build_prompt(question: str, section: Optional[str], context: str) -> str:
    # Strict instructions: only use provided context; include [Page X]
    sec_line = f"Section hint: {section}" if section else "Section hint: (none)"
    return (
        "You are a careful financial analyst. Answer ONLY using the quotes provided in CONTEXT.\n"
        "Rules:\n"
        " - Do NOT invent facts or numbers.\n"
        " - Every factual claim must be attributable to a sentence in CONTEXT.\n"
        " - Include page tags like [Page 26] inline where you use a fact.\n"
        " - If the answer is not in CONTEXT, say you cannot find it.\n\n"
        f"{sec_line}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "Now answer succinctly with bullet points and page tags.\n"
    )


def call_llm(model_path: str, prompt: str, n_ctx: int, temperature: float, max_tokens: int, repeat_penalty: float) -> str:
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        verbose=False
    )
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        echo=False
    )
    txt = out["choices"][0]["text"]
    return txt.strip()


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Simple PDF RAG with optional extractive answers.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--section", default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--repeat_penalty", type=float, default=1.1)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--chunk_chars", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--min_chars_per_page", type=int, default=900)
    # New extractive options
    parser.add_argument("--extractive", action="store_true",
                        help="Build answer only from verbatim quotes with [Page X] tags.")
    parser.add_argument("--max_quotes", type=int, default=8,
                        help="Max number of quotes in extractive mode.")
    args = parser.parse_args()

    print(f"[RAG] Loading PDF: {args.pdf}")
    pages_raw = load_pdf_pages(args.pdf)
    pages_merged = coalesce_short_pages(pages_raw, args.min_chars_per_page)

    # Build chunks
    all_chunks = make_chunks(pages_merged, args.chunk_chars, args.chunk_overlap)
    print(f"[RAG] {len(all_chunks)} chunks")

    # Rerank
    chosen_chunks = rerank_chunks(all_chunks, args.question, args.section, args.k)

    # Debug print top pages without backslashes in f-string expressions
    print("[RAG] Top (page, score) after strict section rerank:")
    for c in chosen_chunks[:min(12, len(chosen_chunks))]:
        preview = c.text[:120] + ("..." if len(c.text) > 120 else "")
        preview = preview.replace("\n", " ")
        print("  - p.{:d} | {:.3f} | {}".format(c.page, c.score, preview))

    # Extractive branch
    if args.extractive:
        print("[RAG] Building extractive answer…")
        snippets = extract_snippets(chosen_chunks, args.question, args.section or "", args.max_quotes)
        answer = build_extractive_answer(snippets)
        print("\n=== Answer (Extractive) ===")
        print(answer)
        return

    # Generative branch (guardrails)
    context = build_context_block(chosen_chunks, limit_chars=min(args.n_ctx * 2, 8000))
    prompt = build_prompt(args.question, args.section, context)
    print("Loading model:", args.model_path)
    answer = call_llm(
        model_path=args.model_path,
        prompt=prompt,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        repeat_penalty=args.repeat_penalty
    )
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
