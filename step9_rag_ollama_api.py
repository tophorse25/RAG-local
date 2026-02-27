# step9_rag_ollama_api.py
# RAG + Ollama (API) with strict quote-only answering (no "Not found" BS when answer exists)

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Optional

import requests

# IMPORTANT: this imports your step6 builder
# Make sure step6_build_prompt.py has: build_prompt(question: str) -> str
from step6_build_prompt import build_prompt


def extract_context_block(prompt: str) -> str:
    """
    Extract the CONTEXT: ... block from the prompt.
    """
    m = re.search(r"\nCONTEXT:\n(.*)\n\nQUESTION:", prompt, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1)


def find_best_quote_line(prompt: str, question: str) -> Optional[str]:
    """
    Deterministic fallback: if the exact answer is present as a line in the prompt context,
    return that line. (We use this to avoid LLM refusal/incorrect 'Not found'.)

    For the NVIDIA 10-Q period question, we want the cover line:
    'For the quarterly period ended ...'
    """
    context = extract_context_block(prompt)
    if not context:
        return None

    # Rule: return ONE line quote from context.
    # Heuristic for this question:
    if "period does this report cover" in question.lower():
        # Try the most reliable cover-page marker
        for line in context.splitlines():
            if "for the quarterly period ended" in line.lower():
                return line.strip()

    return None


def call_ollama_chat(model: str, prompt: str, base_url: str = "http://localhost:11434") -> str:
    """
    Call Ollama /api/chat.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        # keep it deterministic-ish
        "options": {
            "temperature": 0.0,
        },
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "") or ""


def enforce_one_line_quote(answer: str, prompt: str) -> str:
    """
    Enforce:
      - ONE line
      - must be a direct quote from CONTEXT (substring match)
      - else -> Not found

    We also "salvage" by extracting a line from the model output that appears in context.
    """
    context = extract_context_block(prompt)

    # Normalize
    ans = (answer or "").strip()
    if not ans:
        return "Not found in the provided context."

    # Take first non-empty line only
    first_line = ""
    for line in ans.splitlines():
        line = line.strip()
        if line:
            first_line = line
            break

    if not first_line:
        return "Not found in the provided context."

    # Remove wrapping quotes if model added them
    first_line = first_line.strip('"\'')

    # Direct containment check (quote must exist verbatim in context)
    if first_line in context:
        return first_line

    # Salvage: find any line from model output that appears in context
    for line in ans.splitlines():
        line = line.strip().strip('"\'')

        if not line:
            continue
        if line in context:
            return line

    return "Not found in the provided context."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--question", type=str, default="What period does this report cover?")
    parser.add_argument("--show_prompt", action="store_true")
    parser.add_argument("--dry_run_quote", action="store_true", help="No LLM; print best quote from context if exists.")
    parser.add_argument("--base_url", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    prompt = build_prompt(args.question)

    if args.show_prompt:
        print("\n--- PROMPT ---\n")
        print(prompt)
        print("\n--- END PROMPT ---\n")

    # 1) Deterministic no-LLM path (debug + also useful in CI)
    if args.dry_run_quote:
        q = find_best_quote_line(prompt, args.question)
        if q:
            print(q)
            return 0
        print("Not found in the provided context.")
        return 0

    # 2) Deterministic fallback BEFORE calling LLM:
    # If the exact quote is already in context, just return it.
    direct = find_best_quote_line(prompt, args.question)
    if direct:
        print(direct)
        return 0

    # 3) Otherwise call Ollama
    try:
        raw = call_ollama_chat(args.model, prompt, base_url=args.base_url)
    except requests.RequestException as e:
        print(f"[Error] Ollama API call failed: {e}", file=sys.stderr)
        return 2

    final = enforce_one_line_quote(raw, prompt)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())