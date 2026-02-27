# step7_answer_local.py
import re
from step6_build_prompt import build_prompt

# simple date/period patterns seen in 10-Qs
PERIOD_PATTERNS = [
    r"for the quarterly period ended ([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"for the quarter ended ([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"for the six months ended ([A-Za-z]+\s+\d{1,2},\s+\d{4})",
    r"for the fiscal year ended ([A-Za-z]+\s+\d{1,2},\s+\d{4})",
]

def extract_context(prompt: str) -> str:
    # prompt format from step6: "CONTEXT:\n...\n\nQUESTION:"
    m = re.search(r"CONTEXT:\n(.*)\n\nQUESTION:", prompt, flags=re.S)
    return m.group(1).strip() if m else ""

def find_period(ctx: str) -> str | None:
    lower = ctx.lower()
    for pat in PERIOD_PATTERNS:
        m = re.search(pat, lower, flags=re.I)
        if m:
            # pull original substring from ctx using span on lower
            start, end = m.span(1)
            return ctx[start:end]
    return None

def answer(question: str) -> str:
    prompt = build_prompt(question)
    ctx = extract_context(prompt)
    period = find_period(ctx)
    if period:
        return f"The report covers the {period}."
    # fallback: just return the first 2–3 lines of context
    first_lines = "\n".join(ctx.splitlines()[:3])
    return f"I couldn't find an explicit 'period ended' line. Closest context:\n{first_lines}"

if __name__ == "__main__":
    q = "What period does this report cover?"
    ans = answer(q)
    print("Q:", q)
    print("A:", ans)
