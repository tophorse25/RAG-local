# step9_rag_fake_llm.py
from step6_build_prompt import build_prompt
import re

def extract_answer_from_context(prompt: str) -> str:
    # 1) pull CONTEXT block out of the prompt
    m = re.search(r"CONTEXT:\s*(.*)\n\nQUESTION:", prompt, flags=re.S)
    if not m:
        return "I couldn't find the context block."
    ctx = m.group(1)

    # 2) try the usual 10-Q wording
    m2 = re.search(r"(?:quarterly\s+)?period\s+ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
                   ctx, flags=re.I)
    if m2:
        return f"Quarterly period ended {m2.group(1)}."

    # 3) fallbacks
    m3 = re.search(r"ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", ctx, flags=re.I)
    if m3:
        return f"Period ended {m3.group(1)}."

    return "Not stated in retrieved context."

def main():
    q = "What period does this report cover?"
    prompt = build_prompt(q)
    answer = extract_answer_from_context(prompt)
    print(answer)
    print("\n---\n[debug prompt below]\n")
    print(prompt)

if __name__ == "__main__":
    main()
