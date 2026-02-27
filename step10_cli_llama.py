# step10_cli_llama.py
from step9_rag_ollama import answer

if __name__ == "__main__":
    while True:
        q = input("\nQ> ").strip()
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        print("A>", answer(q))
