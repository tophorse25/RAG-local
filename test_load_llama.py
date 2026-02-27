from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = Path("/Users/wuzihan/Desktop/job/RAG/models/tinyllama-1.1b-chat-v1.0-Q4_0.gguf")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_threads=4,      # you can bump later
    n_gpu_layers=0,   # force pure CPU for now
    verbose=True,
)

out = llm("Hello, how are you?", max_tokens=32)
print(out["choices"][0]["text"])
