Local RAG System over NVIDIA Form 10-Q

Overview

This project implements a fully local Retrieval-Augmented Generation (RAG) pipeline for question answering over NVIDIA’s Form 10-Q PDF document.

The entire system runs locally without any external API dependencies. It uses local embeddings, a local vector database, and a local large language model served through Ollama.

The goal of this project is to demonstrate end-to-end RAG system design, retrieval control, and hallucination mitigation in a reproducible offline environment.

Architecture

The system follows this pipeline:

Parse PDF using pypdf

Split document into semantic chunks

Generate embeddings locally using sentence-transformers

Store embeddings in a Chroma vector database

Retrieve top-k relevant chunks for a query

Construct a constrained prompt

Send prompt to a local LLM via Ollama

Return answer based strictly on retrieved context

No cloud services are used.

Components

Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Vector database:
Chroma (persistent local storage)

Language models (via Ollama):
qwen2.5:1.5b
deepseek-r1:32b

Core scripts:

step4_local_embed_store.py
Builds embeddings and stores them in Chroma.

step6_build_prompt.py
Retrieves relevant chunks and constructs a constrained prompt.

step9_rag_ollama_api.py
Sends the prompt to a local model via Ollama and generates the final answer.

Installation

Clone the repository.

Create and activate a virtual environment.

python -m venv .venv
.venv\Scripts\activate (Windows)

Install dependencies.

pip install -r requirements.txt

Install Ollama from ollama.com.

Start the Ollama server.

ollama serve

Pull a model.

ollama pull qwen2.5:1.5b

Usage

Place the PDF (e.g., samplenvidia.pdf) in the project root.

Build the vector database.

python step4_local_embed_store.py

Test retrieval and prompt construction.

python step6_build_prompt.py --question "What period does this report cover?"

Run full RAG pipeline.

python step9_rag_ollama_api.py --model qwen2.5:1.5b

Example Question

What period does this report cover?

Example Output

For the quarterly period ended July 28, 2024

Hallucination Control

The prompt enforces strict answering rules:

The model must use only the provided context.

The answer must be either a direct quote from the context or:
Not found in the provided context.

The model may not construct new date ranges.

The model may not combine multiple fragments.

This reduces hallucination and forces retrieval-grounded responses.

Project Structure

RAG-local/

step4_local_embed_store.py
step6_build_prompt.py
step9_rag_ollama_api.py
requirements.txt
README.txt

Generated artifacts such as chroma_db, data folders, model files, and PDFs are excluded via .gitignore.

Purpose

This project demonstrates:

End-to-end RAG system implementation

Local embedding and retrieval engineering

Vector database integration

Prompt constraint design

Offline LLM deployment using Ollama

Reproducible AI workflows without external APIs

This is a self-contained, local document question-answering system built from scratch.
