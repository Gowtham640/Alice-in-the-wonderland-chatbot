📚 Semantic RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that performs context-grounded question answering over long-form literary text using semantic search and local LLM inference.

This project demonstrates how to build a production-style retrieval pipeline that minimizes hallucination while maintaining high semantic accuracy.

🚀 Overview

This system:

Ingests a Markdown text file (e.g., Alice’s Adventures in Wonderland)

Splits the document into semantically meaningful chunks

Generates dense embeddings using BAAI/bge-base-en-v1.5

Stores vectors in ChromaDB

Retrieves relevant context using Max Marginal Relevance (MMR)

Uses a local LLM via Ollama to generate grounded answers

Enforces strict prompt constraints to prevent hallucination

The model answers strictly from retrieved context and refuses when information is not present.

🧠 Architecture

User Query
↓
Embedding Generation (BGE)
↓
Vector Search (ChromaDB + MMR)
↓
Top-k Context Selection
↓
Prompt Construction
↓
Local LLM (Mistral / LLaMA3 via Ollama)
↓
Grounded Answer

🛠 Tech Stack

LangChain

ChromaDB

BAAI/bge-base-en-v1.5 embeddings

Ollama (local LLM inference)

Python

📂 Project Structure
semantic/
│
├── data/
│   └── books/
│       └── alice.md
│
├── chroma/              # Persisted vector store
├── ingest.py            # Document ingestion & embedding
├── querydata.py         # Query interface
└── README.md

⚙️ Installation
pip install -U langchain langchain-chroma langchain-huggingface langchain-ollama chromadb


Install Ollama and pull a model:

ollama pull mistral

📥 Ingestion

To embed and store the document:

python ingest.py


This:

Loads the Markdown file

Splits it into chunks

Generates embeddings

Stores them in ChromaDB

❓ Querying
python querydata.py "Why was Alice beginning to feel tired while sitting by her sister on the bank?"


Example Output:

Alice was beginning to feel tired because she had nothing to do while sitting by her sister on the bank.

🔍 Retrieval Strategy

Uses BGE embeddings with normalized vectors

Applies Max Marginal Relevance (MMR) to balance relevance and diversity

Retrieves top-k documents before generation

Limits context size to reduce noise

Prevents hallucination using strict prompt constraints

🎯 Key Features

Context-grounded answering

No external knowledge usage

Reduced hallucination risk

Optimized semantic retrieval

Fully local execution (no API dependency)

📈 Future Improvements

Hybrid retrieval (BM25 + dense embeddings)

Cross-encoder re-ranking

Query expansion

Retrieval evaluation benchmarking

Streaming response support

📌 Use Case

This project demonstrates practical implementation of:

Retrieval-Augmented Generation (RAG)

Vector search systems

LLM grounding techniques

Semantic search optimization

Suitable for:

Document QA systems

Knowledge base assistants

Research retrieval systems

Context-aware chatbots
