## RAGs

### Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [1. The Jungle Book](#1-the-jungle-book)

### About

This is a collection of RAGs (Retrieval Augmented Generation) models that I have created for different purposes.

### Setup

```bash
# clone this repo then run the following commands
python -m venv .venv
source .venv/bin/activate
pip install -r rags/requirements.txt
```

### 1. The Jungle Book

This is a simple RAG model that uses Ollama to generate responses based on the embeddings of the text. When running the first time, It might take while to create the embedding.

To run the model, use the following command:

```bash
python rags/1_the_jungle_book_rag_model.py
```
