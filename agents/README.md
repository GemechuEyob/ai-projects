## Agents

### Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [1. Restaurant Reviews](#1-restaurant-reviews)

### About

This is a collection of agents that I have created for different purposes.

### Setup

```bash
# clone this repo then run the following commands
python -m venv .venv
source .venv/bin/activate
pip install -r agents/requirements.txt
```

### 1. Restaurant Reviews

This is a simple agent that can answer questions about restaurant reviews.
To run the restaurant reviews agent, you need to pull the mxbai-embed-large model from ollama.

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

Then run the following command to start the agent:

```bash
python agents/1_pizza_restaurant_review_model.py
```
