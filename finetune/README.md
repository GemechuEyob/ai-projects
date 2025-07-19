## Fine-Tuning

### Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [1. Product Information Extraction](#1-product-information-extraction)

### About

This is a collection of fine-tuned models that I have created for different purposes.

### Setup

```bash
# clone this repo then run the following commands
python -m venv .venv
source .venv/bin/activate
pip install -r finetune/requirements.txt
```

### 1. Product Information Extraction

This is a simple fine-tuned model that can extract product information from a given HTML string. You can opt in to run the py script locally but if you don't have a good gpu,
you can run the ipynb file in google collab with a t4 or better.

```bash
python finetune/1_product_info_extraction.py

# to run the pre-tuned model
ollama create product-info-extraction-model -f finetune/Modelfile

```
