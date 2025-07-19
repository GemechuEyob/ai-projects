import ollama
import logging
import json
import time
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        paragraphs = [
            paragraph.strip().replace("\n", " ") for paragraph in f.read().split("\n\n")
        ]
        return paragraphs


def get_embeddings(embedding_file_path, model, chunks):
    if os.path.exists(embedding_file_path):
        with open(embedding_file_path, "r") as f:
            return json.load(f)

    embeddings = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Getting embedding for chunk {i}/{len(chunks)}")
        embeddings.append(ollama.embeddings(model, prompt=chunk))

    logger.info("Saving embeddings...")
    with open(embedding_file_path, "w") as f:
        json.dump([e.model_dump() for e in embeddings], f)
    logger.info(f"Saved embeddings to {embedding_file_path}")
    return embeddings


def main():
    file_path = "rags/1_the_jungle_book.txt"
    embedding_file_path = "rags/1_the_jungle_book_embeddings.json"
    logger.info(f"Parsing file: {file_path}")
    paragraphs = parse_file(file_path)
    logger.info(f"Parsed {len(paragraphs)} paragraphs")

    logger.info("Getting embeddings...")
    start_time = time.time()
    embeddings = get_embeddings(embedding_file_path, "mistral", paragraphs)
    logger.info(f"Got {len(embeddings)} embeddings")
    logger.info(f"Took {time.time() - start_time} seconds")

    while True:
        user_query = input("What would you like to know about the jungle book?: ")
        logger.info("Searching for similar embeddings...")
        start_time = time.time()
        similar_embeddings = ollama.similarity_search(embeddings, "mistral", user_query)
        logger.info(f"Found {len(similar_embeddings)} similar embeddings")
        logger.info(f"Took {time.time() - start_time} seconds")

        logger.info("Generating response...")
        start_time = time.time()
        response = ollama.generate("mistral", user_query, similar_embeddings)
        logger.info(f"Took {time.time() - start_time} seconds")
        logger.info("Response:")
        logger.info(response)


if __name__ == "__main__":
    main()

# python 1_the_jungle_book_rag_model.py
