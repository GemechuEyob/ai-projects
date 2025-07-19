import ollama
import logging
import json
import time
import os
from numpy.linalg import norm
import numpy as np

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
        try:
            with open(embedding_file_path, "r") as f:
                embeddings = json.load(f)
                # Validate loaded embeddings
                if embeddings and all(
                    "embedding" in e and e["embedding"] for e in embeddings
                ):
                    logger.info(
                        f"Loaded {len(embeddings)} embeddings from {embedding_file_path}"
                    )
                    return embeddings
                logger.warning("Invalid embeddings found in cache, regenerating...")
        except Exception as e:
            logger.warning(
                f"Error loading embeddings from {embedding_file_path}: {str(e)}"
                "\nRegenerating embeddings..."
            )

    embeddings = []
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            logger.warning(f"Skipping empty chunk at index {i} ({i + 1}/{len(chunks)})")
            continue

        logger.info(f"Getting embedding for chunk {i + 1}/{len(chunks)}")
        try:
            result = ollama.embeddings(model, prompt=chunk)
            if not result or "embedding" not in result or not result["embedding"]:
                logger.warning(f"Received empty embedding for chunk {i}, skipping...")
                continue
            embeddings.append(result)
            time.sleep(0.1)  # Small delay to avoid rate limiting
        except Exception as e:
            logger.error(f"Error getting embedding for chunk {i}: {str(e)}")
            continue

    if not embeddings:
        raise ValueError(
            "No valid embeddings were generated. Please check your input data and model."
        )

    logger.info(f"Saving {len(embeddings)} embeddings to {embedding_file_path}...")
    try:
        with open(embedding_file_path, "w") as f:
            json.dump([e.model_dump() for e in embeddings], f)
        logger.info(f"Successfully saved embeddings to {embedding_file_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")

    return embeddings


def search_similar_embeddings(embeddings, query_embedding, k=5):
    # Debug prints
    print(f"Number of embeddings: {len(embeddings)}")
    print(
        f"Query embedding shape: {np.array(query_embedding).shape if query_embedding is not None else 'None'}"
    )

    # Check if query_embedding is valid
    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("Query embedding is empty or None")

    # Filter out any invalid embeddings
    valid_embeddings = []
    for i, emb in enumerate(embeddings):
        if not emb or "embedding" not in emb or len(emb["embedding"]) == 0:
            print(f"Warning: Empty or invalid embedding at index {i}")
            continue
        valid_embeddings.append(emb)

    if not valid_embeddings:
        raise ValueError("No valid embeddings found to compare with")

    query_embedding = np.array(query_embedding)
    query_embedding_norm = norm(query_embedding)

    similarity_scores = []
    for i, emb in enumerate(valid_embeddings):
        try:
            emb_vec = np.array(emb["embedding"])
            emb_norm = norm(emb_vec)
            if emb_norm == 0:  # Avoid division by zero
                similarity = 0.0
            else:
                similarity = np.dot(emb_vec, query_embedding) / (
                    emb_norm * query_embedding_norm
                )
            similarity_scores.append((similarity, i))
        except Exception as e:
            print(f"Error processing embedding {i}: {str(e)}")
            continue

    # Sort by similarity score in descending order and return top k
    return sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:k]


def main():
    file_path = "rags/1_the_jungle_book.txt"
    embedding_file_path = "rags/1_the_jungle_book_embeddings.json"

    # Load or generate embeddings
    logger.info(f"Parsing file: {file_path}")
    paragraphs = parse_file(file_path)
    logger.info(f"Parsed {len(paragraphs)} paragraphs")

    logger.info("Getting embeddings...")
    start_time = time.time()
    embeddings = get_embeddings(embedding_file_path, "mistral", paragraphs)
    logger.info(f"Got {len(embeddings)} embeddings")
    logger.info(f"Took {time.time() - start_time} seconds")

    while True:
        user_prompt = input("What would you like to know about the jungle book?: ")

        # Get embedding for the user's query
        user_prompt_embedding = ollama.embeddings("mistral", user_prompt)["embedding"]

        # Find similar embeddings
        logger.info("Searching for similar embeddings...")
        start_time = time.time()
        similar_embeddings = search_similar_embeddings(
            embeddings, user_prompt_embedding
        )
        logger.info(f"Found {len(similar_embeddings)} similar embeddings")
        logger.info(f"Took {time.time() - start_time} seconds")

        # Get the actual text chunks for the similar embeddings
        context_chunks = [paragraphs[idx] for _, idx in similar_embeddings]

        # Combine the context chunks into a single string
        context = "\n\n".join(context_chunks)
        logger.info(f"Context: {context}")

        # Create a more detailed prompt
        system_prompt = """You are a helpful assistant that answers questions about The Jungle Book. 
        Use the following context to answer the question. If you don't know the answer, say you don't know.
        Context: {context}"""

        full_prompt = f"""Context: {context}
        
        Question: {user_prompt}
        
        Answer:"""

        logger.info("Generating response...")
        start_time = time.time()

        # Generate response using the context and user's question
        response = ollama.generate(
            model="mistral", prompt=full_prompt, system=system_prompt, stream=False
        )

        logger.info(f"Took {time.time() - start_time} seconds")
        logger.info("Response:")
        logger.info(response["response"])  # Access the response text


if __name__ == "__main__":
    main()

# python rags/1_the_jungle_book_rag_model.py
