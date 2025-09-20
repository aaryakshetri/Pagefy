from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free


def chunk(text, n=800, overlap=100):
    """Split text into overlapping chunks. Overlap preserves context at boundaries."""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + n])
        i += n - overlap
    return chunks


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


def top_k(query, texts, embs, k=4):
    q = model.encode([query], normalize_embeddings=True)[0]
    sims = embs @ q
    idx = sims.argsort()[-k:][::-1]
    return [(texts[i], float(sims[i])) for i in idx]
