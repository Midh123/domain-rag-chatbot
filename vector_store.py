"""Module 4 and 5: embeddings, FAISS vector store, retrieval."""
import json
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS

import config


@lru_cache(maxsize=1)
def get_embeddings():
    """all-MiniLM-L6-v2 with normalised vectors (so L2 ranking == cosine ranking)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def build_index(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def save_index(vector_store, folder=None, manifest=None):
    folder = Path(folder or config.INDEX_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(folder))
    (folder / "manifest.json").write_text(json.dumps(manifest or {}, indent=2))


def index_exists(folder=None):
    return (Path(folder or config.INDEX_DIR) / "index.faiss").exists()


def load_index(embeddings, folder=None):
    folder = Path(folder or config.INDEX_DIR)
    vs = FAISS.load_local(str(folder), embeddings, allow_dangerous_deserialization=True)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return vs, manifest


def retrieve(vector_store, question, k=None):
    """Return [(Document, cosine_similarity)] best first.
    For unit-length vectors FAISS returns squared L2 distance d, and cosine = 1 - d/2."""
    results = vector_store.similarity_search_with_score(question, k=k or config.TOP_K)
    return [(doc, float(1 - dist / 2)) for doc, dist in results]
