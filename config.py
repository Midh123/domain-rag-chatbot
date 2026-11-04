"""Central settings for the Domain-Specific RAG Chatbot."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Chunking (guide: size 700-1000, overlap 100-150)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Retrieval (guide: top 3-5 chunks)
TOP_K = 4
MIN_SIMILARITY = 0.25      # if even the best chunk is below this, refuse without calling the LLM

# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")   # Groq model; change in .env if needed

# Upload limits
MAX_FILE_MB = 20
MAX_FILES = 10
MAX_QUESTION_CHARS = 500

# Folders
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "vector_store" / "saved_index"
