"""Module 6: the full RAG pipeline (retrieve -> grounded answer -> sources)."""
import os
import time

import config
import document_loader
import vector_store as vs_module
from prompt import NOT_FOUND_MSG, format_context, get_prompt


def make_llm():
    """Create the Groq chat model. The key comes from the environment (.env, Streamlit secrets, Colab)."""
    from langchain_groq import ChatGroq
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or to the platform's secrets.")
    kwargs = dict(model=config.LLM_MODEL, temperature=0, max_tokens=600, timeout=60, max_retries=2)
    if "gpt-oss" in config.LLM_MODEL:
        kwargs["reasoning_effort"] = "low"      # reasoning models answer much faster on low effort
    return ChatGroq(**kwargs)


class RAGPipeline:
    def __init__(self, embeddings=None, llm=None):
        self._embeddings = embeddings
        self._llm = llm
        self.vector_store = None
        self.manifest = {}

    # ---- lazy components (so tests can inject fakes) ----
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = vs_module.get_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm()
        return self._llm

    @property
    def ready(self):
        return self.vector_store is not None

    # ---- indexing ----
    def build(self, pages, save=True):
        """pages = list of page Documents. Returns number of chunks."""
        chunks = document_loader.split_documents(pages)
        self.vector_store = vs_module.build_index(chunks, self.embeddings)
        self.manifest = {
            "files": sorted({p.metadata["source"] for p in pages}),
            "pages": len(pages),
            "chunks": len(chunks),
        }
        if save:
            vs_module.save_index(self.vector_store, manifest=self.manifest)
        return len(chunks)

    def load_saved(self):
        if not vs_module.index_exists():
            return False
        self.vector_store, self.manifest = vs_module.load_index(self.embeddings)
        return True

    # ---- question answering ----
    def retrieve(self, question):
        return vs_module.retrieve(self.vector_store, question, config.TOP_K)

    def ask(self, question):
        """Returns dict: answer, sources, retrieved, seconds, refused."""
        start = time.time()
        question = (question or "").strip()[: config.MAX_QUESTION_CHARS]
        if not question:
            return self._result("Please type a question.", [], [], start, refused=False)
        if not self.ready:
            return self._result("Please upload and process documents first.", [], [], start, refused=False)

        hits = self.retrieve(question)
        retrieved = [{"source": d.metadata["source"], "page": d.metadata["page"],
                      "score": round(s, 3), "text": d.page_content} for d, s in hits]

        # Refuse BEFORE calling the LLM when nothing relevant was found
        if not hits or hits[0][1] < config.MIN_SIMILARITY:
            return self._result(NOT_FOUND_MSG, [], retrieved, start, refused=True)

        try:
            answer = (get_prompt() | self.llm).invoke(
                {"context": format_context(hits), "question": question}).content.strip()
        except Exception as exc:
            return self._result(f"The language model request failed: {exc}", [], retrieved, start, refused=False)

        refused = NOT_FOUND_MSG.lower() in answer.lower()
        sources = [] if refused else self._pick_sources(hits)
        return self._result(answer, sources, retrieved, start, refused=refused)

    @staticmethod
    def _pick_sources(hits):
        """Show only the pages that are close to the best match (not every retrieved chunk)."""
        best = hits[0][1]
        cutoff = max(config.MIN_SIMILARITY, best - 0.15)
        seen, sources = set(), []
        for doc, score in hits:
            key = (doc.metadata["source"], doc.metadata["page"])
            if score >= cutoff and key not in seen:
                seen.add(key)
                sources.append({"source": key[0], "page": key[1]})
        return sources

    @staticmethod
    def _result(answer, sources, retrieved, start, refused):
        return {"answer": answer, "sources": sources, "retrieved": retrieved,
                "seconds": round(time.time() - start, 2), "refused": refused}
