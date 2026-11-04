"""Offline tests: no internet, no API key. Run with:  pytest -q
A hashing 'embedding' and a fake LLM stand in for the real models so the pipeline logic can be checked."""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import document_loader
from langchain_core.embeddings import Embeddings
from prompt import NOT_FOUND_MSG
from rag_pipeline import RAGPipeline


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing vectors, normalised (a stand-in for all-MiniLM-L6-v2)."""
    dim = 384

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


from langchain_core.runnables import RunnableLambda


@pytest.fixture(scope="module")
def pipe():
    pages, report = document_loader.load_folder()
    assert all(r["status"] == "accepted" for r in report)
    p = RAGPipeline(embeddings=HashEmbeddings(), llm=RunnableLambda(lambda pv: type("R", (), {"content": "ok"})()))
    p.build(pages, save=False)
    return p


def test_validate_pdf():
    assert document_loader.validate_pdf("a.txt", b"%PDF")[0] is False
    assert document_loader.validate_pdf("a.pdf", b"")[0] is False
    assert document_loader.validate_pdf("a.pdf", b"hello")[0] is False
    assert document_loader.validate_pdf("a.pdf", b"%PDF-1.4 ...")[0] is True
    big = b"%PDF" + b"0" * (config.MAX_FILE_MB * 1024 * 1024)
    assert document_loader.validate_pdf("a.pdf", big)[0] is False


def test_pages_have_metadata():
    pages, _ = document_loader.load_folder()
    assert len(pages) == 10                       # 4 + 3 + 3 pages
    assert all({"source", "page"} <= set(p.metadata) for p in pages)


def test_chunks_keep_metadata_and_size():
    pages, _ = document_loader.load_folder()
    chunks = document_loader.split_documents(pages)
    assert all("source" in c.metadata and "page" in c.metadata for c in chunks)
    assert max(len(c.page_content) for c in chunks) <= config.CHUNK_SIZE


def test_empty_and_fake_pdf_rejected():
    pages, report = document_loader.load_documents([("x.pdf", b"not a pdf"), ("y.txt", b"%PDF")])
    assert pages == [] and all(r["status"] == "rejected" for r in report)


def test_retrieval_finds_expected_pages(pipe):
    import csv
    rows = list(csv.DictReader(open(config.BASE_DIR / "tests" / "test_questions.csv")))
    answerable = [r for r in rows if r["type"] != "unavailable"]
    hits = 0
    for r in answerable:
        got = {(d["source"], d["page"]) for d in pipe.ask(r["question"])["retrieved"]}
        hits += (r["expected_source"], int(r["expected_page"])) in got
    # the crude hashing embedding is weaker than MiniLM, so only require a sensible floor
    assert hits >= 10, f"only {hits}/{len(answerable)} retrieved correctly"


def test_refuses_without_llm_when_nothing_matches(pipe):
    res = pipe.ask("zzzz qqqq xxxx")
    assert res["answer"] == NOT_FOUND_MSG and res["refused"] and res["sources"] == []


def test_source_is_returned_with_answer(pipe):
    res = pipe.ask("How many days of paid annual leave do employees get?")
    assert res["sources"] and {"source", "page"} <= set(res["sources"][0])
    assert ("Employee_Handbook.pdf", 2) in {(s["source"], s["page"]) for s in res["sources"]}


def test_empty_question_and_no_documents():
    p = RAGPipeline(embeddings=HashEmbeddings())
    assert "upload" in p.ask("hello")["answer"].lower()
