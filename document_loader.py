"""Module 1 and 2: PDF validation, text extraction with page metadata, chunking."""
import io
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config


def validate_pdf(name: str, data: bytes):
    """Return (ok, reason). Checks extension, size and the real PDF header."""
    if not name.lower().endswith(".pdf"):
        return False, "not a .pdf file"
    if len(data) == 0:
        return False, "file is empty"
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        return False, f"larger than {config.MAX_FILE_MB} MB"
    if not data.startswith(b"%PDF"):
        return False, "file content is not a valid PDF"
    return True, "ok"


def extract_pages(name: str, data: bytes):
    """One Document per non-empty page, with source file name and page number."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF is password protected")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = " ".join((page.extract_text() or "").split())
        except Exception:
            text = ""                       # a broken page must not stop the whole file
        if text:                            # skip empty pages safely
            pages.append(Document(page_content=text, metadata={"source": name, "page": number}))
    return pages, len(reader.pages)


def load_documents(files):
    """files = list of (name, bytes). Returns (pages, report).
    report is a list of dicts with name, status ('accepted'/'rejected'), message, pages."""
    pages, report = [], []
    for name, data in files[: config.MAX_FILES]:
        ok, reason = validate_pdf(name, data)
        if not ok:
            report.append({"name": name, "status": "rejected", "message": reason, "pages": 0})
            continue
        try:
            doc_pages, total = extract_pages(name, data)
        except Exception as exc:
            report.append({"name": name, "status": "rejected", "message": f"could not read: {exc}", "pages": 0})
            continue
        if not doc_pages:
            report.append({"name": name, "status": "rejected", "pages": 0,
                           "message": "no selectable text (scanned PDF? OCR is an optional extension)"})
            continue
        pages.extend(doc_pages)
        report.append({"name": name, "status": "accepted", "pages": len(doc_pages),
                       "message": f"{len(doc_pages)} of {total} pages have text"})
    for name, _ in files[config.MAX_FILES:]:
        report.append({"name": name, "status": "rejected", "message": f"more than {config.MAX_FILES} files", "pages": 0})
    return pages, report


def load_folder(folder=None):
    """Read every PDF in a folder (used for sample documents and evaluation)."""
    folder = Path(folder or config.DOCUMENTS_DIR)
    files = [(p.name, p.read_bytes()) for p in sorted(folder.glob("*.pdf"))]
    return load_documents(files)


def split_documents(pages):
    """Module 3: chunk with overlap. Page metadata is copied to every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)
