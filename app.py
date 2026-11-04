"""Module 7: Streamlit interface.
Run with:  streamlit run app.py
"""
import os
import shutil

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
try:  # Streamlit Cloud keeps secrets in st.secrets
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

import config
import document_loader
from prompt import VERIFY_NOTICE
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")

# ---------------- session state ----------------
ss = st.session_state
ss.setdefault("pipeline", None)       # RAGPipeline once documents are processed
ss.setdefault("doc_report", [])       # per-file status shown in the sidebar
ss.setdefault("messages", [])         # chat history
ss.setdefault("uploader_key", 0)      # changing it resets the file uploader


@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def get_pipeline_shell():
    """One embedding model per server process; each session gets its own index."""
    from vector_store import get_embeddings
    return get_embeddings()


def process(files):
    """Validate, extract, chunk, embed and index. files = list of (name, bytes)."""
    pages, report = document_loader.load_documents(files)
    ss.doc_report = report
    if not pages:
        ss.pipeline = None
        st.sidebar.error("No usable text found in the uploaded files.")
        return
    with st.spinner("Building the search index..."):
        pipe = RAGPipeline(embeddings=get_pipeline_shell())
        n = pipe.build(pages, save=True)
    ss.pipeline = pipe
    ss.messages = []
    st.sidebar.success(f"Indexed {n} chunks from {len(pipe.manifest['files'])} document(s).")


def clear_documents():
    ss.pipeline = None
    ss.doc_report = []
    ss.messages = []
    ss.uploader_key += 1
    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Documents")
    uploads = st.file_uploader(
        f"Upload PDF files (max {config.MAX_FILE_MB} MB each)",
        type=["pdf"], accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
    )
    if uploads:
        st.caption("Selected: " + ", ".join(u.name for u in uploads))

    if st.button("Process Documents", type="primary", disabled=not uploads, use_container_width=True):
        process([(u.name, u.getvalue()) for u in uploads])

    if st.button("Use sample documents", use_container_width=True):
        process([(p.name, p.read_bytes()) for p in sorted(config.DOCUMENTS_DIR.glob("*.pdf"))])

    if ss.pipeline is None:
        from vector_store import index_exists
        if index_exists() and st.button("Load previously saved index", use_container_width=True):
            pipe = RAGPipeline(embeddings=get_pipeline_shell())
            if pipe.load_saved():
                ss.pipeline = pipe
                ss.doc_report = [{"name": n, "status": "accepted", "message": "loaded from saved index"}
                                 for n in pipe.manifest.get("files", [])]
                st.rerun()

    for r in ss.doc_report:
        (st.success if r["status"] == "accepted" else st.error)(f"{r['name']}: {r['message']}")

    if st.button("Clear / replace documents", use_container_width=True):
        clear_documents()
        st.rerun()

    st.divider()
    st.header("2. Chat")
    if st.button("Clear Chat", use_container_width=True):
        ss.messages = []
        st.rerun()
    st.caption(VERIFY_NOTICE)

# ---------------- main area ----------------
st.title("📄 Domain-Specific RAG Chatbot")
st.write("Ask questions about your uploaded PDFs. Answers come only from the documents and show the source file and page.")

if ss.pipeline is None:
    st.info("Upload one or more PDFs in the sidebar and press **Process Documents** (or try the sample documents).")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("sources"):
                st.caption("Sources: " + "; ".join(f"{s['source']} (page {s['page']})" for s in m["sources"]))
            st.caption(f"Response time: {m['seconds']} s")

question = st.chat_input("Ask a question about your documents", disabled=ss.pipeline is None)
if question:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            res = ss.pipeline.ask(question)
        st.markdown(res["answer"])
        if res["sources"]:
            st.caption("Sources: " + "; ".join(f"{s['source']} (page {s['page']})" for s in res["sources"]))
        st.caption(f"Response time: {res['seconds']} s")
        with st.expander("Show retrieved passages"):
            for r in res["retrieved"]:
                st.markdown(f"**{r['source']}, page {r['page']}** (similarity {r['score']})")
                st.write(r["text"])
    ss.messages.append({"role": "assistant", "content": res["answer"],
                        "sources": res["sources"], "seconds": res["seconds"]})
