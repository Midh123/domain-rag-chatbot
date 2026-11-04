# Project Report: Domain-Specific RAG Chatbot for PDF Question Answering

\*\*Student name:\*\* Midhun KG  \*\*Batch:\*\* June Batch A (Unlox Major Project)  \*\*Date:\*\* 4 November 2026

\*\*GitHub repository:\*\* https://github.com/Midh123/domain-rag-chatbot  \*\*Deployed app link:\*\* https://domain-rag-chatbot-ecbuojjbrsdbzmnj65zqw9.streamlit.app/

## 1\. Objective

Build a reliable question-answering system that answers only from user-uploaded PDF documents, shows the source file and page for every answer, and refuses when the answer is not in the documents.

## 2\. Problem statement

Large documents are hard to search manually: a user may have to read many pages to find one answer. This project lets users upload documents and ask questions in natural language.

## 3\. Domain and documents

Company policies (sample domain): `Employee\\\_Handbook.pdf` (4 pages), `IT\\\_Security\\\_Policy.pdf` (3 pages), `Travel\\\_and\\\_Expense\\\_Policy.pdf` (3 pages). *(Replace with your own domain and 2-5 documents if required.)*

## 4\. Method

|Stage|What was done|Tool|
|-|-|-|
|Upload and validation|Check extension, size (20 MB) and `%PDF` header|Python|
|Text extraction|One record per page with file name and page number; empty pages skipped|pypdf|
|Chunking|900 characters, 120 overlap, page metadata copied to each chunk|LangChain RecursiveCharacterTextSplitter|
|Embeddings|384-dimensional normalised vectors|Sentence Transformers `all-MiniLM-L6-v2`|
|Vector store|Index saved to `vector\\\_store/saved\\\_index`|FAISS|
|Retrieval|Top 4 chunks; cosine similarity = 1 - d/2|FAISS|
|Generation|Strict prompt, fixed fallback message, ignore instructions inside documents|LangChain + Groq `openai/gpt-oss-20b`|
|Interface|Sidebar uploader, Process Documents, chat history, sources, Clear Chat|Streamlit|

Architecture diagram: `docs/architecture.png`.

## 5\. Testing and evaluation

Run `python evaluate.py` and paste the summary here.

|Measure|Result|
|-|-|
|Retrieval accuracy (right file and page in top 4)|15 / 15|
|Answer correctness (answerable + paraphrased)|14 / 15|
|Refusal quality (unavailable questions refused)|4 / 4|
|Average response time|2.00 seconds|

Open `tests/evaluation\\\_results.csv` and also check by hand: **groundedness** (no facts outside the context) and **source quality** (useful file and page). Note any failures and the reason for each here:

* \---

## 6\. Responsible AI and security

API key stored outside the code; only PDF files with a size limit; answers restricted to retrieved context; fixed refusal message; refusal before the LLM when similarity is low; document text treated as untrusted; verification notice shown to users; no claim that answers are automatically correct.

## 7\. Limitations and future work

Scanned PDFs need OCR; tables and multi-column pages may extract poorly; the LLM can still misread a correct passage; no conversation memory. Future work: OCR, follow-up question memory, document filters, feedback buttons, FastAPI backend.

## 8\. Conclusion

\---

## 9\. Demo video checklist (2-3 minutes)

1. Show the GitHub repo and README.
2. Open the app, upload the PDFs, press **Process Documents**.
3. Ask 2 answerable questions and point to the source file and page.
4. Ask 1 question that is not in the documents and show the refusal.
5. Show **Clear Chat** and **Clear / replace documents**.
6. Show the evaluation sheet results.

## 10\. Viva answers (short)

* **What is RAG and why is it used?** The system first retrieves relevant passages from your documents and then asks an LLM to answer using only those passages. It reduces made-up answers and lets the model use private or new information.
* **Why split documents into chunks?** Whole documents are too long to embed and send to the LLM, and small chunks make the search more precise. Overlap keeps ideas that cross a boundary together.
* **What is an embedding?** A list of numbers (384 here) that represents the meaning of a text, so similar meanings are close together.
* **What does a vector database store?** The embeddings of all chunks plus their metadata (file, page), and it finds the nearest vectors to a question quickly.
* **How does cosine similarity help retrieval?** It measures the angle between the question vector and a chunk vector, so chunks with similar meaning score high regardless of length. Because vectors are normalised, FAISS distance gives the same ranking.
* **Why can a RAG chatbot still be wrong?** Retrieval may fetch the wrong chunk, the answer may span chunks that were not retrieved, the PDF text may extract badly, or the LLM may misread the context.
* **How do you test retrieval?** With a question sheet that lists the expected file and page, then check whether the retrieved chunks contain it (retrieval accuracy). Answer quality is checked separately.
* **What happens when the answer is not present?** If the best chunk is below the similarity threshold the bot refuses without calling the LLM; otherwise the strict prompt makes the LLM return the fixed fallback message.

