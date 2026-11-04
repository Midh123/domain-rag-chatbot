"""Testing and evaluation sheet (guide section 10).

Usage:
    python evaluate.py                 # full test: retrieval + LLM answers + refusal + time
    python evaluate.py --retrieval-only   # no LLM / no API key needed
Results are saved to tests/evaluation_results.csv
"""
import argparse
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

import config
import document_loader
from rag_pipeline import RAGPipeline

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.BASE_DIR / "tests" / "test_questions.csv"))
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    pages, report = document_loader.load_folder()
    for r in report:
        print(f"{r['status']:9} {r['name']}: {r['message']}")
    pipe = RAGPipeline()
    print("Chunks indexed:", pipe.build(pages, save=False))

    with open(args.csv, newline="") as f:
        questions = list(csv.DictReader(f))

    rows = []
    for q in questions:
        unavailable = q["type"] == "unavailable"
        if args.retrieval_only:
            hits = pipe.retrieve(q["question"])
            retrieved = [{"source": d.metadata["source"], "page": d.metadata["page"]} for d, _ in hits]
            best = hits[0][1] if hits else 0
            answer, secs = "(LLM not run)", 0
            refused = best < config.MIN_SIMILARITY
        else:
            res = pipe.ask(q["question"])
            retrieved, answer, secs, refused = res["retrieved"], res["answer"], res["seconds"], res["refused"]

        found = [f"{r['source']} p{r['page']}" for r in retrieved]
        if unavailable:
            retrieval_ok = ""
            correct = refused if not args.retrieval_only else "n/a"
        else:
            wanted = (q["expected_source"], int(q["expected_page"]))
            retrieval_ok = any((r["source"], r["page"]) == wanted for r in retrieved)
            if args.retrieval_only:
                correct = retrieval_ok
            else:
                kw = q["expected_keyword"].lower()
                correct = retrieval_ok and not refused and (kw in answer.lower() if kw else True)
        rows.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "expected_source": q["expected_source"] if unavailable else f"{q['expected_source']}, page {q['expected_page']}",
            "retrieved_sources": "; ".join(found), "retrieval_correct": retrieval_ok,
            "refused": refused, "answer": answer.replace("\n", " "), "correct": correct, "seconds": secs,
        })
        print(f"[{q['id']:>2}] {q['type']:<11} correct={correct}  {secs}s  {q['question']}")
        if not args.retrieval_only:
            time.sleep(0.3)                      # stay inside free-tier rate limits

    out = config.BASE_DIR / "tests" / "evaluation_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ans = [r for r in rows if r["type"] != "unavailable"]
    una = [r for r in rows if r["type"] == "unavailable"]
    print("\n=== SUMMARY ===")
    print(f"Retrieval accuracy (right file + page in top-{config.TOP_K}): "
          f"{sum(r['retrieval_correct'] is True for r in ans)}/{len(ans)}")
    if not args.retrieval_only:
        print(f"Answer correctness (answerable + paraphrased): {sum(r['correct'] is True for r in ans)}/{len(ans)}")
        print(f"Refusal quality (unavailable questions refused): {sum(r['correct'] is True for r in una)}/{len(una)}")
        times = [r["seconds"] for r in rows]
        print(f"Average response time: {sum(times)/len(times):.2f} s (max {max(times):.2f} s)")
    print("Saved:", out)
    print("Note: 'correct' uses a keyword check. Read the answers in the CSV for groundedness and source quality.")


if __name__ == "__main__":
    main()
