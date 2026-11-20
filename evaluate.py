"""Run the 40 questions of eval.jsonl through the assistant and print a report.
Judge = the keywords of the dataset (rough!) + abstention check. Use --verbose to read every answer."""
import json, sys
from assistant import Assistant

def main(verbose=False):
    a = Assistant("kb")
    rows = [json.loads(l) for l in open("eval.jsonl", encoding="utf-8")]
    ok = 0
    for r in rows:
        out = a.ask(r["question"])
        if r["answerable"]:
            ans = out["answer"].lower()
            good = out["answerable"] and all(k in ans for k in r["answer_keywords"])
            src_ok = {s["id"] for s in out["sources"]} >= set(r["source_articles"])
            good_all = good and src_ok
        else:
            good = good_all = not out["answerable"]
        ok += good_all
        if verbose or not good_all:
            print(("PASS " if good_all else "FAIL "), r["id"], r["question"])
            print("     ->", out["answer"].replace("\n", " | ")[:300], out["confidence"], [s["id"] for s in out["sources"]])
    print(f"\n{ok}/{len(rows)} correct")

if __name__ == "__main__":
    main("--verbose" in sys.argv)
