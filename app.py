"""Flask web app for the Velmora help-center assistant.
Run:  pip install -r requirements.txt && python app.py   ->  http://127.0.0.1:4000
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from assistant import Assistant

BASE = Path(__file__).parent
app = Flask(__name__)
assistant = Assistant(BASE / "kb")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Please type a question."), 400
    if len(question) > 500:
        return jsonify(error="Question too long (500 characters max)."), 400
    return jsonify(assistant.ask(question))


@app.get("/api/eval")
def run_eval():
    """Runs the 40 evaluation questions and returns the detailed results."""
    rows = [json.loads(l) for l in (BASE / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    results, ok = [], 0
    for r in rows:
        out = assistant.ask(r["question"])
        if r["answerable"]:
            good = (out["answerable"] and all(k in out["answer"].lower() for k in r["answer_keywords"])
                    and {s["id"] for s in out["sources"]} >= set(r["source_articles"]))
        else:
            good = not out["answerable"]
        ok += good
        results.append({"id": r["id"], "question": r["question"], "pass": good,
                        "answer": out["answer"], "sources": [s["id"] for s in out["sources"]]})
    return jsonify(score=f"{ok}/{len(rows)}", results=results)


@app.get("/api/kb")
def kb():
    return jsonify([{"id": a["id"], "title": a["title"], "category": a["category"],
                     "updated": str(a["updated"])} for a in assistant.articles.values()])


if __name__ == "__main__":
    app.run(debug=True, port=4000)
