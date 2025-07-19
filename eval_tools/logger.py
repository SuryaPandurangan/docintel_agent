import json


def save_eval_to_jsonl(question, answer, sources, metrics, filename="eval_logs.jsonl"):
    entry = {
        "question": question,
        "answer": answer,
        "sources": [doc.page_content for doc in sources],
        "metrics": metrics,
    }
    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")
