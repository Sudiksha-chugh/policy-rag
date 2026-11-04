"""Compare identical queries and cutoff with and without graph traversal."""
import json
import os
from pathlib import Path
from statistics import mean

from rag import PolicyRag


def evaluate(rag, cases, use_graph, k):
    recall, reciprocal_rank, complete, abstentions = [], [], [], []
    for case in cases:
        expected = set(case["expected"])
        retrieved = [item["id"] for item in rag.retrieve(case["question"], use_graph, k)]
        if not expected:
            abstentions.append(not retrieved)
        else:
            found = expected.intersection(retrieved)
            recall.append(len(found) / len(expected))
            complete.append(found == expected)
            reciprocal_rank.append(next(
                (1 / rank for rank, item in enumerate(retrieved, start=1) if item in expected), 0
            ))
        print(f"  expected={sorted(expected)} retrieved={retrieved} | {case['question']}")
    return {
        f"recall@{k}": mean(recall),
        f"mrr@{k}": mean(reciprocal_rank),
        f"all_sources@{k}": mean(complete),
        "unanswerable_abstention": mean(abstentions),
    }


if __name__ == "__main__":
    cases = json.loads(Path(__file__).with_name("benchmark.json").read_text())
    rag = PolicyRag(os.getenv("EMBEDDING_MODEL"))
    for use_graph in (False, True):
        print("Vector + graph" if use_graph else "Vector only")
        print(json.dumps(evaluate(rag, cases, use_graph, k=2), indent=2))
