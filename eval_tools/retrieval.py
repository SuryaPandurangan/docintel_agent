from typing import List


def precision_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([d for d in retrieved_at_k if d in relevant_set]) / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([d for d in retrieved_at_k if d in relevant_set]) / len(relevant_set)


def hit_rate_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> int:
    return int(any(doc in relevant for doc in retrieved[:k]))


def mrr(relevant: List[str], retrieved: List[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0
