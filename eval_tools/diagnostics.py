import time


def measure_latency(func, *args, **kwargs):
    start = time.time()
    _ = func(*args, **kwargs)
    return time.time() - start


def compute_coverage(total_chunks: int, retrieved_chunks: int) -> float:
    return (retrieved_chunks / total_chunks) * 100


def robustness_score(original_answer: str, paraphrased_answer: str) -> float:
    # Placeholder: use similarity or edit distance
    return 100 if original_answer.strip() == paraphrased_answer.strip() else 70
