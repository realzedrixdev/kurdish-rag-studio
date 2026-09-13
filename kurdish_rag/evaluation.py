import json
from pathlib import Path


def evaluate(pipeline, dataset: str | Path, k: int = 5) -> dict[str, float]:
    cases = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    reciprocal_ranks, hits, recalls = [], [], []
    for case in cases:
        expected = set(case["relevant_sources"])
        results = pipeline.retrieve(case["question"], k)
        ranks = [i for i, result in enumerate(results, 1) if result.source in expected]
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
        hits.append(float(bool(ranks)))
        recalls.append(len({r.source for r in results} & expected) / len(expected))
    size = len(cases) or 1
    return {"cases": len(cases), "mrr": sum(reciprocal_ranks) / size,
            f"hit_rate@{k}": sum(hits) / size, f"recall@{k}": sum(recalls) / size}
