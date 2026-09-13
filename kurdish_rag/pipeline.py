from pathlib import Path

from .chunking import chunk_text
from .index import HybridIndex, SearchResult
from .ollama import generate


class RAGPipeline:
    def __init__(self, database: str | Path):
        self.index = HybridIndex(database)

    def ingest_directory(self, directory: str | Path) -> int:
        count = 0
        root = Path(directory)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            chunks = chunk_text(path.read_text(encoding="utf-8"))
            self.index.replace_source(str(path.relative_to(root)), [(c.ordinal, c.text) for c in chunks])
            count += len(chunks)
        return count

    def retrieve(self, question: str, limit: int = 5) -> list[SearchResult]:
        return self.index.search(question, limit)

    def ask(self, question: str, model: str, limit: int = 5) -> tuple[str, list[SearchResult]]:
        results = self.retrieve(question, limit)
        evidence = "\n\n".join(f"[{r.source}#{r.ordinal}] {r.text}" for r in results)
        prompt = ("Answer only from the evidence. Cite claims using [source#chunk]. "
                  "If evidence is insufficient, say so.\n\n"
                  f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\nANSWER:")
        return generate(prompt, model), results
