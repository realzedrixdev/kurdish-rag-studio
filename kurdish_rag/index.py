from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .normalize import tokens, trigrams


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    source: str
    ordinal: int
    text: str
    score: float


class HybridIndex:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS chunks(
          id INTEGER PRIMARY KEY, source TEXT NOT NULL, ordinal INTEGER NOT NULL,
          text TEXT NOT NULL, terms TEXT NOT NULL, grams TEXT NOT NULL,
          UNIQUE(source, ordinal));
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        """)

    def replace_source(self, source: str, chunks: list[tuple[int, str]]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM chunks WHERE source=?", (source,))
            self.connection.executemany(
                "INSERT INTO chunks(source,ordinal,text,terms,grams) VALUES(?,?,?,?,?)",
                [(source, ordinal, text, json.dumps(tokens(text), ensure_ascii=False),
                  json.dumps(sorted(trigrams(text)), ensure_ascii=False)) for ordinal, text in chunks],
            )

    def close(self) -> None:
        self.connection.close()

    def _rows(self):
        return self.connection.execute("SELECT id,source,ordinal,text,terms,grams FROM chunks").fetchall()

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        rows = self._rows()
        if not rows:
            return []
        docs = [(row, json.loads(row[4]), set(json.loads(row[5]))) for row in rows]
        query_terms, query_grams = tokens(query), trigrams(query)
        average_length = sum(len(doc[1]) for doc in docs) / len(docs) or 1
        document_frequency = Counter(term for _, terms_, _ in docs for term in set(terms_))
        bm25, fuzzy = {}, {}
        for row, terms_, grams_ in docs:
            frequencies, length = Counter(terms_), len(terms_)
            score = 0.0
            for term in query_terms:
                df = document_frequency[term]
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                score += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * length / average_length)) if tf else 0
            bm25[row[0]] = score
            union = len(query_grams | grams_)
            fuzzy[row[0]] = len(query_grams & grams_) / union if union else 0.0
        fused: Counter[int] = Counter()
        for scores in (bm25, fuzzy):
            for rank, (chunk_id, _) in enumerate(sorted(scores.items(), key=lambda item: item[1], reverse=True), 1):
                fused[chunk_id] += 1 / (60 + rank)
        by_id = {row[0]: row for row in rows}
        return [SearchResult(cid, by_id[cid][1], by_id[cid][2], by_id[cid][3], score)
                for cid, score in fused.most_common(limit)]
