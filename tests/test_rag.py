import tempfile
import unittest
from pathlib import Path

from kurdish_rag.chunking import chunk_text
from kurdish_rag.pipeline import RAGPipeline


class RAGTests(unittest.TestCase):
    def test_chunking_preserves_all_content(self):
        chunks = chunk_text("First paragraph.\n\n" + "second " * 80, max_chars=120, overlap_chars=10)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([c.ordinal for c in chunks], list(range(len(chunks))))

    def test_hybrid_search_handles_kurdish_unicode_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp, "docs"); docs.mkdir()
            Path(docs, "erbil.txt").write_text("هەولێر پایتەختی هەرێمی کوردستانە.", encoding="utf-8")
            Path(docs, "ai.txt").write_text("زیرەکی دەستکرد بۆ چارەسەری کێشەکان بەکاردێت.", encoding="utf-8")
            pipeline = RAGPipeline(Path(temp, "index.db")); pipeline.ingest_directory(docs)
            self.assertEqual(pipeline.retrieve("پايتەختی كوردستان", 1)[0].source, "erbil.txt")
            pipeline.index.close()


if __name__ == "__main__": unittest.main()
