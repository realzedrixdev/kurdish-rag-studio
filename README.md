# Kurdish RAG Studio

**A private, offline retrieval-augmented generation system for Kurdish, Arabic, and English documents.**

Kurdish RAG Studio turns local `.txt` and `.md` collections into a searchable knowledge base, retrieves evidence with a hybrid lexical/character model, and can ask a local Ollama model to answer with citations. Documents never need to leave the machine.

## Why it is more than a chatbot

The project implements the complete RAG pipeline rather than hiding it behind a framework:

1. Unicode-aware multilingual normalization
2. Paragraph-preserving document chunking with overlap
3. Persistent SQLite document and chunk storage
4. BM25 term scoring for precise keyword retrieval
5. Character trigram scoring for spelling and Unicode variation
6. Reciprocal-rank fusion of both result lists
7. Context-budget selection and citation assembly
8. Optional generation through a local Ollama endpoint
9. Retrieval evaluation with MRR, Recall@K, and Hit Rate@K

## Architecture

```text
documents → normalize → chunk → SQLite index
                                │
question → BM25 ────────────────┤
         → char trigrams ───────┤→ rank fusion → evidence → Ollama → cited answer
                                │
evaluation questions ───────────┘→ MRR / Recall@K / Hit Rate@K
```

## Quick start

```bash
python -m kurdish_rag ingest examples/docs --db knowledge.db
python -m kurdish_rag search "What is the AI Olympiad?" --db knowledge.db
python -m kurdish_rag ask "Why does offline AI matter?" --db knowledge.db --model llama3.2
python -m kurdish_rag evaluate examples/evaluation.jsonl --db knowledge.db
python -m unittest discover -s tests
```

The `search` command requires only Python 3.10+. The `ask` command additionally requires a running Ollama instance.

## Design choices

- **Hybrid retrieval:** Kurdish spelling and keyboard variants can make exact token matching brittle. Character trigrams recover near matches while BM25 preserves precision.
- **SQLite persistence:** portable, inspectable, transactional, and suitable for edge deployments.
- **Citations by construction:** retrieved chunk identifiers are preserved through prompting and displayed with answers.
- **Framework-free core:** the algorithms stay readable for judges, researchers, and students.

## Repository map

```text
kurdish_rag/
  normalize.py   multilingual text normalization
  chunking.py    paragraph-aware overlapping chunks
  index.py       SQLite storage and hybrid retrieval
  ollama.py      local generation client
  pipeline.py    ingestion, retrieval, answer assembly
  evaluation.py  retrieval metrics
tests/            algorithm and integration tests
examples/         sample documents and evaluation set
```

## Privacy and limitations

Search is fully offline. Generation is private when the configured Ollama endpoint is local. This prototype is intended for transparent experimentation; production deployments should add authentication, encrypted storage, and domain-specific evaluation data.

## License

MIT — built by [Karden Karwan](https://realzedrix.com).
