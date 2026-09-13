import argparse
import json

from .evaluation import evaluate
from .pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Offline hybrid RAG studio")
    parser.add_argument("command", choices=("ingest", "search", "ask", "evaluate"))
    parser.add_argument("value")
    parser.add_argument("--db", default="knowledge.db")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    pipeline = RAGPipeline(args.db)
    if args.command == "ingest":
        print(json.dumps({"chunks_indexed": pipeline.ingest_directory(args.value)}))
    elif args.command == "search":
        for result in pipeline.retrieve(args.value, args.limit):
            print(f"[{result.source}#{result.ordinal}] {result.score:.5f}\n{result.text}\n")
    elif args.command == "ask":
        answer, _ = pipeline.ask(args.value, args.model, args.limit)
        print(answer)
    else:
        print(json.dumps(evaluate(pipeline, args.value, args.limit), indent=2))


if __name__ == "__main__":
    main()
