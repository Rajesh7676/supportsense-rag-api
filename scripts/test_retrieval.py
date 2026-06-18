import sys
from pathlib import Path

# Add project root (SupportSense) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_docs import load_raw_documents
from src.rag import RAGPipeline


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_raw_documents()
    print(f"Loaded {len(docs)} documents")

    rag = RAGPipeline()
    print("Ingesting into vector store...")
    rag.ingest(docs)

    query = "How many casual leave days do I get per year?"
    print(f"\nQuery: {query}")

    result = rag.answer(query, k=3)

    print("\n===== PROMPT TO LLM =====\n")
    print(result["prompt"])

    print("\n===== LLM ANSWER =====\n")
    print(result["answer"])

    print("\n===== RETRIEVED DOC METADATA =====\n")
    for i, d in enumerate(result["retrieved_docs"], start=1):
        print(f"[{i}] source={d.metadata}")