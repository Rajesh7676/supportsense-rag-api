import sys
from pathlib import Path
from typing import List, Dict

# Resolve project root (SupportSense folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import RAGPipeline

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw_documents() -> List[Dict]:
    """
    Load all .txt files from data/raw into a list of dicts.
    Each dict has: doc_id, title, text, metadata.
    """
    docs: List[Dict] = []
    for file in RAW_DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(
            {
                "doc_id": file.stem,                          # e.g. "hr_policy"
                "title": file.stem.replace("_", " ").title(), # "Hr Policy"
                "text": text,
                "metadata": {
                    "source": str(file),
                },
            }
        )
    return docs


if __name__ == "__main__":
    documents = load_raw_documents()
    print(f"Loaded {len(documents)} documents")

    rag = RAGPipeline()
    rag.ingest(documents)