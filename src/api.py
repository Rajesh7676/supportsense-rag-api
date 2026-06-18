from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_docs import load_raw_documents
from src.rag import RAGPipeline


app = FastAPI(title="SupportSense RAG API")

# Initialize RAG pipeline once
rag = RAGPipeline()

# Ingest documents on startup (simple version)
docs = load_raw_documents()
rag.ingest(docs)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = 3


class AskResponse(BaseModel):
    answer: str
    question: str
    context: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    result = rag.answer(req.question, k=req.top_k or 3)

    return AskResponse(
        answer=result["answer"],
        question=req.question,
        context=result["prompt"],  # you can change to just context if you want
    )