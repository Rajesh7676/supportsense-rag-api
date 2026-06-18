from typing import List, Dict
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from client import call_llm  # Groq LLM client


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict


class SentenceTransformerEmbeddingFunction:
    """
    Wrapper so that Chroma can call embed_documents / embed_query
    using a SentenceTransformer model under the hood.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        # Used by Chroma when indexing documents
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str):
        # Used by Chroma when querying
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class RAGPipeline:
    """
    Handles:
    - Chunking raw documents
    - Building embeddings
    - Creating a Chroma vector store
    - Retrieving relevant chunks
    - Building a prompt and calling LLM
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 150,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "chroma_store",
    ):
        # 1) Chunking configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # 2) Embedding model wrapper
        self.embedding_model = SentenceTransformerEmbeddingFunction(
            embedding_model_name
        )

        # 3) Vector store config
        self.persist_directory = persist_directory
        self.vectorstore: Chroma | None = None

    # ---------- Step 1+2: raw docs -> chunks ----------
    def build_chunks(self, docs: List[Dict]) -> List[Chunk]:
        """
        Convert raw documents (with text + metadata) into smaller chunks.
        """
        chunks: List[Chunk] = []
        for doc in docs:
            split_docs = self.text_splitter.create_documents(
                texts=[doc["text"]],
                metadatas=[{"doc_id": doc["doc_id"], **doc["metadata"]}],
            )
            for idx, sd in enumerate(split_docs):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}_{idx}",
                        doc_id=doc["doc_id"],
                        text=sd.page_content,
                        metadata=sd.metadata,
                    )
                )
        return chunks

    # ---------- Step 3: embeddings + vector store ----------
    def build_vectorstore(self, chunks: List[Chunk]):
        """
        Build a Chroma vector store from chunks using SentenceTransformer embeddings.
        """
        texts = [c.text for c in chunks]
        metadatas = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, **c.metadata}
            for c in chunks
        ]

        # Create Chroma DB using our embedding wrapper
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        # Chroma 0.4+ auto-persist; this call is optional
        self.vectorstore.persist()

    # ---------- Retrieval ----------
    def retrieve(self, query: str, k: int = 4):
        """
        Retrieve top-k most similar chunks for a given query.
        """
        # If vectorstore is not in memory yet, load it from disk
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory,
            )

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def build_context_from_docs(self, docs) -> str:
        """
        Join retrieved documents into a single context string.
        """
        parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            source = meta.get("source") or meta.get("doc_id") or f"chunk_{i}"
            parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def answer(self, query: str, k: int = 4) -> Dict:
        """
        Full RAG answer pipeline:
        - Retrieve relevant chunks
        - Build prompt from context
        - Call external LLM (Groq) to get final answer
        """
        # 1) Retrieval
        retrieved_docs = self.retrieve(query, k=k)

        # 2) Context build
        context = self.build_context_from_docs(retrieved_docs)

        # 3) Prompt build
        prompt = (
            "You are an internal support assistant. "
            "Use ONLY the context below to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # 4) LLM call (Groq via client.py)
        llm_answer = call_llm(prompt)

        # 5) Return everything (useful for debugging / UI)
        return {
            "prompt": prompt,
            "retrieved_docs": retrieved_docs,
            "answer": llm_answer,
        }

    # ---------- Convenience: full ingestion pipeline ----------
    def ingest(self, docs: List[Dict]):
        """
        Full ingestion pipeline:
        - Build chunks
        - Build vector store
        """
        chunks = self.build_chunks(docs)
        print(f"Built {len(chunks)} chunks")
        self.build_vectorstore(chunks)
        print("Vector store built and persisted")