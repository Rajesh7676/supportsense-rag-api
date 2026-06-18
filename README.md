# SupportSense RAG API

SupportSense is a Retrieval-Augmented Generation (RAG) backend built with FastAPI, ChromaDB, sentence-transformers, and Groq.  
It lets you ask natural-language questions about internal SupportSense documents (for example, HR policies) and returns concise, context-aware answers.

---

## Features

- Question answering over your own documents using a RAG pipeline.
- FastAPI REST API with clean JSON responses.
- Groq LLM integration to generate grounded answers from retrieved context.
- Separate ingestion script, RAG pipeline, and API layer for easier maintenance.
- Environment-variable based secret handling (no API keys hardcoded in code).

---

## Project Structure

```text
SupportSense/
├─ client.py                # Example Python client to call the /ask endpoint
├─ requirements.txt         # Python dependencies
├─ pyproject.toml           # Optional project metadata/config
├─ uv.lock                  # Lockfile for reproducible installs
├─ .gitignore               # Ignores .env, virtualenv, cache, and vector store
├─ data/
│  └─ raw/                  # Source documents (for example, HR policy text files)
├─ scripts/
│  └─ ingest_docs.py        # Script to build the vector store from raw docs
├─ src/
│  ├─ api.py                # FastAPI app exposing /health and /ask
│  └─ rag.py                # RAGPipeline: embeddings, ChromaDB, Groq integration
└─ README.md                # Project documentation (this file)
```

---

## How It Works

1. Document ingestion  
   - `scripts/ingest_docs.py` reads text files from `data/raw/`.  
   - The content is split into chunks, embedded with a sentence-transformers model, and stored in a persistent ChromaDB collection (for example, in a `chroma_store/` directory on disk).

2. RAG pipeline  
   - `src/rag.py` defines a `RAGPipeline` that:
     - Loads the persisted ChromaDB vector store.
     - Uses a retriever to fetch the top-k most relevant chunks for a question.
     - Builds a prompt that includes both the user question and retrieved context.
     - Sends the prompt to a Groq LLM (using the `groq` Python client) and returns the answer along with useful metadata.

3. FastAPI service  
   - `src/api.py` creates a FastAPI app with:
     - `GET /health` – simple health check endpoint.
     - `POST /ask` – main RAG endpoint that accepts a question and optional parameters (such as `top_k`) and returns the generated answer.

---

## Requirements

- Python 3.10 or newer
- A Groq API key (from https://console.groq.com)
- Dependencies listed in `requirements.txt`, which include (at a high level):
  - `fastapi`, `uvicorn`
  - `chromadb`
  - `sentence-transformers`
  - `groq`
  - `langchain` and related utilities

Install dependencies locally:

```bash
pip install -r requirements.txt
```

---

## Local Setup

1. Clone the repository

```bash
git clone https://github.com/<your-username>/supportsense-rag-api.git
cd supportsense-rag-api
```

2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Set your Groq API key

Create a `.env` file in the project root (this file is ignored by git):

```env
GROQ_API_KEY=your_groq_api_key_here
```

The code (for example in `rag.py` or `api.py`) reads this via `os.getenv("GROQ_API_KEY")`.

5. Add your documents

Place your internal text files (HR policies, FAQs, and similar content) into `data/raw/`, for example:

```text
data/raw/
├─ leave_policy.txt
├─ work_from_home_policy.txt
└─ general_guidelines.txt
```

6. Build the vector store

Run the ingestion script:

```bash
python scripts/ingest_docs.py
```

This will:

- Load documents from `data/raw/`
- Chunk and embed them
- Persist vectors to a local ChromaDB directory (for example, `chroma_store/`)

You should see logs such as “Loaded N documents”, “Built M chunks”, and “Vector store built and persisted”.

7. Run the FastAPI app

From the project root:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Now you can open:

- Swagger UI: http://localhost:8000/docs  
- Health check: http://localhost:8000/health

---

## API Usage

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Ask a question

```http
POST /ask
Content-Type: application/json
```

Example request body:

```json
{
  "question": "How many casual leave days do I get per year?",
  "top_k": 3
}
```

Example response shape (field names may differ depending on your Pydantic models):

```json
{
  "question": "How many casual leave days do I get per year?",
  "answer": "You get 12 days of casual leave per year.",
  "contexts": [
    {
      "content": "Employees are entitled to 12 days of casual leave per calendar year...",
      "source": "leave_policy.txt"
    }
  ],
  "model": "llama-3.1-8b-instant"
}
```

Adjust the fields to match your actual `/ask` response schema.

---

## Example Python Client

`client.py` shows how to call the `/ask` endpoint from Python:

```python
import requests

API_URL = "http://localhost:8000/ask"

payload = {
    "question": "What is the work from home policy?",
    "top_k": 3
}

response = requests.post(API_URL, json=payload, timeout=60)
response.raise_for_status()
print(response.json())
```

Run it with:

```bash
python client.py
```

---

## Future Improvements

- Switch or tune the embedding model for better speed or quality.
- Add loaders for PDF, DOCX, or other formats.
- Add authentication and authorization for production use.
- Add better logging and monitoring for debugging and observability.

---

## License

This project is intended for learning and portfolio purposes.  
