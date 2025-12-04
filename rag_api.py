import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "tiktok-rag"

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY not found in environment variables.")

# --- Initialize clients ---
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    client = None
    print(f"Failed to initialize OpenAI client: {e}")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
except Exception as e:
    index = None
    print(f"Failed to initialize Pinecone client: {e}")


# ----------------- FastAPI App --------------------
app = FastAPI(title="TikTok RAG API", version="1.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------- Models --------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    summarize: bool = True

class SearchResponse(BaseModel):
    answer: str
    matches: List[dict]


# ----------------- Helper Functions --------------------------
def embed_query(text: str) -> List[float]:
    """Generate embedding for the search query using text-embedding-3-small."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return emb.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


def generate_rag_answer(query: str, contexts: List[str]) -> str:
    """Use retrieved chunks to produce a final summarized answer using gpt-4o-mini."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    context_block = "\n\n".join(contexts)

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.

Question:
{query}

Context:
{context_block}

Answer in 2-4 sentences with bullet points when helpful.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ----------------- API Routes --------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/search.html")

@app.get("/search", response_model=SearchResponse)
def search_rag_get(q: str = Query(..., description="Search query"), top_k: int = 5):
    """GET endpoint for search."""
    return perform_search(q, top_k, summarize=True)

@app.post("/search", response_model=SearchResponse)
def search_rag_post(req: SearchRequest):
    """POST endpoint for search."""
    return perform_search(req.query, req.top_k, req.summarize)

def perform_search(query: str, top_k: int, summarize: bool) -> SearchResponse:
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    # 1. Embed query
    vector = embed_query(query)

    # 2. Query Pinecone
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    # Extract text chunks
    contexts = [match.metadata.get("summary", "") for match in results.matches if match.metadata]

    # 3. Generate final LLM answer (RAG)
    if summarize and contexts:
        answer = generate_rag_answer(query, contexts)
    else:
        answer = "Retrieved raw results only (summarize=False or no context found)."

    return SearchResponse(
        answer=answer,
        matches=[{"id": m.id, "score": m.score, "metadata": m.metadata} for m in results.matches]
    )

class IngestRequest(BaseModel):
    url: str

from fastapi import BackgroundTasks
from tiktok_rag_cloud import process_urls

@app.post("/ingest")
def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks):
    """Start background ingestion of a TikTok video."""
    background_tasks.add_task(process_urls, [req.url])
    return {"status": "processing_started", "message": f"Started processing {req.url}"}


@app.get("/health")
def health():
    return {"status": "ok", "message": "TikTok RAG API running!"}
