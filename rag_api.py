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
    top_k: int = 10  # Increased for better retrieval
    summarize: bool = True
    min_score: float = 0.0  # Minimum relevance score
    author_filter: str = ""  # Filter by author

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


def generate_rag_answer(query: str, contexts: List[dict]) -> str:
    """Use retrieved chunks to produce a final summarized answer using gpt-4o."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    # Build context with citations
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get('title', 'Unknown')
        author = ctx.get('author', 'Unknown')
        summary = ctx.get('summary', '')
        context_parts.append(f"[Source {i}] Title: {title} | Author: {author}\n{summary}")
    
    context_block = "\n\n".join(context_parts)

    prompt = f"""You are an expert assistant analyzing TikTok video content.

Question: {query}

Context from videos:
{context_block}

Instructions:
- Provide a comprehensive answer based ONLY on the context above
- Cite sources using [Source X] notation
- Use bullet points for clarity when appropriate
- If the context doesn't contain enough information, acknowledge this

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Upgraded model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Increased for detailed answers
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to gpt-4o-mini if gpt-4o fails
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Error generating answer: {str(e)}"


# ----------------- API Routes --------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/search.html")

@app.get("/search", response_model=SearchResponse)
def search_rag_get(
    q: str = Query(..., description="Search query"), 
    top_k: int = 10,
    min_score: float = 0.0,
    author_filter: str = ""
):
    """GET endpoint for search."""
    return perform_search(q, top_k, summarize=True, min_score=min_score, author_filter=author_filter)

@app.post("/search", response_model=SearchResponse)
def search_rag_post(req: SearchRequest):
    """POST endpoint for search."""
    return perform_search(req.query, req.top_k, req.summarize, req.min_score, req.author_filter)

def perform_search(query: str, top_k: int, summarize: bool, min_score: float = 0.0, author_filter: str = "") -> SearchResponse:
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    # 1. Embed query
    vector = embed_query(query)

    # 2. Query Pinecone with optional metadata filter
    try:
        filter_dict = {}
        if author_filter:
            filter_dict["author"] = {"$eq": author_filter}
        
        results = index.query(
            vector=vector,
            top_k=top_k * 2,  # Get more results for filtering
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    # 3. Filter by minimum score
    filtered_matches = [m for m in results.matches if m.score >= min_score][:top_k]
    
    if not filtered_matches:
        return SearchResponse(
            answer="No results found matching your criteria.",
            matches=[]
        )

    # Extract metadata for answer generation
    contexts_with_metadata = [match.metadata for match in filtered_matches if match.metadata]

    # 4. Generate final LLM answer (RAG)
    if summarize and contexts_with_metadata:
        answer = generate_rag_answer(query, contexts_with_metadata)
    else:
        answer = "Retrieved raw results only (summarize=False or no context found)."

    return SearchResponse(
        answer=answer,
        matches=[{"id": m.id, "score": m.score, "metadata": m.metadata} for m in filtered_matches]
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
