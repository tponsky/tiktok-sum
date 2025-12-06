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
    topic: str = ""

class BulkIngestRequest(BaseModel):
    urls: List[str]
    topic: str = ""

from fastapi import BackgroundTasks
from tiktok_rag_cloud import process_urls

@app.post("/ingest")
def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks):
    """Start background ingestion of a TikTok video."""
    background_tasks.add_task(process_urls, [req.url], req.topic)
    return {"status": "processing_started", "message": f"Started processing {req.url}"}

@app.post("/ingest/bulk")
def ingest_bulk(req: BulkIngestRequest, background_tasks: BackgroundTasks):
    """Start background ingestion of multiple TikTok videos."""
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    if len(req.urls) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 URLs per batch")

    background_tasks.add_task(process_urls, req.urls, req.topic)
    return {
        "status": "processing_started",
        "message": f"Started processing {len(req.urls)} videos",
        "count": len(req.urls)
    }

@app.get("/library")
def get_library(
    topic: str = "",
    author: str = "",
    limit: int = 100
):
    """Get all unique videos in the library, optionally filtered by topic or author."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Build filter
        filter_dict = {}
        if topic:
            filter_dict["topic"] = {"$eq": topic}
        if author:
            filter_dict["author"] = {"$eq": author}

        # Query with a dummy vector to get all results (we'll use list operation instead)
        # For Pinecone, we need to use list() to get all vectors
        # However, list doesn't return metadata, so we use a workaround
        # We'll query with chunk_index=0 to get unique videos
        filter_dict["chunk_index"] = {"$eq": 0}

        # Create a zero vector for querying (returns all with filter)
        dummy_vector = [0.0] * 1536

        results = index.query(
            vector=dummy_vector,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict if filter_dict else {"chunk_index": {"$eq": 0}}
        )

        # Group by source URL
        videos = []
        seen_sources = set()

        for match in results.matches:
            meta = match.metadata or {}
            source = meta.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                videos.append({
                    "id": match.id,
                    "source": source,
                    "title": meta.get("title", "Untitled"),
                    "author": meta.get("author", "Unknown"),
                    "topic": meta.get("topic", ""),
                    "categories": meta.get("categories", meta.get("topic", "")),
                    "upload_date": meta.get("upload_date", ""),
                    "duration": meta.get("duration", 0),
                    "view_count": meta.get("view_count", 0),
                    "ingested_at": meta.get("ingested_at", 0),
                    "summary": meta.get("summary", ""),
                    "key_takeaway": meta.get("key_takeaway", ""),
                    "transcript": meta.get("transcript", ""),
                })

        # Sort by ingested_at descending (most recent first)
        videos.sort(key=lambda x: x.get("ingested_at", 0), reverse=True)

        return {"videos": videos, "count": len(videos)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch library: {str(e)}")

@app.get("/topics")
def get_topics():
    """Get all unique topics in the library."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get a large sample to find topics
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}}
        )

        topics = set()
        for match in results.matches:
            meta = match.metadata or {}
            topic = meta.get("topic", "")
            if topic:
                topics.add(topic)

        return {"topics": sorted(list(topics))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch topics: {str(e)}")

@app.get("/authors")
def get_authors():
    """Get all unique authors in the library."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}}
        )

        authors = set()
        for match in results.matches:
            meta = match.metadata or {}
            author = meta.get("author", "")
            if author:
                authors.add(author)

        return {"authors": sorted(list(authors))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch authors: {str(e)}")


class UpdateCategoriesRequest(BaseModel):
    video_id: str
    categories: List[str]  # List of category names

class UpdateVideoRequest(BaseModel):
    video_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    key_takeaway: Optional[str] = None

@app.post("/video/update")
def update_video_metadata(req: UpdateVideoRequest):
    """Update the title, summary, or key takeaway for a video."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get the video's source URL from the provided ID
        result = index.fetch(ids=[req.video_id])
        if not result.vectors or req.video_id not in result.vectors:
            raise HTTPException(status_code=404, detail="Video not found")

        video_meta = result.vectors[req.video_id].metadata
        source_url = video_meta.get("source", "")

        if not source_url:
            raise HTTPException(status_code=400, detail="Video has no source URL")

        # Find all chunks for this video and update them
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"source": {"$eq": source_url}}
        )

        # Update each chunk's metadata
        updated_count = 0
        for match in results.matches:
            meta = match.metadata.copy()

            # Update fields if provided
            if req.title is not None:
                meta["title"] = req.title
            if req.summary is not None:
                meta["summary"] = req.summary
            if req.key_takeaway is not None:
                meta["key_takeaway"] = req.key_takeaway

            # Upsert with updated metadata
            vec_result = index.fetch(ids=[match.id])
            if vec_result.vectors and match.id in vec_result.vectors:
                vector_data = vec_result.vectors[match.id]
                index.upsert(vectors=[{
                    "id": match.id,
                    "values": vector_data.values,
                    "metadata": meta
                }])
                updated_count += 1

        return {
            "status": "success",
            "message": f"Updated {updated_count} chunks",
            "title": req.title,
            "summary": req.summary,
            "key_takeaway": req.key_takeaway
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update video: {str(e)}")


@app.post("/video/categories")
def update_video_categories(req: UpdateCategoriesRequest):
    """Update the categories for a video (by updating all its chunks)."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get the video's source URL from the provided ID
        result = index.fetch(ids=[req.video_id])
        if not result.vectors or req.video_id not in result.vectors:
            raise HTTPException(status_code=404, detail="Video not found")

        video_meta = result.vectors[req.video_id].metadata
        source_url = video_meta.get("source", "")

        if not source_url:
            raise HTTPException(status_code=400, detail="Video has no source URL")

        # Categories as comma-separated string
        categories_str = ",".join(req.categories) if req.categories else "Uncategorized"
        primary_topic = req.categories[0] if req.categories else "Uncategorized"

        # Find all chunks for this video and update them
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"source": {"$eq": source_url}}
        )

        # Update each chunk's metadata
        updated_count = 0
        for match in results.matches:
            meta = match.metadata.copy()
            meta["categories"] = categories_str
            meta["topic"] = primary_topic

            # Upsert with updated metadata (need to fetch the vector first)
            vec_result = index.fetch(ids=[match.id])
            if vec_result.vectors and match.id in vec_result.vectors:
                vector_data = vec_result.vectors[match.id]
                index.upsert(vectors=[{
                    "id": match.id,
                    "values": vector_data.values,
                    "metadata": meta
                }])
                updated_count += 1

        return {
            "status": "success",
            "message": f"Updated {updated_count} chunks",
            "categories": req.categories
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update categories: {str(e)}")


class AddCategoryRequest(BaseModel):
    name: str

# Store custom categories in memory (in production, use a database)
custom_categories = set()

@app.post("/categories")
def add_category(req: AddCategoryRequest):
    """Add a new custom category."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Category name cannot be empty")
    custom_categories.add(name)
    return {"status": "success", "category": name}

@app.delete("/categories/{name}")
def delete_category(name: str):
    """Remove a custom category."""
    if name in custom_categories:
        custom_categories.discard(name)
    return {"status": "success", "message": f"Category '{name}' removed"}

@app.get("/categories")
def get_all_categories():
    """Get all categories (both from videos and custom)."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get categories from existing videos
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}}
        )

        video_categories = set()
        for match in results.matches:
            meta = match.metadata or {}
            # Get categories (comma-separated string)
            cats = meta.get("categories", meta.get("topic", ""))
            if cats:
                for cat in cats.split(","):
                    cat = cat.strip()
                    if cat:
                        video_categories.add(cat)

        # Combine with custom categories
        all_categories = video_categories.union(custom_categories)

        return {"categories": sorted(list(all_categories))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "message": "TikTok RAG API running!"}
