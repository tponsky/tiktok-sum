import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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

def summarize_transcript(transcript: str) -> str:
    """Summarize a transcript using GPT-4o-mini."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    prompt = (
        "Summarize the following transcript in up to 120 words. "
        "Include key bullet points and a one-sentence top summary.\n\n"
        f"{transcript[:3000]}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def extract_key_takeaway(transcript: str, summary: str, title: str = "") -> str:
    """Extract the single most important takeaway from a video."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    prompt = f"""Based on the video content below, extract the ONE most important takeaway, tip, trick, fact, or insight.

Title: {title}
Summary: {summary}
Transcript excerpt: {transcript[:1500]}

Instructions:
- Identify the key actionable tip, surprising fact, useful trick, or important insight
- Write it as a single, clear, concise sentence (max 25 words)
- Focus on what makes this video valuable - the "aha moment"
- Start with an action verb or key noun (e.g., "Use...", "The secret is...", "Always...")

Key Takeaway:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.3,
    )
    takeaway = resp.choices[0].message.content.strip()
    takeaway = takeaway.strip('"\'.,')
    if takeaway.lower().startswith("key takeaway:"):
        takeaway = takeaway[13:].strip()
    return takeaway


def auto_categorize(summary: str, title: str = "") -> str:
    """Use GPT to automatically categorize a video based on its summary and title."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    prompt = f"""Based on the video title and summary below, assign ONE category/topic that best describes this content.

Title: {title}
Summary: {summary}

Choose from common categories like:
- Fitness & Health
- Cooking & Recipes
- Beauty & Skincare
- Fashion & Style
- Finance & Money
- Technology & Gadgets
- Entertainment & Comedy
- Education & Learning
- Travel & Adventure
- Relationships & Dating
- Productivity & Self-Improvement
- Music & Dance
- Sports
- Gaming
- DIY & Crafts
- Parenting & Family
- Pets & Animals
- News & Current Events
- Science & Nature
- Art & Design

Or create a new specific category if none of these fit well.

Respond with ONLY the category name, nothing else. Keep it short (1-3 words)."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.3,
    )
    category = resp.choices[0].message.content.strip()
    category = category.strip('"\'.,')
    return category


def generate_title(transcript: str, summary: str, key_takeaway: str = "") -> str:
    """Generate a descriptive title based on the video content."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    prompt = f"""Based on the video content below, create a short, descriptive title (5-10 words max).

Key Takeaway: {key_takeaway}
Summary: {summary}
Transcript excerpt: {transcript[:1000]}

Instructions:
- Create a title that captures the main topic or value of the video
- Make it specific and descriptive (not generic like "Great Tips" or "Must Watch")
- Keep it concise: 5-10 words maximum
- Don't use quotes or colons
- Examples of good titles: "How to Use Pumeli for AI Marketing", "The Secret to Perfect Sourdough Bread", "5 Tax Deductions Most People Miss"

Title:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.3,
    )
    title = resp.choices[0].message.content.strip()
    title = title.strip('"\'.,')
    if title.lower().startswith("title:"):
        title = title[6:].strip()
    return title


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
    """Remove a category and reassign affected videos to other categories or Uncategorized."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    # Remove from custom categories if present
    if name in custom_categories:
        custom_categories.discard(name)

    # Reassign videos synchronously so it completes before returning
    reassigned_count = reassign_videos_from_category(name)

    return {"status": "success", "message": f"Category '{name}' removed. {reassigned_count} video(s) reassigned."}


def reassign_videos_from_category(category_to_remove: str) -> int:
    """Reassign videos from a deleted category. Returns count of reassigned videos."""
    reassigned = 0
    try:
        # Get all existing categories (except the one being deleted)
        dummy_vector = [0.0] * 1536

        # First, get chunk_index=0 to identify unique videos and their categories
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}}
        )

        # Collect existing categories and find videos to update
        existing_categories = set()
        videos_to_update = []  # List of (source_url, new_categories, new_topic)

        for match in results.matches:
            meta = match.metadata or {}
            cats = meta.get("categories", "")
            source_url = meta.get("source", "")

            if cats:
                for cat in cats.split(","):
                    cat = cat.strip()
                    if cat and cat != category_to_remove:
                        existing_categories.add(cat)

            # Check if this video has the category to remove
            cats_list = [c.strip() for c in cats.split(",") if c.strip()] if cats else []

            if category_to_remove in cats_list:
                # Remove the deleted category
                new_cats = [c for c in cats_list if c != category_to_remove]

                # If video has no remaining categories, reassign based on content
                if not new_cats:
                    transcript = meta.get("transcript", "")
                    summary = meta.get("summary", "")

                    if (transcript or summary) and existing_categories:
                        new_category = auto_categorize_to_existing(summary, transcript, existing_categories)
                        new_cats = [new_category] if new_category else ["Uncategorized"]
                    else:
                        new_cats = ["Uncategorized"]

                videos_to_update.append({
                    "source": source_url,
                    "categories": ", ".join(new_cats),
                    "topic": new_cats[0]
                })

        print(f"Deleting category '{category_to_remove}'. Found {len(videos_to_update)} videos to update.")
        print(f"Existing categories: {existing_categories}")

        # Now update ALL chunks for each affected video
        for video_info in videos_to_update:
            source_url = video_info["source"]
            new_categories = video_info["categories"]
            new_topic = video_info["topic"]

            if not source_url:
                continue

            # Find all chunks for this video
            chunk_results = index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={"source": {"$eq": source_url}}
            )

            # Update each chunk
            for chunk in chunk_results.matches:
                try:
                    index.update(
                        id=chunk.id,
                        set_metadata={"categories": new_categories, "topic": new_topic}
                    )
                except Exception as e:
                    print(f"Error updating chunk {chunk.id}: {e}")

            print(f"Updated all chunks for video: {source_url} -> {new_categories}")
            reassigned += 1

    except Exception as e:
        print(f"Error reassigning videos from category '{category_to_remove}': {e}")

    return reassigned


def auto_categorize_to_existing(summary: str, transcript: str, existing_categories: set) -> str:
    """Categorize a video into one of the existing categories."""
    if not client or not existing_categories:
        return "Uncategorized"

    categories_list = ", ".join(sorted(existing_categories))

    prompt = f"""Based on the video content below, assign the MOST relevant category from this list:
{categories_list}

Summary: {summary[:500]}
Transcript excerpt: {transcript[:500]}

Instructions:
- Choose ONLY from the categories listed above
- If none fit well, respond with "Uncategorized"
- Respond with ONLY the category name, nothing else

Category:"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        category = resp.choices[0].message.content.strip().strip('"\'.,')

        # Verify it's in existing categories or return Uncategorized
        if category in existing_categories:
            return category
        return "Uncategorized"
    except Exception as e:
        print(f"Auto-categorization error: {e}")
        return "Uncategorized"

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

        # Combine with custom categories and always include Uncategorized
        all_categories = video_categories.union(custom_categories)
        all_categories.add("Uncategorized")

        return {"categories": sorted(list(all_categories))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "message": "TikTok RAG API running!"}


class ReprocessRequest(BaseModel):
    video_id: str


class BulkReprocessRequest(BaseModel):
    video_ids: List[str] = []  # Empty list means reprocess all


@app.post("/video/reprocess")
def reprocess_video(req: ReprocessRequest):
    """Reprocess a single video to regenerate summary, key takeaway, and category."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Fetch the video's current data
        result = index.fetch(ids=[req.video_id])

        if not result.vectors or req.video_id not in result.vectors:
            raise HTTPException(status_code=404, detail="Video not found")

        vector_data = result.vectors[req.video_id]
        meta = vector_data.metadata or {}

        transcript = meta.get("transcript", "")

        if not transcript:
            raise HTTPException(status_code=400, detail="Video has no transcript to reprocess")

        # Generate new summary, key takeaway, category, and title
        new_summary = summarize_transcript(transcript)
        new_key_takeaway = extract_key_takeaway(transcript, new_summary, "")
        new_category = auto_categorize(new_summary, "")
        new_title = generate_title(transcript, new_summary, new_key_takeaway)

        # Update metadata
        meta["summary"] = new_summary
        meta["key_takeaway"] = new_key_takeaway
        meta["topic"] = new_category
        meta["categories"] = new_category
        meta["title"] = new_title

        # Update the vector with new metadata
        index.update(
            id=req.video_id,
            set_metadata=meta
        )

        return {
            "status": "success",
            "video_id": req.video_id,
            "title": new_title,
            "summary": new_summary,
            "key_takeaway": new_key_takeaway,
            "category": new_category
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {str(e)}")


@app.post("/video/reprocess/bulk")
def reprocess_bulk(req: BulkReprocessRequest, background_tasks: BackgroundTasks):
    """Reprocess multiple videos (or all if no IDs provided) in the background."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # If no IDs provided, get all videos
        if not req.video_ids:
            # Query to get all videos (chunk_index=0)
            dummy_vector = [0.0] * 1536
            results = index.query(
                vector=dummy_vector,
                top_k=1000,
                include_metadata=True,
                filter={"chunk_index": {"$eq": 0}}
            )
            video_ids = [match.id for match in results.matches]
        else:
            video_ids = req.video_ids

        if not video_ids:
            return {"status": "success", "message": "No videos to reprocess", "count": 0}

        # Process in background
        background_tasks.add_task(reprocess_videos_task, video_ids)

        return {
            "status": "processing_started",
            "message": f"Started reprocessing {len(video_ids)} videos",
            "count": len(video_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk reprocess failed: {str(e)}")


def reprocess_videos_task(video_ids: List[str]):
    """Background task to reprocess multiple videos."""
    for video_id in video_ids:
        try:
            result = index.fetch(ids=[video_id])

            if not result.vectors or video_id not in result.vectors:
                print(f"Video {video_id} not found, skipping")
                continue

            vector_data = result.vectors[video_id]
            meta = vector_data.metadata or {}

            transcript = meta.get("transcript", "")

            if not transcript:
                print(f"Video {video_id} has no transcript, skipping")
                continue

            # Generate new summary, key takeaway, category, and title
            new_summary = summarize_transcript(transcript)
            new_key_takeaway = extract_key_takeaway(transcript, new_summary, "")
            new_category = auto_categorize(new_summary, "")
            new_title = generate_title(transcript, new_summary, new_key_takeaway)

            # Update metadata
            meta["summary"] = new_summary
            meta["key_takeaway"] = new_key_takeaway
            meta["topic"] = new_category
            meta["categories"] = new_category
            meta["title"] = new_title

            # Update the vector
            index.update(
                id=video_id,
                set_metadata=meta
            )

            print(f"Reprocessed video {video_id}: {new_title}")

        except Exception as e:
            print(f"Error reprocessing {video_id}: {e}")
