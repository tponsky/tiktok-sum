import os
from typing import List, Optional
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import stripe
import simple_auth

# --- Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "tiktok-rag"

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
APP_URL = os.getenv("APP_URL", "https://tiktoksum.staycurrentapp.com")

# Pricing constants
# -----------------------------------------------
# Your actual API costs (approx):
#   - Search: ~$0.0005 (embedding ~$0.00002 + GPT-4o-mini ~$0.0003-0.0005)
#   - Ingest: ~$0.005-0.01 (transcript + summary + embeddings)
#
# With 5x markup (adjust MARKUP_MULTIPLIER to change):
MARKUP_MULTIPLIER = 5.0
BASE_COST_SEARCH = 0.001   # Your actual cost per search
BASE_COST_INGEST = 0.006   # Your actual cost per video ingest (avg)

COST_PER_SEARCH = BASE_COST_SEARCH * MARKUP_MULTIPLIER  # $0.005 per search
COST_PER_INGEST = BASE_COST_INGEST * MARKUP_MULTIPLIER  # $0.03 per video ingest
INITIAL_BALANCE = 2.00   # $2 trial for new users
RELOAD_AMOUNT = 10.00    # $10 reload
RELOAD_THRESHOLD = 1.00  # Suggest reload when < $1

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

# Security
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    """Get current authenticated user from JWT token. Returns None if not authenticated."""
    if not credentials:
        return None

    token = credentials.credentials
    token_data = simple_auth.verify_token(token)
    if not token_data or not token_data.email:
        return None

    user = simple_auth.get_user_by_email(token_data.email)
    if not user:
        return None

    return {
        "user_id": user['id'],
        "email": user['email'],
        "balance": user['balance_usd']
    }


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Require authentication - raises 401 if not authenticated."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    token_data = simple_auth.verify_token(token)
    if not token_data or not token_data.email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = simple_auth.get_user_by_email(token_data.email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "user_id": user['id'],
        "email": user['email'],
        "balance": user['balance_usd']
    }


# ----------------- Models --------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10  # Increased for better retrieval
    summarize: bool = True
    min_score: float = 0.0  # Minimum relevance score
    author_filter: str = ""  # Filter by author
    include_web: bool = False  # Also search the web

class SearchResponse(BaseModel):
    answer: str
    matches: List[dict]
    web_results: str = ""  # Web search results if requested


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


def generate_rag_answer(query: str, contexts: List[dict], web_results: str = "") -> str:
    """Use retrieved chunks to produce a final summarized answer using gpt-4o."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    # Build context with citations - include key_takeaway, author, and source URL
    context_parts = []
    source_map = {}  # Map source numbers to URLs for clickable links

    for i, ctx in enumerate(contexts, 1):
        title = ctx.get('title', 'Unknown')
        author = ctx.get('author', 'Unknown')
        key_takeaway = ctx.get('key_takeaway', '')
        summary = ctx.get('summary', '')
        source_url = ctx.get('source', '')

        # Store source mapping for later
        source_map[i] = {"url": source_url, "title": title, "author": author}

        # Build rich context including key takeaway
        context_entry = f"[Source {i}] \"{title}\" by @{author}"
        if key_takeaway:
            context_entry += f"\nKey Insight: {key_takeaway}"
        context_entry += f"\nSummary: {summary}"
        context_parts.append(context_entry)

    context_block = "\n\n".join(context_parts)

    # Add web results if available
    web_section = ""
    if web_results:
        web_section = f"\n\nAdditional information from web search:\n{web_results}"

    prompt = f"""You are an expert assistant analyzing video content from creators.

Question: {query}

Video sources from your library:
{context_block}{web_section}

Instructions:
- Answer the question using the information provided above
- When citing sources, mention the creator's name (e.g., "@username recommends..." or "According to @username...")
- Include specific details, tips, or recommendations from the videos
- Use [Source X] notation after mentioning advice from that source
- If web results are included, you can also reference those as [Web]
- If a source doesn't clearly relate to the question, don't force it into your answer
- Be specific and actionable in your response

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

        # Add source links at the bottom
        if source_map:
            answer += "\n\n---\n**Sources:**"
            for num, info in source_map.items():
                if info["url"]:
                    answer += f"\n[Source {num}]: [{info['title'][:50]}...]({info['url']}) by @{info['author']}"

        return answer
    except Exception as e:
        # Fallback to gpt-4o-mini if gpt-4o fails
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()

            if source_map:
                answer += "\n\n---\n**Sources:**"
                for num, info in source_map.items():
                    if info["url"]:
                        answer += f"\n[Source {num}]: [{info['title'][:50]}...]({info['url']}) by @{info['author']}"

            return answer
        except:
            return f"Error generating answer: {str(e)}"


# ----------------- API Routes --------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("static/search.html")

@app.get("/search", response_model=SearchResponse)
async def search_rag_get(
    q: str = Query(..., description="Search query"),
    top_k: int = 10,
    min_score: float = 0.0,
    author_filter: str = "",
    include_web: bool = False,
    user: dict = Depends(require_auth)
):
    """GET endpoint for search (requires authentication)."""
    # Check balance
    user_id = user['user_id']
    balance = simple_auth.get_user_balance(user_id)
    if balance < COST_PER_SEARCH:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Current: ${balance:.2f}, Required: ${COST_PER_SEARCH:.2f}"
        )

    # Perform search (filtered by user_id for per-user content)
    result = perform_search(q, top_k, summarize=True, min_score=min_score, author_filter=author_filter, include_web=include_web, user_id=user_id)

    # Deduct cost and log usage
    simple_auth.deduct_from_balance(user_id, COST_PER_SEARCH)
    simple_auth.log_usage(user_id, "search", cost_usd=COST_PER_SEARCH, details=q[:100])

    return result


@app.post("/search", response_model=SearchResponse)
async def search_rag_post(req: SearchRequest, user: dict = Depends(require_auth)):
    """POST endpoint for search (requires authentication)."""
    # Check balance
    user_id = user['user_id']
    balance = simple_auth.get_user_balance(user_id)
    if balance < COST_PER_SEARCH:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Current: ${balance:.2f}, Required: ${COST_PER_SEARCH:.2f}"
        )

    # Perform search (filtered by user_id for per-user content)
    result = perform_search(req.query, req.top_k, req.summarize, req.min_score, req.author_filter, req.include_web, user_id=user_id)

    # Deduct cost and log usage
    simple_auth.deduct_from_balance(user_id, COST_PER_SEARCH)
    simple_auth.log_usage(user_id, "search", cost_usd=COST_PER_SEARCH, details=req.query[:100])

    return result

def expand_query_keywords(query: str) -> set:
    """Expand query with synonyms and related terms using GPT."""
    if not client:
        return set(query.lower().split())

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Given this search query, list the key search terms plus synonyms and closely related words.

Query: "{query}"

Return ONLY a comma-separated list of words (lowercase, no explanations). Include:
- Original key terms from the query
- Synonyms (e.g., learning -> studying, education, tutorial)
- Related concepts (e.g., AI -> artificial intelligence, machine learning)
- Word variations (e.g., learn -> learning, learner)

Example output: learning, studying, education, tutorial, course, train, teach"""
            }],
            max_tokens=100,
            temperature=0.3,
        )
        expanded = response.choices[0].message.content.strip().lower()
        # Parse comma-separated words
        words = set(w.strip() for w in expanded.split(",") if w.strip() and len(w.strip()) > 2)
        # Also include original query words
        words.update(w.lower() for w in query.split() if len(w) > 2)
        return words
    except Exception as e:
        print(f"Query expansion error: {e}")
        return set(w.lower() for w in query.split() if len(w) > 2)


def search_web(query: str) -> str:
    """Search the web using GPT's knowledge and Perplexity-style summarization."""
    if not client:
        return ""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Provide a brief, helpful summary of current information about: "{query}"

Focus on:
- Recent developments or updates (2024-2025)
- Key tools, apps, or resources related to this topic
- Best practices or recommendations from experts

Keep it concise (3-5 bullet points). If this is about a very specific personal topic where web info wouldn't help, just say "No additional web information available."

Response:"""
            }],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Web search error: {e}")
        return ""


def perform_search(query: str, top_k: int, summarize: bool, min_score: float = 0.0, author_filter: str = "", include_web: bool = False, user_id: int = 0) -> SearchResponse:
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    # 1. Expand query with synonyms for better keyword matching
    expanded_keywords = expand_query_keywords(query)
    print(f"Expanded keywords: {expanded_keywords}")

    # 1b. Optionally search the web
    web_results = ""
    if include_web:
        web_results = search_web(query)
        print(f"Web results: {web_results[:200]}...")

    # 2. Embed original query (vector search uses semantic meaning)
    vector = embed_query(query)

    # 3. Query Pinecone with optional metadata filter
    # Search user's own videos + shared/public (user_id=0) + legacy (no user_id)
    try:
        filter_dict = {}
        if author_filter:
            filter_dict["author"] = {"$eq": author_filter}

        # Get more results then filter by user access
        results = index.query(
            vector=vector,
            top_k=top_k * 5,  # Get more results for user filtering + reranking
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        # Filter to only show user's own content + shared content
        accessible_matches = []
        for m in results.matches:
            meta = m.metadata or {}
            video_user_id = meta.get("user_id")
            # Allow: own content, shared content (0), or legacy content (no user_id)
            if video_user_id is None or video_user_id == 0 or video_user_id == user_id:
                accessible_matches.append(m)

        # Replace results.matches with filtered list
        results.matches = accessible_matches

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    # 4. Hybrid reranking: boost results that have keyword matches (using expanded keywords)
    scored_matches = []

    for m in results.matches:
        meta = m.metadata or {}
        # Check for keyword matches in title, key_takeaway, summary, and categories
        title = (meta.get("title", "") or "").lower()
        key_takeaway = (meta.get("key_takeaway", "") or "").lower()
        summary = (meta.get("summary", "") or "").lower()
        categories = (meta.get("categories", "") or "").lower()
        topic = (meta.get("topic", "") or "").lower()
        combined_text = f"{title} {key_takeaway} {summary} {categories} {topic}"

        # Count keyword matches using expanded keywords (includes synonyms)
        keyword_matches = sum(1 for word in expanded_keywords if word in combined_text)

        # Boost score based on keyword matches (hybrid scoring)
        keyword_boost = keyword_matches * 0.03  # 3% boost per keyword match
        hybrid_score = m.score + keyword_boost

        scored_matches.append({
            "match": m,
            "hybrid_score": hybrid_score,
            "keyword_matches": keyword_matches
        })

    # Sort by hybrid score
    scored_matches.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # 4. Filter by minimum score and take top_k
    filtered_matches = [
        sm["match"] for sm in scored_matches
        if sm["match"].score >= min_score
    ][:top_k]

    if not filtered_matches:
        return SearchResponse(
            answer="No results found matching your criteria.",
            matches=[]
        )

    # Extract metadata for answer generation
    contexts_with_metadata = [match.metadata for match in filtered_matches if match.metadata]

    # 5. Generate final LLM answer (RAG)
    if summarize and contexts_with_metadata:
        answer = generate_rag_answer(query, contexts_with_metadata, web_results)
    else:
        answer = "Retrieved raw results only (summarize=False or no context found)."

    return SearchResponse(
        answer=answer,
        matches=[{"id": m.id, "score": m.score, "metadata": m.metadata} for m in filtered_matches],
        web_results=web_results
    )

class IngestRequest(BaseModel):
    url: str
    topic: str = ""

class BulkIngestRequest(BaseModel):
    urls: List[str]
    topic: str = ""

class ShortcutIngestRequest(BaseModel):
    url: str
    topic: str = ""
    api_key: str

from tiktok_rag_cloud import process_urls
import plistlib
import base64


@app.get("/api/shortcut/download")
async def download_shortcut(user: dict = Depends(require_auth)):
    """Generate and download a personalized iOS Shortcut with the user's API key pre-filled."""
    import tempfile

    user_id = user['user_id']
    api_key = simple_auth.get_user_api_key(user_id)

    if not api_key:
        # Auto-generate an API key if user doesn't have one
        api_key = simple_auth.generate_api_key(user_id)
        if not api_key:
            raise HTTPException(status_code=500, detail="Failed to generate API key")

    # Check if template shortcut exists
    template_path = os.path.join(os.path.dirname(__file__), "static", "shortcut_template.shortcut")

    if os.path.exists(template_path):
        # Read template and inject API key
        with open(template_path, 'rb') as f:
            try:
                shortcut_data = plistlib.load(f)
                # Find and replace the API key and APP_URL placeholders in the shortcut actions
                shortcut_data = inject_placeholder_into_shortcut(shortcut_data, api_key, APP_URL)

                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.shortcut') as tmp:
                    plistlib.dump(shortcut_data, tmp)
                    tmp_path = tmp.name

                return FileResponse(
                    tmp_path,
                    media_type='application/octet-stream',
                    filename='Save to Video Library.shortcut',
                    headers={"Content-Disposition": "attachment; filename=\"Save to Video Library.shortcut\""}
                )
            except Exception as e:
                print(f"Error processing shortcut template: {e}")
                raise HTTPException(status_code=500, detail="Failed to generate shortcut")
    else:
        # No template - return instructions with API key
        return {
            "message": "Shortcut template not found. Please set up manually.",
            "api_key": api_key,
            "endpoint": f"{APP_URL}/api/shortcut/ingest"
        }


def inject_placeholder_into_shortcut(shortcut_data: dict, api_key: str, app_url: str) -> dict:
    """Recursively search and replace placeholders in shortcut data."""
    def replace_in_obj(obj):
        if isinstance(obj, dict):
            return {k: replace_in_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_in_obj(item) for item in obj]
        elif isinstance(obj, str):
            # Replace placeholders
            res = obj.replace("{{API_KEY}}", api_key).replace("YOUR_API_KEY_HERE", api_key)
            res = res.replace("{{APP_URL}}", app_url).replace("https://tiktoksum.staycurrentapp.com", app_url)
            return res
        else:
            return obj

    return replace_in_obj(shortcut_data)


# Shortcut-friendly endpoint that uses API key instead of JWT
@app.post("/api/shortcut/ingest")
async def shortcut_ingest_video(request: Request, background_tasks: BackgroundTasks):
    """Ingest a video using API key (for iOS Shortcuts)."""
    # Try to get data from JSON body
    try:
        data = await request.json()
    except:
        data = {}

    # Get API key from body or header
    api_key = data.get("api_key") or request.headers.get("X-API-Key")
    raw_url = data.get("url")
    topic = data.get("topic", "")

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not raw_url:
        raise HTTPException(status_code=400, detail="Missing video URL")

    # Extract clean URL from potential caption text
    from tiktok_rag_cloud import URL_REGEX
    url_match = URL_REGEX.search(raw_url)
    if not url_match:
        # Fallback if URL doesn't match tiktok specific regex but is a URL
        url_match = re.search(r'https?://\S+', raw_url)
    
    url = url_match.group(0) if url_match else raw_url

    # Authenticate via API key
    user = simple_auth.get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not user.get('is_active', True):
        raise HTTPException(status_code=401, detail="Account is disabled")

    user_id = user['id']
    balance = user['balance_usd']

    # Check balance
    if balance < COST_PER_INGEST:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Current: ${balance:.2f}, Required: ${COST_PER_INGEST:.2f}"
        )

    # Deduct cost and log usage
    simple_auth.deduct_from_balance(user_id, COST_PER_INGEST)
    simple_auth.log_usage(user_id, "shortcut_ingest", cost_usd=COST_PER_INGEST, details=url[:100])

    # Process video
    background_tasks.add_task(process_urls, [url], topic, user_id)

    new_balance = simple_auth.get_user_balance(user_id)
    return {
        "status": "processing_started",
        "message": f"Video added to your library",
        "url": url,
        "balance_remaining": f"${new_balance:.2f}"
    }


@app.post("/ingest")
async def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_auth)):
    """Start background ingestion of a video (requires authentication)."""
    # Check balance
    user_id = user['user_id']
    balance = simple_auth.get_user_balance(user_id)
    if balance < COST_PER_INGEST:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Current: ${balance:.2f}, Required: ${COST_PER_INGEST:.2f}"
        )

    # Deduct cost and log usage upfront
    simple_auth.deduct_from_balance(user_id, COST_PER_INGEST)
    simple_auth.log_usage(user_id, "ingest", cost_usd=COST_PER_INGEST, details=req.url[:100])

    # Pass user_id for per-user library
    background_tasks.add_task(process_urls, [req.url], req.topic, user_id)
    return {"status": "processing_started", "message": f"Started processing {req.url}"}

@app.post("/ingest/bulk")
async def ingest_bulk(req: BulkIngestRequest, background_tasks: BackgroundTasks, user: dict = Depends(require_auth)):
    """Start background ingestion of multiple videos (requires authentication)."""
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    if len(req.urls) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 URLs per batch")

    # Check balance for all videos
    user_id = user['user_id']
    total_cost = COST_PER_INGEST * len(req.urls)
    balance = simple_auth.get_user_balance(user_id)
    if balance < total_cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient balance. Current: ${balance:.2f}, Required: ${total_cost:.2f} for {len(req.urls)} videos"
        )

    # Deduct cost and log usage upfront
    simple_auth.deduct_from_balance(user_id, total_cost)
    simple_auth.log_usage(user_id, "bulk_ingest", cost_usd=total_cost, details=f"{len(req.urls)} videos")

    # Pass user_id for per-user library
    background_tasks.add_task(process_urls, req.urls, req.topic, user_id)
    return {
        "status": "processing_started",
        "message": f"Started processing {len(req.urls)} videos",
        "count": len(req.urls),
        "cost": total_cost
    }

@app.get("/library")
def get_library(
    topic: str = "",
    author: str = "",
    limit: int = 100,
    user: dict = Depends(get_current_user)
):
    """Get all unique videos in the library for the current user.

    Shows:
    - User's own videos (user_id matches)
    - Shared/public videos (user_id = 0, including pre-existing content)
    """
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get user_id (0 if not authenticated)
        current_user_id = user['user_id'] if user else 0

        # Build filter - show user's videos AND shared videos (user_id=0)
        filter_dict = {"chunk_index": {"$eq": 0}}

        if topic:
            filter_dict["topic"] = {"$eq": topic}
        if author:
            filter_dict["author"] = {"$eq": author}

        # Create a zero vector for querying (returns all with filter)
        dummy_vector = [0.0] * 1536

        # Query for user's own videos
        user_filter = {**filter_dict, "user_id": {"$eq": current_user_id}}
        user_results = index.query(
            vector=dummy_vector,
            top_k=limit,
            include_metadata=True,
            filter=user_filter
        )

        # Query for shared/public videos (user_id=0 or missing)
        # Note: Pre-existing videos won't have user_id field, so we also get those
        shared_filter = {**filter_dict, "user_id": {"$eq": 0}}
        shared_results = index.query(
            vector=dummy_vector,
            top_k=limit,
            include_metadata=True,
            filter=shared_filter
        )

        # Also query for videos without user_id field (legacy content)
        # Pinecone doesn't easily support "field doesn't exist", so we get all and filter
        all_filter = {**filter_dict}
        all_results = index.query(
            vector=dummy_vector,
            top_k=limit * 2,
            include_metadata=True,
            filter=all_filter
        )

        # Combine results
        videos = []
        seen_sources = set()

        def add_video(match):
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
                    "user_id": meta.get("user_id", 0),
                    "is_own": meta.get("user_id", 0) == current_user_id and current_user_id != 0,
                })

        # Add user's own videos first
        for match in user_results.matches:
            add_video(match)

        # Add shared videos
        for match in shared_results.matches:
            add_video(match)

        # Add legacy videos (those without user_id)
        for match in all_results.matches:
            meta = match.metadata or {}
            if "user_id" not in meta:
                add_video(match)

        # Sort by ingested_at descending (most recent first)
        videos.sort(key=lambda x: x.get("ingested_at", 0), reverse=True)

        return {"videos": videos[:limit], "count": len(videos[:limit])}

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

        # Build the metadata update dict
        metadata_update = {}
        if req.title is not None:
            metadata_update["title"] = req.title
        if req.summary is not None:
            metadata_update["summary"] = req.summary
        if req.key_takeaway is not None:
            metadata_update["key_takeaway"] = req.key_takeaway

        if not metadata_update:
            return {"status": "success", "message": "No fields to update", "updated_count": 0}

        # Find all chunks for this video using multiple query strategies
        dummy_vector = [0.0] * 1536
        all_chunk_ids = set()

        # Strategy 1: Query with source filter
        results = index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            filter={"source": {"$eq": source_url}}
        )
        for match in results.matches:
            all_chunk_ids.add(match.id)

        # Strategy 2: Also try fetching by ID pattern (timestamp_0, timestamp_1, etc.)
        # The video_id should be like "1234567890_0"
        base_id = req.video_id.rsplit("_", 1)[0] if "_" in req.video_id else req.video_id
        potential_ids = [f"{base_id}_{i}" for i in range(20)]  # Check up to 20 chunks
        fetch_result = index.fetch(ids=potential_ids)
        if fetch_result.vectors:
            for vid in fetch_result.vectors:
                all_chunk_ids.add(vid)

        print(f"Found {len(all_chunk_ids)} chunks for video {source_url}")

        # Update each chunk using index.update() for efficiency
        updated_count = 0
        for chunk_id in all_chunk_ids:
            try:
                index.update(
                    id=chunk_id,
                    set_metadata=metadata_update
                )
                updated_count += 1
            except Exception as e:
                print(f"Error updating chunk {chunk_id}: {e}")

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

    print(f"=== DELETE CATEGORY REQUEST ===")
    print(f"Category to delete: '{name}'")

    # Remove from custom categories if present
    if name in custom_categories:
        custom_categories.discard(name)

    # Get library data first (this reliably returns all videos)
    library_data = get_library(limit=1000)
    videos = library_data.get("videos", [])

    print(f"Retrieved {len(videos)} videos from library")

    # Find videos with this category
    videos_to_update = []
    existing_categories = set()

    for video in videos:
        cats_str = video.get("categories", "")
        cats_list = [c.strip() for c in cats_str.split(",") if c.strip()] if cats_str else []

        # Collect all categories except the one being deleted
        for cat in cats_list:
            if cat != name:
                existing_categories.add(cat)

        # Check if this video has the category to delete
        if name in cats_list:
            print(f"Found video with category '{name}': {video.get('title', 'Unknown')[:50]}")
            videos_to_update.append({
                "id": video.get("id"),
                "source": video.get("source"),
                "current_categories": cats_list,
                "summary": video.get("summary", ""),
                "transcript": video.get("transcript", "")
            })

    print(f"Found {len(videos_to_update)} videos to update")
    print(f"Existing categories: {existing_categories}")

    # Update each video
    reassigned_count = 0
    dummy_vector = [0.0] * 1536

    for video_info in videos_to_update:
        source_url = video_info["source"]
        current_cats = video_info["current_categories"]

        # Remove the deleted category
        new_cats = [c for c in current_cats if c != name]

        # If no categories left, assign to Uncategorized or re-categorize
        if not new_cats:
            summary = video_info.get("summary", "")
            transcript = video_info.get("transcript", "")

            if (summary or transcript) and existing_categories:
                new_category = auto_categorize_to_existing(summary, transcript, existing_categories)
                new_cats = [new_category] if new_category else ["Uncategorized"]
            else:
                new_cats = ["Uncategorized"]

        new_categories_str = ",".join(new_cats)
        new_topic = new_cats[0]

        # Find and update all chunks for this video
        try:
            chunk_results = index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={"source": {"$eq": source_url}}
            )

            for chunk in chunk_results.matches:
                index.update(
                    id=chunk.id,
                    set_metadata={"categories": new_categories_str, "topic": new_topic}
                )

            print(f"Updated {len(chunk_results.matches)} chunks: {source_url[:50]} -> {new_categories_str}")
            reassigned_count += 1

        except Exception as e:
            print(f"Error updating video {source_url}: {e}")

    return {"status": "success", "message": f"Category '{name}' removed. {reassigned_count} video(s) reassigned."}


def reassign_videos_from_category(category_to_remove: str) -> int:
    """Reassign videos from a deleted category. Returns count of reassigned videos."""
    reassigned = 0
    try:
        # Get all existing categories (except the one being deleted)
        dummy_vector = [0.0] * 1536

        # Get ALL videos by querying with a much larger limit and potentially multiple batches
        # Pinecone serverless has a max of 10,000 per query
        all_matches = []
        batch_size = 10000

        # First batch
        results = index.query(
            vector=dummy_vector,
            top_k=batch_size,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}}
        )
        all_matches.extend(results.matches)

        # If we got a full batch, there might be more - but for now, 10k should be enough
        # In production, you'd want to implement pagination using Pinecone's list() API
        print(f"Retrieved {len(all_matches)} videos from Pinecone")

        # Collect existing categories and find videos to update
        existing_categories = set()
        videos_to_update = []  # List of (source_url, new_categories, new_topic)

        for match in all_matches:
            meta = match.metadata or {}
            cats = meta.get("categories", "")
            source_url = meta.get("source", "")

            # Debug logging
            if not source_url:
                print(f"Warning: Video {match.id} has no source URL")

            if cats:
                for cat in cats.split(","):
                    cat = cat.strip()
                    if cat and cat != category_to_remove:
                        existing_categories.add(cat)

            # Check if this video has the category to remove
            cats_list = [c.strip() for c in cats.split(",") if c.strip()] if cats else []

            # Debug: Print sample of first 5 videos
            if len(all_matches) <= 5 or match == all_matches[0]:
                print(f"Sample video {match.id}: raw_categories='{cats}' (repr: {repr(cats)}), parsed={cats_list}")

            if category_to_remove in cats_list:
                print(f"Found video with category '{category_to_remove}': {match.id} (source: {source_url})")
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
                    "categories": ",".join(new_cats),  # No space after comma to match storage format
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
                print(f"Skipping video with no source URL")
                continue

            # Find all chunks for this video
            chunk_results = index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={"source": {"$eq": source_url}}
            )

            print(f"Found {len(chunk_results.matches)} chunks for video {source_url}")

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
        import traceback
        traceback.print_exc()

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
        # Use larger limit to get all videos (max 10,000)
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
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
    return {
        "status": "ok", 
        "message": "TikTok RAG API running!",
        "version": "1.0.2",
        "last_update": "2025-12-19T13:30Z"
    }


@app.get("/test-route")
def test_route():
    return {"message": "Test route is working!"}


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


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.get("/login.html", include_in_schema=False)
async def login_page():
    return FileResponse("static/login.html")

@app.get("/signup.html", include_in_schema=False)
async def signup_page():
    return FileResponse("static/signup.html")


@app.post("/api/auth/signup", response_model=simple_auth.Token)
async def signup(user: simple_auth.UserCreate):
    """Create a new user account with $2 trial balance."""
    try:
        # Check if user already exists
        existing_user = simple_auth.get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create new user
        new_user = simple_auth.create_user(user.email, user.password)
        if not new_user:
            raise HTTPException(status_code=500, detail="Failed to create user")

        # Generate access token
        access_token_expires = timedelta(minutes=simple_auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = simple_auth.create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )

        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/api/auth/login", response_model=simple_auth.Token)
async def login(user: simple_auth.UserLogin):
    """Authenticate and get access token."""
    authenticated_user = simple_auth.authenticate_user(user.email, user.password)
    if not authenticated_user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # Generate access token
    access_token_expires = timedelta(minutes=simple_auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = simple_auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(require_auth)):
    """Get current user information."""
    return {
        "user_id": user['user_id'],
        "email": user['email'],
        "balance_usd": round(user['balance'], 2)
    }


# ============================================================================
# Password Reset Endpoints
# ============================================================================

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class VerifyResetTokenRequest(BaseModel):
    token: str


@app.post("/api/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Request a password reset. Sends reset link via email."""
    import email_service

    token = simple_auth.create_password_reset_token(req.email)

    # For security, always return success even if email doesn't exist
    # This prevents email enumeration attacks
    if token:
        reset_url = f"{APP_URL}/reset-password.html?token={token}"
        print(f"Password reset requested for {req.email}")

        # Send email with reset link
        if email_service.is_configured():
            email_sent = email_service.send_password_reset_email(req.email, reset_url)
            if not email_sent:
                print(f"Warning: Failed to send reset email to {req.email}")
        else:
            print(f"Warning: Email service not configured. Reset URL: {reset_url}")

    return {"message": "If an account exists with this email, a password reset link has been sent."}


@app.post("/api/auth/verify-reset-token")
async def verify_reset_token(req: VerifyResetTokenRequest):
    """Verify that a password reset token is valid."""
    token_data = simple_auth.verify_reset_token(req.token)

    if token_data:
        return {
            "valid": True,
            "email": token_data['email']
        }

    return {"valid": False, "email": None}


@app.post("/api/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Reset password using a valid token."""
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    success = simple_auth.reset_password(req.token, req.new_password)

    if success:
        return {"message": "Password has been reset successfully. You can now log in."}

    raise HTTPException(status_code=400, detail="Invalid or expired reset token")


@app.get("/forgot-password.html", include_in_schema=False)
async def forgot_password_page():
    return FileResponse("static/forgot-password.html")


@app.get("/reset-password.html", include_in_schema=False)
async def reset_password_page():
    return FileResponse("static/reset-password.html")


# ============================================================================
# Stripe Billing Endpoints
# ============================================================================

class UsageResponse(BaseModel):
    balance_usd: float
    total_spent: float
    needs_reload: bool


class CheckoutSessionResponse(BaseModel):
    session_url: str


@app.get("/api/user/usage", response_model=UsageResponse)
async def get_user_usage_endpoint(user: dict = Depends(require_auth)):
    """Get user's current balance and usage."""
    user_id = user['user_id']
    balance = simple_auth.get_user_balance(user_id)
    total_spent = simple_auth.get_user_total_cost(user_id)
    needs_reload = balance < RELOAD_THRESHOLD

    return UsageResponse(
        balance_usd=round(balance, 2),
        total_spent=round(total_spent, 2),
        needs_reload=needs_reload
    )


# ============================================================================
# API Key Management (for Shortcuts integration)
# ============================================================================

@app.get("/api/user/api-key")
async def get_api_key(user: dict = Depends(require_auth)):
    """Get user's current API key."""
    user_id = user['user_id']
    api_key = simple_auth.get_user_api_key(user_id)
    return {"api_key": api_key}


@app.post("/api/user/api-key/generate")
async def generate_api_key(user: dict = Depends(require_auth)):
    """Generate a new API key (replaces existing one)."""
    user_id = user['user_id']
    api_key = simple_auth.generate_api_key(user_id)
    if api_key:
        return {"api_key": api_key, "message": "API key generated successfully"}
    raise HTTPException(status_code=500, detail="Failed to generate API key")


@app.delete("/api/user/api-key")
async def revoke_api_key(user: dict = Depends(require_auth)):
    """Revoke user's API key."""
    user_id = user['user_id']
    if simple_auth.revoke_api_key(user_id):
        return {"message": "API key revoked successfully"}
    raise HTTPException(status_code=500, detail="Failed to revoke API key")


@app.post("/api/stripe/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(user: dict = Depends(require_auth)):
    """Create a Stripe checkout session for adding funds."""
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    user_id = user['user_id']
    user_email = user['email']

    # Get or create Stripe customer
    customer_id = simple_auth.get_stripe_customer_id(user_id)

    if not customer_id:
        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=user_email,
            metadata={"user_id": str(user_id)}
        )
        customer_id = customer.id
        simple_auth.set_stripe_customer_id(user_id, customer_id)

    # Create checkout session for $10
    try:
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": "Video RAG Search Credits",
                        "description": f"Add ${RELOAD_AMOUNT:.2f} to your account balance"
                    },
                    "unit_amount": int(RELOAD_AMOUNT * 100),  # Convert to cents
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=f"{APP_URL}/?payment=success",
            cancel_url=f"{APP_URL}/?payment=cancelled",
            metadata={
                "user_id": str(user_id),
                "amount_usd": str(RELOAD_AMOUNT)
            }
        )

        return CheckoutSessionResponse(session_url=session.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {str(e)}")


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle checkout.session.completed event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = int(session["metadata"].get("user_id"))
        
        # Use amount_total from session for security (source of truth), convert cents to dollars
        amount_usd = float(session.get("amount_total", 0)) / 100.0

        if user_id and amount_usd > 0:
            # Add funds to user's balance
            simple_auth.add_to_balance(user_id, amount_usd)
            
            # Log the deposit (cost_usd=0 because it's a credit, not spending)
            simple_auth.log_usage(
                user_id, 
                "deposit", 
                cost_usd=0.0, 
                details=f"Stripe Deposit: ${amount_usd:.2f}"
            )
            print(f"Added ${amount_usd:.2f} to user {user_id}'s balance")

    return {"status": "success"}


@app.get("/api/debug/status")
async def debug_status(user: dict = Depends(require_auth)):
    """Debug endpoint to check logs and balance."""
    user_id = user['user_id']
    usage = simple_auth.get_user_usage(user_id)
    
    # Format usage for readability
    formatted_usage = []
    for row in usage:
        formatted_usage.append({
            "timestamp": row['timestamp'],
            "action": row['action'],
            "cost": row['cost_usd'],
            "details": row['details']
        })
        
    return {
        "user_id": user_id,
        "email": user['email'],
        "balance": simple_auth.get_user_balance(user_id),
        "recent_usage": formatted_usage,
        "openai_configured": OPENAI_API_KEY is not None,
        "pinecone_configured": PINECONE_API_KEY is not None,
        "env_app_url": APP_URL
    }


# ============================================================================
# Admin Endpoints
# ============================================================================

# Your admin email - only this user can run admin commands
ADMIN_EMAIL = "tponsky@gmail.com"


@app.post("/api/admin/claim-legacy-videos")
async def claim_legacy_videos(user: dict = Depends(require_auth)):
    """Claim all legacy videos (without user_id) for the current user.
    Only the admin user can run this.
    """
    if user['email'] != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Admin access required")

    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        # Get all videos without user_id
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True
        )

        claimed_count = 0
        user_id = user['user_id']

        for match in results.matches:
            meta = match.metadata or {}
            # Only update if user_id is not set
            if "user_id" not in meta:
                # Update metadata to add user_id
                index.update(
                    id=match.id,
                    set_metadata={"user_id": user_id}
                )
                claimed_count += 1

        return {
            "status": "success",
            "claimed_count": claimed_count,
            "user_id": user_id,
            "message": f"Claimed {claimed_count} legacy videos for user {user['email']}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to claim videos: {str(e)}")


@app.post("/api/admin/transfer-all-videos")
async def transfer_all_videos(from_user_id: int, user: dict = Depends(require_auth)):
    """Transfer all videos from one user to the admin user.
    Only the admin user can run this.
    """
    if user['email'] != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Admin access required")

    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized")

    try:
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True
        )

        transferred_count = 0
        to_user_id = user['user_id']

        for match in results.matches:
            meta = match.metadata or {}
            if meta.get("user_id") == from_user_id:
                index.update(
                    id=match.id,
                    set_metadata={"user_id": to_user_id}
                )
                transferred_count += 1

        return {
            "status": "success",
            "transferred_count": transferred_count,
            "from_user_id": from_user_id,
            "to_user_id": to_user_id,
            "message": f"Transferred {transferred_count} videos from user {from_user_id} to {user['email']}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transfer videos: {str(e)}")
