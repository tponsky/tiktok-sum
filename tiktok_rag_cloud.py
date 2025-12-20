"""
tiktok_rag_cloud.py

Cloud pipeline for VidRecall:
- Reads video links from clipboard
- Downloads audio via yt-dlp
- Converts audio to WAV (ffmpeg)
- Sends to OpenAI speech-to-text
- Splits transcript into chunks, summarizes each chunk with OpenAI
- Generates embeddings with OpenAI
- Upserts embeddings + metadata into Pinecone index
"""

import os
import re
import tempfile
import subprocess
import time
from typing import List
import pyperclip
import yt_dlp
from dotenv import load_dotenv
import simple_auth

# --- Load ENV ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var required")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY env var required")

# --- OpenAI Client ---
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Pinecone ---
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "tiktok-rag"

existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# --- Regex ---
URL_REGEX = re.compile(r"https?://(?:www\.)?tiktok\.com/[^\s]+")

# -------- Helpers -------- #

def get_urls_from_clipboard() -> List[str]:
    text = pyperclip.paste()
    return URL_REGEX.findall(text)


def download_audio(url: str, out_dir: str) -> tuple[str, dict]:
    """Download audio using yt-dlp and return WAV path + video metadata"""
    # Set platform-specific headers for better compatibility
    url_lower = url.lower()
    
    if 'tiktok.com' in url_lower:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.tiktok.com/",
        }
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.youtube.com/",
        }
    elif 'twitter.com' in url_lower or 'x.com' in url_lower:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://twitter.com/",
        }
    else:
        # Default headers for other platforms
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "http_headers": headers,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fname = ydl.prepare_filename(info)

    base, _ = os.path.splitext(fname)
    wav_path = base + ".wav"

    cmd = ["ffmpeg", "-y", "-i", fname, "-ac", "1", "-ar", "16000", wav_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract metadata
    metadata = {
        "title": info.get("title", ""),
        "author": info.get("uploader", info.get("creator", "")),
        "upload_date": info.get("upload_date", ""),
        "duration": info.get("duration", 0),
        "view_count": info.get("view_count", 0),
    }

    return wav_path, metadata


def transcribe_with_openai(audio_path: str) -> str:
    """Transcribe audio with GPT-4o-mini-transcribe"""
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return resp.text


def simple_sent_tokenize(text: str) -> List[str]:
    """Simple sentence tokenizer using regex (no NLTK dependency)."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Chunk text with overlap to preserve context."""
    sents = simple_sent_tokenize(text)
    chunks, cur = [], ""

    for s in sents:
        if len(cur) + len(s) + 1 > max_chars:
            if cur:
                chunks.append(cur.strip())
                # Start next chunk with overlap from end of current chunk
                words = cur.split()
                overlap_text = " ".join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                cur = overlap_text + " " + s if overlap_text else s
            else:
                cur = s
        else:
            cur += " " + s

    if cur.strip():
        chunks.append(cur.strip())

    return chunks


def summarize_chunk(chunk: str) -> str:
    prompt = (
        "Summarize the following transcript chunk in up to 120 words. "
        "Include key bullet points and a one-sentence top summary.\n\n"
        f"{chunk}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def extract_key_takeaway(transcript: str, summary: str, title: str = "") -> str:
    """Extract the single most important takeaway, tip, or insight from a video."""
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

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.3,
        )
        takeaway = resp.choices[0].message.content.strip()
        # Clean up
        takeaway = takeaway.strip('"\'.,')
        # Remove "Key Takeaway:" prefix if present
        if takeaway.lower().startswith("key takeaway:"):
            takeaway = takeaway[13:].strip()
        return takeaway
    except Exception as e:
        print(f"Key takeaway extraction error: {e}")
        return ""


def generate_title(transcript: str, summary: str, key_takeaway: str = "") -> str:
    """Generate a descriptive title based on the video content."""
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

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.3,
        )
        title = resp.choices[0].message.content.strip()
        # Clean up
        title = title.strip('"\'.,')
        # Remove "Title:" prefix if present
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        return title
    except Exception as e:
        print(f"Title generation error: {e}")
        return ""


def auto_categorize(summary: str, title: str = "", existing_categories: set = None) -> str:
    """Use GPT to automatically categorize a video based on its summary and title.

    If existing_categories is provided, only assigns from those categories.
    Otherwise falls back to 'Uncategorized'.
    """
    # If we have existing categories, only use those
    if existing_categories:
        categories_list = ", ".join(sorted(existing_categories))
        prompt = f"""Based on the video title and summary below, assign ONE category from this list that best describes this content.

Title: {title}
Summary: {summary}

Available categories: {categories_list}

You MUST choose one of the categories from the list above. If none fit well, respond with "Uncategorized".

Respond with ONLY the exact category name from the list, nothing else."""
    else:
        # No existing categories - use Uncategorized
        return "Uncategorized"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.3,
        )
        category = resp.choices[0].message.content.strip()
        # Clean up the response - remove quotes, periods, etc.
        category = category.strip('"\'.,')

        # Verify it's in existing categories
        if existing_categories and category in existing_categories:
            return category
        return "Uncategorized"
    except Exception as e:
        print(f"Auto-categorization error: {e}")
        return "Uncategorized"


def embed_text(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in resp.data]


def upsert_to_pinecone(vectors: List[dict]):
    index.upsert(vectors)


def get_existing_categories(user_id: int = None) -> set:
    """Fetch all existing categories for a user from Pinecone."""
    try:
        # Build filter for user_id
        filter_dict = {}
        if user_id is not None:
            filter_dict["user_id"] = user_id

        # Query with a dummy vector to get all videos
        dummy_vector = [0.0] * 1536
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        categories = set()
        for match in results.matches:
            meta = match.metadata or {}
            cats = meta.get("categories", meta.get("topic", ""))
            if cats:
                for cat in cats.split(","):
                    cat = cat.strip()
                    if cat and cat != "Uncategorized":
                        categories.add(cat)

        return categories
    except Exception as e:
        print(f"Error fetching existing categories: {e}")
        return set()


# -------- Main Pipeline -------- #

def process_urls(urls: List[str], topic: str = "", user_id: int = None):
    """Process TikTok URLs and ingest them into Pinecone.

    Args:
        urls: List of TikTok URLs to process
        topic: Optional topic/category for organizing videos
        user_id: Optional user ID for per-user libraries (None = shared/public)
    """
    # Fetch existing categories for this user (only once at the start)
    existing_categories = get_existing_categories(user_id) if user_id else set()
    if existing_categories:
        print(f"Found {len(existing_categories)} existing categories for user")

    with tempfile.TemporaryDirectory() as tmpdir:
        for url in urls:
            print("Processing:", url)

            # 1. Download audio and get metadata
            try:
                wav, video_metadata = download_audio(url, tmpdir)
            except Exception as e:
                print("Download error:", e)
                if user_id:
                    # Refund the user if download fails
                    simple_auth.add_to_balance(user_id, 0.03)
                    simple_auth.log_usage(user_id, "ingest_error", details=f"Download failed for {url}: {str(e)} (Refunded $0.03)")
                continue

            # 2. Transcribe
            try:
                transcript = transcribe_with_openai(wav)
            except Exception as e:
                print("Transcription error:", e)
                if user_id:
                    simple_auth.log_usage(user_id, "ingest_error", details=f"Transcription failed for {url}: {str(e)}")
                continue

            # 3. Chunk
            chunks = chunk_text(transcript)

            # 4. Summaries
            summaries = []
            for i, ch in enumerate(chunks):
                try:
                    s = summarize_chunk(ch)
                except Exception as e:
                    print(f"Summary error (chunk {i}):", e)
                    s = ch[:400]
                summaries.append({"chunk": ch, "summary": s})

            # 4b. Auto-categorize if no topic provided
            video_topic = topic
            original_title = video_metadata.get("title", "")
            first_summary = summaries[0]["summary"] if summaries else ""

            if not video_topic and summaries:
                video_topic = auto_categorize(first_summary, original_title, existing_categories)
                print(f"Auto-categorized as: {video_topic}")

            # 4c. Extract key takeaway
            key_takeaway = ""
            if summaries:
                key_takeaway = extract_key_takeaway(transcript, first_summary, original_title)
                print(f"Key takeaway: {key_takeaway}")

            # 4d. Generate descriptive title from content
            video_title = original_title
            if summaries:
                generated_title = generate_title(transcript, first_summary, key_takeaway)
                if generated_title:
                    video_title = generated_title
                    print(f"Generated title: {video_title}")

            # 5. Prepare docs with enhanced metadata
            docs, metas = [], []
            timestamp = int(time.time())

            # Store categories as comma-separated string (Pinecone doesn't support arrays well for filtering)
            # Primary topic is the first one
            categories_str = video_topic if video_topic else "Uncategorized"

            for i, item in enumerate(summaries):
                # Include title, key_takeaway, and summary for better semantic matching
                merged = f"Title: {video_title}\nKey Takeaway: {key_takeaway}\n\nContent:\n{item['chunk']}\n\nSummary:\n{item['summary']}"
                docs.append(merged)
                metas.append({
                    "source": url,
                    "summary": item["summary"],
                    "chunk_index": i,
                    "title": video_title,
                    "author": video_metadata.get("author", ""),
                    "upload_date": video_metadata.get("upload_date", ""),
                    "duration": video_metadata.get("duration", 0),
                    "view_count": video_metadata.get("view_count", 0),
                    "topic": video_topic,
                    "categories": categories_str,  # Comma-separated list for multiple categories
                    "key_takeaway": key_takeaway,
                    "transcript": transcript[:3000],  # Store first 3000 chars of transcript
                    "ingested_at": timestamp,
                    "user_id": user_id if user_id else 0,  # 0 = shared/public content
                })

            # 6. Embeddings
            embeddings = embed_text(docs)

            # 7. Upsert
            vectors = [
                {
                    "id": f"{timestamp}_{i}",
                    "values": embeddings[i],
                    "metadata": metas[i]
                }
                for i in range(len(embeddings))
            ]

            upsert_to_pinecone(vectors)
            print(f"Upserted {len(vectors)} vectors for {url}")
            if user_id:
                simple_auth.log_usage(user_id, "ingest_success", details=f"Successfully ingested {url}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TikTok RAG Ingestion")
    parser.add_argument("urls", nargs="*", help="TikTok URLs to process")
    args = parser.parse_args()

    urls = args.urls
    if not urls:
        print("No URLs provided via CLI. Checking clipboard...")
        try:
            urls = get_urls_from_clipboard()
        except Exception as e:
            print(f"Clipboard error: {e}")
            urls = []

    if not urls:
        print("No TikTok URLs found.")
    else:
        process_urls(urls)
        print("Done.")