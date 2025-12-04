"""
tiktok_rag_cloud.py

Cloud pipeline:
- Reads TikTok links from clipboard
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
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

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

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

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
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        # Add headers to bypass TikTok blocking
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.tiktok.com/",
        },
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
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return resp.text


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Chunk text with overlap to preserve context."""
    sents = sent_tokenize(text)
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


def embed_text(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in resp.data]


def upsert_to_pinecone(vectors: List[dict]):
    index.upsert(vectors)


# -------- Main Pipeline -------- #

def process_urls(urls: List[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        for url in urls:
            print("Processing:", url)

            # 1. Download audio and get metadata
            try:
                wav, video_metadata = download_audio(url, tmpdir)
            except Exception as e:
                print("Download error:", e)
                continue

            # 2. Transcribe
            try:
                transcript = transcribe_with_openai(wav)
            except Exception as e:
                print("Transcription error:", e)
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

            # 5. Prepare docs with enhanced metadata
            docs, metas = [], []
            timestamp = int(time.time())

            for i, item in enumerate(summaries):
                merged = item["chunk"] + "\n\nSummary:\n" + item["summary"]
                docs.append(merged)
                metas.append({
                    "source": url,
                    "summary": item["summary"],
                    "chunk_index": i,
                    "title": video_metadata.get("title", ""),
                    "author": video_metadata.get("author", ""),
                    "upload_date": video_metadata.get("upload_date", ""),
                    "duration": video_metadata.get("duration", 0),
                    "view_count": video_metadata.get("view_count", 0),
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