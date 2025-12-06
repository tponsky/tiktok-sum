#!/usr/bin/env python3
"""
Test script to debug category deletion issue
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "tiktok-rag"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def test_category_search(category_name: str):
    """Test searching for videos with a specific category."""
    print(f"\n{'='*60}")
    print(f"Testing category: '{category_name}'")
    print(f"{'='*60}")

    dummy_vector = [0.0] * 1536

    # Get first 20 videos
    results = index.query(
        vector=dummy_vector,
        top_k=20,
        include_metadata=True,
        filter={"chunk_index": {"$eq": 0}}
    )

    print(f"\nRetrieved {len(results.matches)} videos")
    print("\nAll categories found:")

    all_categories = set()
    videos_with_target = []

    for i, match in enumerate(results.matches):
        meta = match.metadata or {}
        cats = meta.get("categories", "")
        title = meta.get("title", "Untitled")
        source = meta.get("source", "")

        # Parse categories
        cats_list = [c.strip() for c in cats.split(",") if c.strip()] if cats else []

        for cat in cats_list:
            all_categories.add(cat)

        # Check if this video has the target category
        if category_name in cats_list:
            videos_with_target.append({
                "id": match.id,
                "title": title,
                "source": source,
                "categories": cats_list
            })

        # Print first 5 videos in detail
        if i < 5:
            print(f"\nVideo {i+1}:")
            print(f"  ID: {match.id}")
            print(f"  Title: {title}")
            print(f"  Raw categories: '{cats}' (repr: {repr(cats)})")
            print(f"  Parsed categories: {cats_list}")

    print(f"\nAll unique categories found: {sorted(all_categories)}")
    print(f"\nVideos with '{category_name}': {len(videos_with_target)}")

    if videos_with_target:
        print("\nDetails of matching videos:")
        for v in videos_with_target:
            print(f"  - {v['title']}")
            print(f"    Categories: {v['categories']}")
            print(f"    Source: {v['source']}")
    else:
        print(f"\nNo videos found with category '{category_name}'")
        print("This might be why deletion reports '0 videos reassigned'!")

        # Check for similar categories
        print("\nChecking for similar categories...")
        for cat in sorted(all_categories):
            if category_name.lower() in cat.lower() or cat.lower() in category_name.lower():
                print(f"  Similar: '{cat}'")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        category = " ".join(sys.argv[1:])
    else:
        category = "Technology & Gadgets"

    test_category_search(category)

    # Also test some other common categories
    print("\n\nTesting a few other categories for comparison:")
    for test_cat in ["Uncategorized", "Technology", "Gadgets"]:
        print(f"\n--- Testing '{test_cat}' ---")
        test_category_search(test_cat)
