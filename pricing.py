"""Single source of truth for pricing.

Both rag_api.py and tiktok_rag_cloud.py import from here so a price change
can never drift between the charge path and the refund path.
"""
import os

MARKUP_MULTIPLIER = float(os.getenv("MARKUP_MULTIPLIER", "5.0"))

# Approximate real API cost per operation
BASE_COST_SEARCH = float(os.getenv("BASE_COST_SEARCH", "0.001"))
BASE_COST_INGEST = float(os.getenv("BASE_COST_INGEST", "0.006"))

COST_PER_SEARCH = BASE_COST_SEARCH * MARKUP_MULTIPLIER   # $0.005
COST_PER_INGEST = BASE_COST_INGEST * MARKUP_MULTIPLIER   # $0.03

INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "2.00"))
RELOAD_AMOUNT = float(os.getenv("RELOAD_AMOUNT", "10.00"))
RELOAD_THRESHOLD = float(os.getenv("RELOAD_THRESHOLD", "1.00"))
