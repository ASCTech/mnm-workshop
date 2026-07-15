"""Retrieve a small, capped sample of arXiv paper metadata — one analog for the
``syllabi`` archetype (semi-structured docs → few-field extraction → embed →
topic clustering). Abstracts stand in for syllabus text; the arXiv category is a
ready-made ground-truth label to validate clusters against.

This uses the public arXiv Atom API, which is the polite programmatic route for a
*sample*. arXiv asks for no more than one request every 3 seconds from a single
IP, so the per-host interval defaults to 3s here.

Scale-up: the full, machine-readable arXiv metadata dump is on Kaggle under
**CC0 / public domain** (Cornell-University/arxiv, 1.7M+ papers) — use that for
the real corpus rather than paging the API.

Docs: https://info.arxiv.org/help/api/ and https://info.arxiv.org/help/bulk_data.html
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import PoliteSession, log, write_manifest

API = "https://export.arxiv.org/api/query"
OUT_DIR = Path(__file__).resolve().parent / "data" / "arxiv"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

DEFAULT_LIMIT = 50
DEFAULT_CATEGORIES = ("cs.LG", "cs.CL", "stat.ML")
PAGE_SIZE = 25  # keep individual API responses small


def parse_entries(xml: str) -> list[dict]:
    root = ET.fromstring(xml)
    out: list[dict] = []
    for e in root.findall("a:entry", NS):
        arxiv_id = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
        out.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(e.findtext("a:title", "", NS).split()),
            "published": e.findtext("a:published", "", NS),
            "updated": e.findtext("a:updated", "", NS),
            "primary_category": (
                e.find("arxiv:primary_category", NS).get("term")
                if e.find("arxiv:primary_category", NS) is not None else ""
            ),
            "categories": [c.get("term") for c in e.findall("a:category", NS)],
            "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
            "abstract": " ".join(e.findtext("a:summary", "", NS).split()),
        })
    return out


def harvest(session: PoliteSession, categories, limit: int) -> list[dict]:
    query = " OR ".join(f"cat:{c}" for c in categories)
    rows: list[dict] = []
    start = 0
    while len(rows) < limit:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(PAGE_SIZE, limit - len(rows)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        xml = session.get_text(API, params=params)
        batch = parse_entries(xml)
        if not batch:
            log("  no more results returned by the API")
            break
        rows.extend(batch)
        log(f"  fetched {len(rows)}/{limit}")
        start += len(batch)
        # arXiv API results are eventually-consistent; a brief settle helps paging.
        time.sleep(0.5)
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max papers to fetch (default {DEFAULT_LIMIT})")
    ap.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES),
                    help="arXiv categories to sample from")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rate", type=float, default=3.0,
                    help="min seconds between API requests (arXiv asks for >=3)")
    args = ap.parse_args()

    session = PoliteSession(min_interval=args.rate)
    log(f"Fetching up to {args.limit} arXiv records from {args.categories}")
    rows = harvest(session, args.categories, args.limit)
    n = write_manifest(args.out / "metadata.jsonl", rows)
    log(f"Done. {n} records under {args.out}  (metadata.jsonl)")


if __name__ == "__main__":
    main()
