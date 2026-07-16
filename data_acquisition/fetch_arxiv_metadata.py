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
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import PoliteSession, log, write_manifest

API = "https://export.arxiv.org/api/query"
OUT_DIR = Path(__file__).resolve().parent / "data" / "arxiv"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

DEFAULT_LIMIT = 50
# A deliberately DIVERSE default spanning distinct fields (not just CS/ML), so the
# ground-truth category label has real spread to validate clusters against. We
# fetch a balanced quota PER category (see harvest) rather than one merged query,
# which otherwise returns a corpus dominated by the highest-volume categories and
# leaves the others with too few papers to be meaningful labels.
DEFAULT_CATEGORIES = (
    "cs.LG", "cs.CL", "cs.CV", "astro-ph.GA", "math.PR",
    "q-bio.NC", "econ.EM", "eess.AS", "stat.ME", "cond-mat.stat-mech",
)
PAGE_SIZE = 100  # arXiv allows up to a few hundred; keep per-category paging brisk
OVERSAMPLE = 3   # fetch this-times the per-category quota, then sample down


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


def fetch_category(session: PoliteSession, category: str, want: int) -> list[dict]:
    """Fetch up to `want` recent papers whose PRIMARY category is `category`.

    Filtering on the primary category (rather than any cross-listed category)
    keeps the ground-truth label clean — a cs.LG paper cross-listed to stat.ML
    counts once, under cs.LG.
    """
    rows: list[dict] = []
    start = 0
    while len(rows) < want:
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": min(PAGE_SIZE, want - len(rows)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        xml = session.get_text(API, params=params)
        raw = parse_entries(xml)
        if not raw:
            break
        rows.extend(e for e in raw if e["primary_category"] == category)
        start += len(raw)
        time.sleep(0.5)  # arXiv results are eventually-consistent; let paging settle
    return rows[:want]


def harvest(session: PoliteSession, categories, limit: int, seed: int) -> list[dict]:
    """Balanced per-category harvest: ~limit/len(categories) papers each.

    We oversample each category and randomly (seeded) draw the quota, so the
    sample isn't always the identical newest-N and spreads across a wider window.
    """
    per_cat = max(1, limit // len(categories))
    rng = random.Random(seed)
    rows: list[dict] = []
    for cat in categories:
        pool = fetch_category(session, cat, per_cat * OVERSAMPLE)
        rng.shuffle(pool)
        picked = pool[:per_cat]
        rows.extend(picked)
        log(f"  {cat}: pool={len(pool)} -> kept {len(picked)}  (running {len(rows)})")
    rng.shuffle(rows)  # interleave categories so downstream order isn't grouped
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
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the per-category random draw (reproducible)")
    args = ap.parse_args()

    session = PoliteSession(min_interval=args.rate)
    log(f"Fetching ~{args.limit} arXiv records, balanced across {len(args.categories)} "
        f"categories: {args.categories}")
    rows = harvest(session, args.categories, args.limit, args.seed)
    n = write_manifest(args.out / "metadata.jsonl", rows)
    log(f"Done. {n} records under {args.out}  (metadata.jsonl)")


if __name__ == "__main__":
    main()
