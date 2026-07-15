"""Retrieve a small, capped sample of NSF award abstracts — the second analog for
the ``syllabi`` archetype (institutional documents → few-field extraction → embed
→ topic clustering). NSF award records are U.S. government works and therefore
**public domain**; each abstract reads like a small structured document with a
title, program, directorate and dates you can extract and then cluster into a
research-topic map.

This uses the Research.gov awards API (``api.nsf.gov``), which returns JSON, 25
records per page, and lets us request exactly the fields we want.

Scale-up: for the full corpus, grab NSF's per-year bulk downloads (one XML file
per award) from https://www.nsf.gov/awardsearch/download.jsp

Docs: https://www.research.gov/common/webapi/awardapisearch-v1.htm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PoliteSession, log, write_manifest

API = "https://api.nsf.gov/services/v1/awards.json"
OUT_DIR = Path(__file__).resolve().parent / "data" / "nsf_awards"

DEFAULT_LIMIT = 50
DEFAULT_KEYWORD = "machine learning"
PAGE_SIZE = 25  # the API's fixed page size
FIELDS = (
    "id,title,abstractText,fundProgramName,primaryProgram,"
    "startDate,expDate,awardeeName,awardeeStateCode"
)


def harvest(session: PoliteSession, keyword: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    offset = 1  # NSF offsets are 1-based
    while len(rows) < limit:
        params = {"keyword": keyword, "printFields": FIELDS, "offset": offset}
        data = session.get_json(API, params=params)
        resp = data.get("response", {})
        if "serviceNotification" in resp:
            log(f"  API notice: {resp['serviceNotification']}")
            break
        batch = resp.get("award", [])
        if not batch:
            log("  no more awards returned")
            break
        rows.extend(batch)
        log(f"  fetched {len(rows)}/{limit}")
        offset += len(batch)
        if len(batch) < PAGE_SIZE:
            break
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max awards to fetch (default {DEFAULT_LIMIT})")
    ap.add_argument("--keyword", default=DEFAULT_KEYWORD,
                    help=f"search keyword (default {DEFAULT_KEYWORD!r})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rate", type=float, default=1.0,
                    help="min seconds between requests (default 1.0)")
    args = ap.parse_args()

    session = PoliteSession(min_interval=args.rate)
    log(f"Fetching up to {args.limit} NSF awards matching {args.keyword!r}")
    rows = harvest(session, args.keyword, args.limit)
    # Drop awards with an empty abstract — they're not useful for the demo.
    kept = [r for r in rows if (r.get("abstractText") or "").strip()]
    if len(kept) < len(rows):
        log(f"  dropped {len(rows) - len(kept)} awards with empty abstracts")
    n = write_manifest(args.out / "awards.jsonl", kept)
    log(f"Done. {n} awards under {args.out}  (awards.jsonl)")


if __name__ == "__main__":
    main()
