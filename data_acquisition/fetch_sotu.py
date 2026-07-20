"""Retrieve **State of the Union addresses as plain text** from The American
Presidency Project (presidency.ucsb.edu), the text analogue that replaces the
audio/ASR path for the ``bt_scoring`` branch.

Why this source: SOTU addresses are US-government works and therefore public
domain (17 U.S.C. § 105), so they are freely redistributable — the point of the
workshop's data layer. They are labelled by president (and, via a small local
lookup, party), and they span decades, giving both a per-president grouping and a
"data over time" axis. Being text, they feed the pairwise LLM-as-judge →
Bradley-Terry pipeline directly, with no lossy ASR step in the middle.

Flow: text in here → pairwise LLM-as-judge on some dimension (e.g. economic
left/right) → Bradley-Terry scale, validated against the party label the judge
never saw (``bt_scoring``).

Politeness: presidency.ucsb.edu's robots.txt blocks named AI crawlers
(ClaudeBot/GPTBot/…) but allows ``/documents/`` for a generic User-Agent, with a
``Crawl-delay: 10``. Our honest ``mnms-data-acquisition`` UA is compliant; the
default ``--rate 10`` honours that crawl-delay. The whole 1950–2020 corpus
(~2 listing pages + ~65 documents) fetches in ~11 minutes.

Site structure (Drupal 7, fully server-rendered — plain requests + BeautifulSoup,
no JavaScript): the SOTU "spoken addresses" category page enumerates every
address; each document page carries the speech text, president, date, and title
in stable CSS containers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bs4 import BeautifulSoup

from common import PoliteSession, log, write_manifest

BASE = "https://www.presidency.ucsb.edu"
LISTING = (
    BASE + "/documents/app-categories/spoken-addresses-and-remarks/"
    "presidential/state-the-union-addresses"
)
OUT_DIR = Path(__file__).resolve().parent / "data" / "sotu"

DEFAULT_LIMIT = 0        # 0 = every address in the year window
DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2020
PER_PAGE = 60            # listing supports items_per_page in {5,10,20,40,60}
MAX_PAGES = 12           # safety cap while paginating the listing

# APP labels each address with the president's full name. Map that to a short,
# filesystem-safe key (used in the identifier) and a party (the ground-truth
# axis bt_scoring validates the emergent scale against). Ordered so the more
# specific "George W. Bush" is matched before the bare "Bush" (= George H. W.
# Bush, whom APP lists simply as "George Bush").
_PRESIDENTS: list[tuple[str, str, str]] = [
    ("George W. Bush", "bush_w", "Republican"),
    ("Bush", "bush_hw", "Republican"),
    ("Truman", "truman", "Democrat"),
    ("Eisenhower", "eisenhower", "Republican"),
    ("Kennedy", "kennedy", "Democrat"),
    ("Johnson", "johnson", "Democrat"),
    ("Nixon", "nixon", "Republican"),
    ("Ford", "ford", "Republican"),
    ("Carter", "carter", "Democrat"),
    ("Reagan", "reagan", "Republican"),
    ("Clinton", "clinton", "Democrat"),
    ("Obama", "obama", "Democrat"),
    ("Trump", "trump", "Republican"),
]


def classify_president(name: str) -> tuple[str, str] | None:
    """(short_key, party) for a president's display name, or None if unknown."""
    n = " ".join((name or "").split())
    for needle, key, party in _PRESIDENTS:
        if needle in n:
            return key, party
    return None


def _year_of(iso_date: str) -> int | None:
    """Year from an ISO date string like '2006-01-31T21:12:00+00:00'."""
    if iso_date and len(iso_date) >= 4 and iso_date[:4].isdigit():
        return int(iso_date[:4])
    return None


def enumerate_listing(session: PoliteSession) -> list[dict]:
    """Walk the SOTU category pages, returning one record per address with the
    doc path, ISO date, president name, and title — all present in the listing
    HTML, so the per-document fetch is only needed for the speech text itself."""
    records: list[dict] = []
    seen: set[str] = set()
    for page in range(MAX_PAGES):
        url = f"{LISTING}?items_per_page={PER_PAGE}&page={page}"
        html = session.get_text(url)
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("div.views-row")
        if not rows:
            break
        page_new = 0
        for row in rows:
            node = row.select_one("div[about^='/documents/']")
            date_span = row.select_one("span.date-display-single")
            if node is None or date_span is None:
                continue
            path = node["about"]
            if path in seen:
                continue
            seen.add(path)
            page_new += 1
            pres_link = row.select_one("div.col-sm-4 a")
            title_link = row.select_one("div.field-title a")
            records.append({
                "path": path,
                "iso_date": date_span.get("content", ""),
                "president": pres_link.get_text(strip=True) if pres_link else "",
                "title": title_link.get_text(strip=True) if title_link else "",
            })
        if page_new == 0:
            break
    return records


def extract_text(session: PoliteSession, url: str) -> str:
    """Fetch a document page and pull the speech body out of field-docs-content."""
    soup = BeautifulSoup(session.get_text(url), "html.parser")
    content = soup.select_one("div.field-docs-content")
    if content is None:
        return ""
    paras = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    paras = [p for p in paras if p]
    if paras:
        return "\n\n".join(paras)
    return content.get_text("\n", strip=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="max addresses to fetch (0 = all in the year window)")
    ap.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                    help=f"earliest year to include (default {DEFAULT_START_YEAR})")
    ap.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR,
                    help=f"latest year to include (default {DEFAULT_END_YEAR})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rate", type=float, default=10.0,
                    help="min seconds between requests (default 10.0; honours the "
                         "site's Crawl-delay: 10)")
    args = ap.parse_args()

    session = PoliteSession(min_interval=args.rate)
    texts_dir = args.out / "texts"

    log(f"Enumerating SOTU spoken-address listing at {LISTING}")
    listing = enumerate_listing(session)

    # Keep only addresses in the year window with a president we can label.
    candidates: list[dict] = []
    for rec in listing:
        year = _year_of(rec["iso_date"])
        if year is None or not (args.start_year <= year <= args.end_year):
            continue
        cls = classify_president(rec["president"])
        if cls is None:
            log(f"  skip (unmapped president {rec['president']!r}): {rec['path']}")
            continue
        key, party = cls
        rec.update(year=year, key=key, party=party,
                   identifier=f"{key}_{year}")
        candidates.append(rec)

    candidates.sort(key=lambda r: r["iso_date"])  # chronological
    if args.limit > 0:
        candidates = candidates[: args.limit]
    log(f"  {len(listing)} addresses listed; {len(candidates)} in "
        f"{args.start_year}-{args.end_year} with a known president")

    rows: list[dict] = []
    for rec in candidates:
        ident = rec["identifier"]
        url = BASE + rec["path"]
        dest = texts_dir / f"{ident}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            text = dest.read_text(encoding="utf-8")
            fetched = False
        else:
            text = extract_text(session, url)
            if not text:
                log(f"  {ident}: no speech text found, skipping")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            fetched = True
        words = len(text.split())
        rows.append({
            "identifier": ident,
            "president": rec["president"],
            "party": rec["party"],
            "year": rec["year"],
            "date": rec["iso_date"][:10] or None,
            "title": rec["title"],
            "word_count": words,
            "text_path": str(dest),
            "source_url": url,
            "license": "public-domain-usgov",
        })
        log(f"  [{len(rows)}/{args.limit or len(candidates)}] {ident}  "
            f"{rec['party'][:3]}  {words} words  {'downloaded' if fetched else 'cached'}")

    n = write_manifest(args.out / "manifest.jsonl", rows)
    log(f"Done. {n} SOTU addresses under {args.out}  (manifest.jsonl)")
    log("  next: judge pairwise on a dimension (e.g. economic left/right) via "
        "../bt_scoring for a Bradley-Terry scale.")


if __name__ == "__main__":
    main()
