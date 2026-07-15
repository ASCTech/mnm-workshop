"""Retrieve a small, capped sample of the PMC Open Access subset restricted to
**reusable licenses** (CC0 / CC BY) — the analog for the ``article_coding``
archetype (full-text scholarly articles → multi-model codebook coding +
inter-model agreement).

Why this source: the PMC OA subset is the closest redistributable stand-in for a
corpus of paywalled research articles. Each article names a license; we keep only
the ones that permit reuse/redistribution (default: CC0 and CC BY).

How it works (polite + durable):
  1. Enumerate records via the PMC OA Web Service (``oa.fcgi``) over one-day
     windows — this is where the per-article ``license`` lives.
  2. Keep records whose license is in --licenses; stop once we have --limit.
  3. Download the full text from the **AWS Open Data mirror** (bucket
     ``pmc-oa-opendata``), which serves individual articles over plain HTTPS with
     no login. We list the ``PMC<id>.`` prefix to find the latest version, then
     fetch the requested formats (default: .txt full text + .xml JATS).

Note: NLM is removing the legacy FTP ``oa_package``/``oa_comm`` tree in August
2026; this uses the current version-based AWS layout instead. License filtering
still comes from oa.fcgi, since the new S3 keys no longer encode the use-group.

Docs: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/  and  /tools/oa-service/
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

from common import PoliteSession, log, write_manifest

OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
OUT_DIR = Path(__file__).resolve().parent / "data" / "pmc_oa"

DEFAULT_LIMIT = 20
DEFAULT_LICENSES = ("CC0", "CC BY")
DEFAULT_FORMATS = ("txt", "xml")  # small + LLM-friendly; add "pdf" to mirror "PDFs in"
# oa.fcgi date windows filter by when an article entered the OA subset, not by
# publication date; any recent span yields plenty of CC BY articles.
DEFAULT_START = "2023-01-01"
MAX_WINDOWS = 60  # safety cap on how many days we'll scan for a small sample

_KEY_RE = None  # compiled per-PMCID in latest_version()


def enumerate_records(session: PoliteSession, licenses: set[str], start: date):
    """Yield (pmcid, license) for matching OA records, one date window at a time."""
    day = start
    for _ in range(MAX_WINDOWS):
        params = {
            "from": day.isoformat(),
            "until": (day + timedelta(days=1)).isoformat(),
            "format": "tgz",
        }
        root = ET.fromstring(session.get_text(OA_SERVICE, params=params))
        err = root.find("error")
        if err is not None:
            log(f"  oa.fcgi note for {day}: {err.text}")
        for rec in root.findall("records/record"):
            if rec.get("license", "none") in licenses:
                yield rec.get("id"), rec.get("license")
        day += timedelta(days=1)


def latest_version(session: PoliteSession, pmcid: str) -> str | None:
    """List the S3 ``PMC<id>.`` prefix and return the highest version (e.g. '2')."""
    xml = session.get_text(
        S3_BASE, params={"list-type": "2", "prefix": f"{pmcid}.", "max-keys": "200"}
    )
    versions = {int(m) for m in re.findall(rf"<Key>{pmcid}\.(\d+)/", xml)}
    return str(max(versions)) if versions else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max articles to fetch (default {DEFAULT_LIMIT})")
    ap.add_argument("--licenses", nargs="+", default=list(DEFAULT_LICENSES),
                    help="accepted license strings, e.g. 'CC0' 'CC BY' 'CC BY-SA'")
    ap.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS),
                    help="file formats to download per article (txt, xml, pdf)")
    ap.add_argument("--start", default=DEFAULT_START,
                    help=f"first OA-service date window (default {DEFAULT_START})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rate", type=float, default=1.0,
                    help="min seconds between requests to a host (default 1.0)")
    args = ap.parse_args()

    licenses = set(args.licenses)
    session = PoliteSession(min_interval=args.rate)
    log(f"Fetching up to {args.limit} PMC OA articles; licenses={sorted(licenses)} "
        f"formats={args.formats}")

    rows: list[dict] = []
    for pmcid, lic in enumerate_records(session, licenses, date.fromisoformat(args.start)):
        ver = latest_version(session, pmcid)
        if ver is None:
            log(f"  {pmcid}: not on AWS OA mirror yet, skipping")
            continue
        stem = f"{pmcid}.{ver}"
        got_files: dict[str, str] = {}
        for fmt in args.formats:
            url = f"{S3_BASE}/{stem}/{stem}.{fmt}"
            dest = args.out / "articles" / pmcid / f"{stem}.{fmt}"
            try:
                session.download(url, dest)
            except Exception as e:  # a missing format shouldn't sink the article
                log(f"    {stem}.{fmt}: {e}")
                continue
            got_files[fmt] = str(dest)
        if not got_files:
            continue
        rows.append({"pmcid": pmcid, "license": lic, "version": ver,
                     "files": got_files, "s3_prefix": f"{S3_BASE}/{stem}/"})
        log(f"  [{len(rows)}/{args.limit}] {pmcid} v{ver}  {lic}  "
            f"{'+'.join(got_files)}")
        if len(rows) >= args.limit:
            break

    n = write_manifest(args.out / "manifest.jsonl", rows)
    log(f"Done. {n} articles under {args.out}  (manifest.jsonl)")
    if n < args.limit:
        log(f"  note: only {n} found within {MAX_WINDOWS} day-windows from {args.start}; "
            f"widen --start or --licenses to get more.")


if __name__ == "__main__":
    main()
