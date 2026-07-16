"""Retrieve a small, capped sample of public-domain **presidential audio** from
the Internet Archive ``presidential_recordings`` collection — the analog for the
``podcasts`` archetype, chosen for the user's top preference: it has *paired
audio*, so it plugs straight into the transcription/ASR branch, and it spans
decades (FDR → the tape-era presidencies), giving the "data over time" axis too.

Flow: audio in here → transcribe with the ``transcription`` branch → feed the
resulting text to a pairwise LLM-as-judge → Bradley-Terry ranking (the podcast
archetype's statistics). These recordings carry date metadata but **no
ground-truth transcript**, so this is an ASR *input* source, not a WER benchmark
like Common Voice.

Politeness: we list with one search call, then download exactly ONE small audio
file (lowest-bitrate MP3) per item, throttled per host.

Text-first alternative (if you'd rather skip audio): presidential-debate
transcripts from the American Presidency Project — also public domain, also
spanning 1960→present.

Docs: https://archive.org/developers/  (advancedsearch, metadata, download)
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from common import PoliteSession, log, write_manifest

SEARCH = "https://archive.org/advancedsearch.php"
METADATA = "https://archive.org/metadata/"
DOWNLOAD = "https://archive.org/download/"
OUT_DIR = Path(__file__).resolve().parent / "data" / "presidential_audio"

DEFAULT_LIMIT = 5  # audio is heavy; start very small
DEFAULT_COLLECTION = "presidential_recordings"
# Prefer the smallest usable single-file audio derivative.
AUDIO_FORMAT_PREFERENCE = ("64Kbps MP3", "VBR MP3", "Ogg Vorbis")

# Map an item's identifier prefix to a canonical speaker so we can stratify the
# sample by president instead of taking whatever the default sort returns (the
# collection is dominated by a few speakers/years, so a naive head-of-list grab
# comes back almost entirely one president in one year).
_SPEAKER_PREFIXES = (
    ("fdr", "fdr"), ("dde", "eisenhower"), ("jfk", "kennedy"), ("lbj", "johnson"),
    ("nixon", "nixon"), ("reagan", "reagan"), ("gwb", "gw_bush"),
    ("dictabelt", "dictabelt"),
)


def speaker_key(identifier: str) -> str:
    ident = (identifier or "").lower()
    for prefix, name in _SPEAKER_PREFIXES:
        if ident.startswith(prefix):
            return name
    m = re.match(r"[a-z]+", ident)
    return m.group(0) if m else ident


def list_items(session: PoliteSession, collection: str) -> list[dict]:
    """Fetch the FULL item list for the collection (it's small — ~100s of items)."""
    params = {
        "q": f"collection:{collection}",
        "fl[]": ["identifier", "title", "year", "date"],
        "rows": 5000,
        "page": 1,
        "sort[]": "date asc",
        "output": "json",
    }
    data = session.get_json(SEARCH, params=params)
    return data["response"]["docs"]


def select_diverse(docs: list[dict], limit: int, seed: int) -> list[dict]:
    """Speaker-stratified, time-spread selection.

    Group items by canonical speaker, order each speaker's items chronologically,
    then round-robin across speakers so the sample spreads over presidents *and*
    (within each) over time — rather than clustering on the most-represented
    speaker/year the way a plain sort does.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        groups[speaker_key(d.get("identifier", ""))].append(d)
    for items in groups.values():
        items.sort(key=lambda x: str(x.get("date") or x.get("year") or ""))

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)  # seeded: reproducible but not alphabetical
    cursors = {k: 0 for k in keys}
    selected: list[dict] = []
    while len(selected) < limit:
        progressed = False
        for k in keys:
            if cursors[k] < len(groups[k]):
                selected.append(groups[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:  # every speaker exhausted
            break
    return selected


def pick_audio(files: list[dict]) -> dict | None:
    """Choose one small, single-file audio derivative (no ZIP/M3U playlists)."""
    def is_single_audio(f: dict) -> bool:
        name = f.get("name", "").lower()
        return name.endswith((".mp3", ".ogg")) and "zip" not in f.get("format", "").lower()

    candidates = [f for f in files if is_single_audio(f)]
    if not candidates:
        return None
    # Rank by our format preference, then by smallest size.
    def rank(f: dict):
        fmt = f.get("format", "")
        pref = AUDIO_FORMAT_PREFERENCE.index(fmt) if fmt in AUDIO_FORMAT_PREFERENCE else 99
        return (pref, int(f.get("size", 1 << 62)))

    return min(candidates, key=rank)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"max audio items to fetch (default {DEFAULT_LIMIT})")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Internet Archive collection (default {DEFAULT_COLLECTION})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--rate", type=float, default=1.0,
                    help="min seconds between requests (default 1.0)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the speaker round-robin order (reproducible)")
    ap.add_argument("--no-stratify", action="store_true",
                    help="disable speaker stratification (take oldest-first, the old behaviour)")
    args = ap.parse_args()

    session = PoliteSession(min_interval=args.rate)
    log(f"Listing items from IA collection {args.collection!r}")
    all_items = list_items(session, args.collection)
    if args.no_stratify:
        items = all_items[: args.limit]
    else:
        items = select_diverse(all_items, args.limit, args.seed)
        spread = sorted({speaker_key(d.get("identifier", "")) for d in items})
        log(f"  {len(all_items)} items available; selected {len(items)} across "
            f"speakers: {', '.join(spread)}")

    rows: list[dict] = []
    for doc in items:
        ident = doc["identifier"]
        meta = session.get_json(METADATA + ident)
        audio = pick_audio(meta.get("files", []))
        if audio is None:
            log(f"  {ident}: no single-file audio derivative, skipping")
            continue
        fname = audio["name"]
        dest = args.out / ident / fname
        got = session.download(f"{DOWNLOAD}{ident}/{fname}", dest)
        rows.append({
            "identifier": ident,
            "title": doc.get("title"),
            "date": doc.get("date"),
            "year": doc.get("year"),
            "audio_format": audio.get("format"),
            "audio_size": int(audio.get("size", 0)),
            "audio_path": str(dest),
            "source_url": f"{DOWNLOAD}{ident}/{fname}",
        })
        mb = int(audio.get("size", 0)) / 1e6
        log(f"  [{len(rows)}/{args.limit}] {ident}  {doc.get('date','?')[:10]}  "
            f"{audio.get('format')}  {mb:.1f} MB  {'downloaded' if got else 'cached'}")
        if len(rows) >= args.limit:
            break

    n = write_manifest(args.out / "manifest.jsonl", rows)
    log(f"Done. {n} audio items under {args.out}  (manifest.jsonl)")
    log("  next: transcribe these via ../transcription, then judge pairwise for a "
        "Bradley-Terry scale.")


if __name__ == "__main__":
    main()
