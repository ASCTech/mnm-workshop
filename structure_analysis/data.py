"""Corpus loading: arXiv abstracts (primary) and NSF award abstracts (secondary).

Both sources are read from the git-ignored `data_acquisition/data/<source>/`
trees produced by the fetchers in `../data_acquisition`. Each row is
normalized to `{doc_id, title, text, label_full, label_top}` where
`label_full` / `label_top` are the ground-truth categories used for stage-3
validation (arXiv `primary_category`, collapsed to its top-level part before
the first '.' for `label_top`; NSF `fundProgramName`, which has no natural
top-level collapse so `label_top == label_full`).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARXIV_PATH = REPO_ROOT / "data_acquisition" / "data" / "arxiv" / "metadata.jsonl"
NSF_PATH = REPO_ROOT / "data_acquisition" / "data" / "nsf_awards" / "awards.jsonl"


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the matching data_acquisition fetcher first "
            f"(see structure_analysis/README.md)."
        )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_arxiv(limit: int | None) -> list[dict]:
    rows = _read_jsonl(ARXIV_PATH, limit)
    out = []
    for r in rows:
        label_full = r.get("primary_category") or "unknown"
        label_top = label_full.split(".")[0] if label_full else "unknown"
        out.append(
            {
                "doc_id": r["arxiv_id"],
                "title": r.get("title", ""),
                "text": f"{r.get('title', '')}\n\n{r.get('abstract', '')}".strip(),
                "label_full": label_full,
                "label_top": label_top,
            }
        )
    return out


def load_nsf(limit: int | None) -> list[dict]:
    rows = _read_jsonl(NSF_PATH, limit)
    out = []
    for r in rows:
        label_full = r.get("fundProgramName") or "unknown"
        out.append(
            {
                "doc_id": str(r["id"]),
                "title": r.get("title", ""),
                "text": f"{r.get('title', '')}\n\n{r.get('abstractText', '')}".strip(),
                "label_full": label_full,
                "label_top": label_full,  # no natural top-level collapse for NSF programs
            }
        )
    return out


def load_corpus(source: str, limit: int | None) -> list[dict]:
    if source == "arxiv":
        return load_arxiv(limit)
    if source == "nsf":
        return load_nsf(limit)
    raise ValueError(f"Unknown source: {source!r} (expected 'arxiv' or 'nsf')")
