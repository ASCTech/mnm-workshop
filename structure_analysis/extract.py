"""Stage 1: LLM extraction of structured fields from each abstract.

Resumable: `extractions.jsonl` is the checkpoint. A re-run reads it first and
skips any `doc_id` already recorded (success *or* failure), unless
`--retry-failed` is passed, in which case failed rows are retried. Per-item
resilient: a bad doc/response never aborts the batch — it's recorded with an
`error` field and the run continues. One retry is attempted on a JSON-parse
or schema-validation failure before giving up on an item.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI
from pydantic import ValidationError
from tqdm import tqdm

from llm import CostLedger, get_client
from schema import AbstractFields, SYSTEM_PROMPT


def _strip_json_fences(content: str) -> str:
    s = content.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    end = len(lines) - 1
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].strip() == "```":
            end = i
            break
    return "\n".join(lines[1:end])


def load_existing(path: Path) -> dict[str, dict]:
    """doc_id -> last-seen record, from a prior run's extractions.jsonl."""
    existing: dict[str, dict] = {}
    if not path.exists():
        return existing
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[rec["doc_id"]] = rec
    return existing


def _call_once(client: OpenAI, model: str, text: str) -> tuple[dict | None, str, int, int, str | None]:
    """One chat call + parse/validate attempt. Returns (parsed, raw, in_tok, out_tok, error)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Abstract:\n{text[:4000]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=600,
        timeout=120.0,
    )
    raw = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = usage.prompt_tokens if usage else 0
    out_tok = usage.completion_tokens if usage else 0
    try:
        obj = json.loads(_strip_json_fences(raw))
        validated = AbstractFields.model_validate(obj)
        return validated.model_dump(), raw, in_tok, out_tok, None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, raw, in_tok, out_tok, f"{type(e).__name__}: {e}"


def extract_one(client: OpenAI, model: str, doc: dict, ledger: CostLedger) -> dict:
    """Extract fields for one doc, with one retry on parse/validation failure."""
    doc_id = doc["doc_id"]
    try:
        parsed, raw, in_tok, out_tok, error = _call_once(client, model, doc["text"])
        cost = ledger.add(model, in_tok, out_tok)
        if error is not None:
            # One retry
            parsed2, raw2, in_tok2, out_tok2, error2 = _call_once(client, model, doc["text"])
            cost += ledger.add(model, in_tok2, out_tok2)
            in_tok, out_tok = in_tok + in_tok2, out_tok + out_tok2
            if error2 is None:
                parsed, error = parsed2, None
            else:
                error = f"retry also failed: {error2}"
        return {
            "doc_id": doc_id,
            "model": model,
            "parsed": parsed,
            "error": error,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
        }
    except Exception as e:  # network errors, rate limits, etc. — never abort the batch
        return {
            "doc_id": doc_id,
            "model": model,
            "parsed": None,
            "error": f"{type(e).__name__}: {e}",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }


def run_extraction(
    docs: list[dict],
    model: str,
    out_path: Path,
    ledger: CostLedger,
    max_workers: int = 8,
    retry_failed: bool = False,
) -> dict:
    """Extract fields for every doc, appending to `out_path`. Returns a summary dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)

    todo = []
    n_cached_ok, n_cached_err = 0, 0
    for doc in docs:
        rec = existing.get(doc["doc_id"])
        if rec is None:
            todo.append(doc)
        elif rec.get("error") is not None and retry_failed:
            todo.append(doc)
        elif rec.get("error") is not None:
            n_cached_err += 1
        else:
            n_cached_ok += 1

    print(
        f"Extraction: {len(docs)} docs total, {n_cached_ok} cached ok, "
        f"{n_cached_err} cached errors (skipped), {len(todo)} to do"
    )

    if not todo:
        return {"n_total": len(docs), "n_new": 0, "n_cached_ok": n_cached_ok, "n_cached_err": n_cached_err}

    client = get_client()
    write_lock = Lock()
    n_ok, n_err = 0, 0
    start = time.time()
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_one, client, model, doc, ledger): doc for doc in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            rec = fut.result()
            with write_lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if rec["error"] is None:
                n_ok += 1
            else:
                n_err += 1
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s — {n_ok} ok, {n_err} failed (this run)")
    return {
        "n_total": len(docs),
        "n_new": len(todo),
        "n_new_ok": n_ok,
        "n_new_err": n_err,
        "n_cached_ok": n_cached_ok,
        "n_cached_err": n_cached_err,
    }


def load_extractions(path: Path) -> dict[str, dict]:
    """doc_id -> record, successful rows only carry a non-None `parsed`."""
    return load_existing(path)
