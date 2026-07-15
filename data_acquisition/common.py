"""Shared, polite HTTP helpers for the data-acquisition fetchers.

"Good manners" here means, concretely:

- an honest, identifying User-Agent with a contact address (set ``CONTACT_EMAIL``
  in the repo-root ``.env`` — falls back to a generic string otherwise);
- a per-host minimum interval between requests (simple client-side rate limit),
  so we never hammer a public endpoint;
- automatic retry with exponential backoff that honours ``Retry-After`` on
  429/5xx responses (via urllib3's ``Retry``);
- streaming downloads that skip files already on disk and write atomically, so a
  re-run resumes instead of re-downloading.

Every fetcher shares one :class:`PoliteSession` so these limits are enforced in
one place.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# _CONTACT = os.environ.get("CONTACT_EMAIL", "unset-contact@example.org")
# USER_AGENT = (
#     f"mnms-data-acquisition/0.1 (workshop demo; +https://github.com/; "
#     f"contact: {_CONTACT})"
# )

USER_AGENT = (
    f"mnms-data-acquisition/0.1 (workshop demo)"
)

class PoliteSession:
    """A ``requests`` session that self-throttles per host and retries nicely."""

    def __init__(
        self,
        *,
        min_interval: float = 1.0,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
        timeout: float = 60.0,
    ) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_hit: dict[str, float] = {}

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": USER_AGENT})
        self._s.mount("https://", adapter)
        self._s.mount("http://", adapter)

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        now = time.monotonic()
        wait = self.min_interval - (now - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def get(self, url: str, **kw: Any) -> requests.Response:
        self._throttle(url)
        kw.setdefault("timeout", self.timeout)
        resp = self._s.get(url, **kw)
        resp.raise_for_status()
        return resp

    def get_json(self, url: str, **kw: Any) -> Any:
        return self.get(url, **kw).json()

    def get_text(self, url: str, **kw: Any) -> str:
        return self.get(url, **kw).text

    def download(self, url: str, dest: Path, *, skip_existing: bool = True) -> bool:
        """Stream ``url`` to ``dest``. Returns True if downloaded, False if skipped."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            return False
        self._throttle(url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self._s.get(url, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(dest)
        return True


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write ``rows`` as JSON Lines; returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def log(msg: str) -> None:
    print(msg, flush=True)
