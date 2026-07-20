# data_acquisition

Capped, polite retrieval of openly-licensed corpora for the workshop branches. These fetchers
give each branch redistributable data with a representative analytical shape, so the case
studies are available for anyone who clones the repo. The transcription branch already
runs on Common Voice CC0 but fetches separately.

Each source is a separate script. All caps start small; raise `--limit` when
you're ready.

| Script | Stands in for | Source | License | Analysis it enables |
|---|---|---|---|---|
| `fetch_pmc_oa.py` | `article_coding` | PMC Open Access subset | CC0 / CC BY (commercial-use group) | multi-model codebook coding + inter-model agreement over full-text articles |
| `fetch_arxiv_metadata.py` | `syllabi` | arXiv Atom API | CC0 (Kaggle dump) | field extraction → embed → topic clustering; category = ground-truth label |
| `fetch_nsf_awards.py` | `syllabi` | NSF awards API | Public domain (US gov) | field extraction → clustering of research abstracts |
| `fetch_sotu.py` | `podcasts` | American Presidency Project (SOTU) | Public domain (US gov) | pairwise LLM-judge on economic left/right → Bradley-Terry scale, validated vs. party |
| `fetch_presidential_audio.py` | `podcasts` (legacy) | Internet Archive `presidential_recordings` | Public domain (US gov) | former ASR path for `bt_scoring`; kept as an audio source (see note) |

## Usage

From the repo root (uv workspace member):

```bash
uv sync
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 20
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 50
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 50 --keyword "topic modeling"
uv run --package data_acquisition python data_acquisition/fetch_sotu.py            # all SOTU 1950–2020
uv run --package data_acquisition python data_acquisition/fetch_presidential_audio.py --limit 5
```

Outputs land in `data_acquisition/data/<source>/` (git-ignored). Each writes a
JSON-Lines manifest recording exactly what was fetched (ids, licenses, paths,
key metadata).

## Good manners (shared)

All scripts use `common.PoliteSession`, which:

- sends an honest User-Agent with an optional contact address — set `CONTACT_EMAIL` in
  the repo-root `.env` (falls back to a generic string);
- enforces a per-host minimum interval between requests (`--rate`, defaults
  chosen per source — e.g. arXiv defaults to 3 s per its API guidance);
- retries with exponential backoff and honors `Retry-After` on 429/5xx;
- streams downloads, skips files already present, and writes atomically, so
  re-runs resume instead of re-fetching.
- Many/most sites that openly host data are being hammered by bots, so being a good internet citizen matters.



## Per-source notes

### `fetch_pmc_oa.py` — PMC Open Access (article_coding)
Enumerates articles via the PMC OA Web Service (`oa.fcgi`) over one-day windows,
keeps only records whose per-article license is in `--licenses` (default `CC0`,
`CC BY`), then downloads the full text (`--formats`, default `.txt` + `.xml`)
from the **AWS Open Data mirror** (`pmc-oa-opendata`, HTTPS, no login). NLM is
removing the legacy FTP `oa_package`/`oa_comm` tree in August 2026, so this uses
the current version-based S3 layout (`PMC<id>.<ver>/…`). Scale-up: sync whole
prefixes from the same bucket (see the AWS Open Data registry entry `ncbi-pmc`).

### `fetch_arxiv_metadata.py` — arXiv metadata (syllabi)
Samples recent papers in `--categories` via the arXiv Atom API (title, abstract,
categories, dates). Abstracts play the role of syllabus text for extract-then-
cluster; the arXiv category gives you a label to score clusters against. Scale-up:
the full **CC0** metadata dump (Cornell-University/arxiv on Kaggle, 1.7M+ papers).

### `fetch_nsf_awards.py` — NSF award abstracts (syllabi)
Pages the Research.gov awards API for awards matching `--keyword`, keeping the
abstract + program/directorate/date fields. US-government works, so public
domain. Scale-up: NSF's per-year bulk XML at `awardsearch/download.jsp`.

### `fetch_sotu.py` — State of the Union addresses (podcasts)
Enumerates the American Presidency Project's "State of the Union" spoken-address
category (2 listing requests), filters to `--start-year`..`--end-year` (default
1950–2020), then fetches each document page and extracts the speech text,
president, and date from its Drupal containers (BeautifulSoup, no JavaScript).
Writes one `texts/<president>_<year>.txt` per address plus a manifest row with a
**party** label (from a local president→party map) — the ground truth `bt_scoring`
validates its emergent economic-left/right scale against. SOTU addresses are
US-government works → public domain. The site's `robots.txt` blocks named AI
crawlers but allows `/documents/` for a generic UA with `Crawl-delay: 10`, so the
default `--rate 10` honors that; the full window fetches in ~11 min.

### `fetch_presidential_audio.py` — presidential audio (podcasts, legacy)
Lists the Internet Archive `presidential_recordings` collection oldest-first and
downloads one small (lowest-bitrate MP3) audio file per item. Public domain,
spanning decades, but **no reference transcript**. This was `bt_scoring`'s original
feed (audio → Granite ASR → pairwise judge); on commodity/CPU hardware the ASR
transcripts proved too noisy to carry the judgment, so `bt_scoring` moved to SOTU
text (`fetch_sotu.py`). Kept as a general-purpose public-domain audio source.
