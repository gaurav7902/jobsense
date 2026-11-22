# JobSense CLI

JobSense CLI is a local, conversational job finder. It uses persistent ChromaDB retrieval, local Sentence Transformers embeddings, BM25 plus Reciprocal Rank Fusion, optional Gemini answer generation, and a preference file for basic session memory.

## Setup

Linux/macOS:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py --reset
python chat.py
```

Set `GEMINI_API_KEY` in `.env` for Gemini answers and structured filter extraction. The CLI still works without an API key: it uses the local Pydantic fallback parser and prints retrieved jobs directly.

Refresh the live dataset with 800 public postings:

```bash
python fetch_jobs.py --limit 800
python ingest.py --reset
```

The fetcher combines public RemoteOK listings and Greenhouse board APIs, deduplicates records, and writes attribution URLs in `data/jobs.json`. RemoteOK asks applications to link back to RemoteOK as the source.

The first ingestion downloads the `all-MiniLM-L6-v2` Sentence Transformer model and creates the persistent ChromaDB index under `chroma_db/`. Re-run `python ingest.py --reset` after changing `data/jobs.json`.

Optional model override:

```text
GEMINI_MODEL=gemini-3.8-flash
```

## Run

```bash
python chat.py
```

Example query:

```text
You: remote python backend internship, 50k+
```

CLI commands:

- `exit`, `quit`, or `q`: leave the session
- `/prefs`: show saved preferences
- `/ingest`: rebuild the Chroma index

## Architecture

```text
jobs.json
    |
    v
Character chunking (600 chars, 120 overlap)
    |
    v
Sentence Transformer (all-MiniLM-L6-v2)
    |
    v
Persistent ChromaDB collection + BM25 corpus
    |
User query --> Pydantic filters --> metadata filtering
    |                                  |
    +--> dense vector search            +--> BM25 keyword search
                         \             /
                          Reciprocal Rank Fusion
                                   |
                                   v
                         top 3-5 retrieved jobs
                                   |
                                   v
             Gemini LLM + job IDs + conversation history
```

## Design decisions

**Chunk size:** Descriptions are short, so 600-character chunks preserve enough context to identify requirements and responsibilities without creating many weak fragments. A 120-character overlap avoids losing context at boundaries.

**Local embeddings:** `all-MiniLM-L6-v2` runs locally, needs no API key, and produces compact 384-dimensional embeddings suitable for the 800-posting corpus.

**Hybrid retrieval:** Vector search handles semantic wording while BM25 is good at exact job titles, skills, and companies. Combining both reduces misses caused by vocabulary differences.

**Reciprocal Rank Fusion:** RRF merges rank positions rather than incomparable raw similarity and BM25 scores. The implementation uses `1 / (60 + rank)`, a standard modest-rank constant that is insensitive to the two systems' score scales.

**Structured filters:** The LLM is instructed to return a Pydantic `JobFilters` object containing role, location, remote, minimum salary, and skills. A Pydantic fallback parser handles missing API keys or parsing failures. Remote, minimum salary, and exact location use Chroma metadata constraints; role and skills are also added to the search text and BM25 query.

**Weak-match handling:** The CLI suppresses an answer when retrieval returns no results or the fused evidence is weak. It returns “No good match found in the current job data.” instead of inventing a job.

**Conversation memory:** Chat history is kept in memory for the current process and sent to the generator as context. Statements such as “I prefer remote” or “I prefer on-site” update `preferences.json`; saved preferences are merged into later filter extraction.

## Evaluation

The evaluation uses exact-term, paraphrased, remote, and salary-filtered queries.

| Method      | Recall@5 |   MRR |
| ----------- | -------: | ----: |
| Vector-only |    0.538 | 0.500 |
| Hybrid RRF  |    0.923 | 0.637 |

Run the comparison again:

```bash
python eval.py
```

These results are from the 800-posting live dataset. Recall@5 measures whether at least one relevant posting appears in the top five, while MRR measures the mean reciprocal rank of the first relevant result.

## Project files

- `ingest.py`: loads job data, chunks descriptions, embeds chunks, and writes Chroma data
- `retriever.py`: vector search, BM25, metadata filtering, dedupe, and RRF
- `llm.py`: Gemini filter extraction and answer generation
- `chat.py`: CLI loop and preference/session memory
- `eval.py`: vector and hybrid benchmark
- `fetch_jobs.py`: fetches, normalizes, deduplicates, and writes live postings
- `schemas.py`: Pydantic models
- `data/jobs.json`: 800 live postings from RemoteOK and Greenhouse
- `requirements.txt`: Python dependencies
- `.env.example`: example environment variables

## Limitations and future improvements

- Public job feeds are snapshots and may contain stale listings; refresh regularly before live use.
- Exact location filtering can return zero results for broader queries such as “India” or “Europe.” Location aliases and geographic normalization would improve this.
- The BM25 corpus is rebuilt whenever the retriever starts; a persistent BM25 index would reduce startup cost for larger datasets.
- Salary units can be missing or inconsistent across companies and public sources. A production system needs explicit currency, compensation type, and period normalization.
- Chunk ranking is converted to job-level rankings by deduplicating IDs; larger jobs could benefit from span-aware scores.
- The no-API-key fallback does not create Gemini-quality prose, but it remains useful for local testing.
- Future work could add reranking, cross-encoder scoring, query spell correction, source provenance, and confidence calibration.
