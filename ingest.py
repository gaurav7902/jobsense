import argparse
import json
import sys
import types
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    import onnxruntime  # noqa: F401
except ImportError:
    sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

import chromadb

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "jobs.json"
CHROMA_DIR = ROOT / "chroma_db"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def character_chunks(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Simple character chunking with overlap for short job descriptions."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > size // 2:
                chunk = chunk[:last_space]
                end = start + last_space
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_jobs(path: Path = DATA_FILE) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        jobs = json.load(handle)
    if not isinstance(jobs, list):
        raise ValueError("data/jobs.json must contain a JSON array")
    return jobs


def ingest(reset: bool = False) -> int:
    jobs = load_jobs()
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection("jobs")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name="jobs",
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for job in jobs:
        skills = ", ".join(job.get("skills", []))
        context = (
            f"{job['title']} at {job['company']}. Location: {job['location']}. "
            f"Skills: {skills}. Description: {job['description']}"
        )
        for index, chunk in enumerate(character_chunks(context)):
            ids.append(f"{job['id']}:{index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "job_id": job["id"],
                    "title": job["title"],
                    "company": job["company"],
                    "location": job["location"],
                    "remote": bool(job["remote"]),
                    "salary": int(job["salary"]),
                    "skills": skills,
                    "chunk_index": index,
                }
            )

    embeddings = model.encode(
        documents, normalize_embeddings=True, show_progress_bar=True
    ).tolist()
    batch_size = 4000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"Ingested {len(jobs)} jobs into {len(documents)} chunks at {CHROMA_DIR}")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest JobSense jobs into local ChromaDB"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete the existing jobs collection"
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
