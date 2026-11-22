import json
import sys
import types
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    import onnxruntime  # noqa: F401
except ImportError:
    sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

import chromadb

from schemas import JobFilters, RetrievedJob

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
DATA_FILE = ROOT / "data" / "jobs.json"


class HybridRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection("jobs")
        stored = self.collection.get(include=["documents", "metadatas"])
        self.documents = stored["documents"]
        self.ids = stored["ids"]
        self.metadatas = stored["metadatas"]
        tokenized = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        with DATA_FILE.open("r", encoding="utf-8") as handle:
            self.jobs = {job["id"]: job for job in json.load(handle)}

    def _where(self, filters: JobFilters) -> dict | None:
        clauses: list[dict] = []
        if filters.remote is not None:
            clauses.append({"remote": filters.remote})
        if filters.min_salary is not None:
            clauses.append({"salary": {"$gte": filters.min_salary}})
        if filters.location:
            clauses.append({"location": filters.location})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def vector_search(
        self, query: str, top_k: int = 20, filters: JobFilters | None = None
    ) -> list[RetrievedJob]:
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()[0]
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, max(self.collection.count(), 1)),
        }
        if filters:
            where = self._where(filters)
            if where:
                kwargs["where"] = where
        try:
            result = self.collection.query(
                **kwargs, include=["documents", "metadatas", "distances"]
            )
        except Exception as exc:
            raise RuntimeError(f"Vector search failed: {exc}") from exc
        return self._dedupe(
            result["ids"][0],
            result["metadatas"][0],
            result["documents"][0],
            result["distances"][0],
        )

    def bm25_search(
        self, query: str, top_k: int = 20, filters: JobFilters | None = None
    ) -> list[RetrievedJob]:
        scores = self.bm25.get_scores(query.lower().split())
        allowed: set[int] = set(range(len(self.documents)))
        if filters:
            allowed = {
                index
                for index in allowed
                if (
                    filters.remote is None
                    or self.metadatas[index]["remote"] == filters.remote
                )
                and (
                    filters.min_salary is None
                    or self.metadatas[index]["salary"] >= filters.min_salary
                )
                and (
                    not filters.location
                    or self.metadatas[index]["location"] == filters.location
                )
            }
        ranked = sorted((i for i in allowed), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return self._dedupe(
            [self.ids[i] for i in ranked],
            [self.metadatas[i] for i in ranked],
            [self.documents[i] for i in ranked],
            [1.0 / (1.0 + max(float(scores[i]), 0.0)) for i in ranked],
        )

    def hybrid_search(
        self, query: str, filters: JobFilters, top_k: int = 5
    ) -> list[RetrievedJob]:
        filter_terms = " ".join(
            part
            for part in (
                filters.role or "",
                " ".join(filters.skills),
                filters.location or "",
            )
            if part
        )
        bm25_query = f"{query} {filter_terms}".strip()

        vector_jobs = self.vector_search(
            query + (" " + filter_terms), top_k=20, filters=filters
        )
        bm25_jobs = self.bm25_search(bm25_query, top_k=20, filters=filters)

        rrf: dict[str, float] = {}
        ranks: dict[str, int] = {}
        for ranked in (vector_jobs, bm25_jobs):
            for rank, job in enumerate(ranked, start=1):
                rrf[job.id] = rrf.get(job.id, 0.0) + 1.0 / (60 + rank)
                ranks.setdefault(job.id, rank)

        combined = []
        by_id = {job.id: job for job in vector_jobs}
        by_id.update({job.id: job for job in bm25_jobs})
        for job_id, score in sorted(
            rrf.items(), key=lambda item: item[1], reverse=True
        )[:top_k]:
            job = by_id[job_id].model_copy(update={"score": round(score, 6)})
            combined.append(job)
        return combined

    def _dedupe(
        self,
        ids: list[str],
        metadatas: list[dict],
        documents: list[str],
        distances: list[float],
    ) -> list[RetrievedJob]:
        best: dict[str, RetrievedJob] = {}
        for chunk_id, metadata, document, distance in zip(
            ids, metadatas, documents, distances
        ):
            job_id = metadata["job_id"]
            job_data = self.jobs[job_id]
            score = float(distance) if isinstance(distance, float) else float(distance)
            candidate = RetrievedJob(
                id=job_id,
                title=metadata["title"],
                company=metadata["company"],
                location=metadata["location"],
                remote=bool(metadata["remote"]),
                salary=int(metadata["salary"]),
                skills=list(job_data["skills"]),
                description=job_data["description"],
                chunk_text=document,
                score=score,
            )
            if job_id not in best:
                best[job_id] = candidate
            elif score < best[job_id].score:
                best[job_id] = candidate
        return list(best.values())
