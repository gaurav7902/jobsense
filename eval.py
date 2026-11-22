from llm import JobSenseLLM
from retriever import HybridRetriever

TEST_QUERIES = [
    # Exact BM25 terms.
    ("Django", lambda job: "Django" in job.skills),
    ("Kubernetes", lambda job: "Kubernetes" in job.skills),
    ("PostgreSQL", lambda job: "PostgreSQL" in job.skills),
    # Paraphrased roles and cross-vocabulary descriptions.
    ("server-side engineer", lambda job: (
        "backend" in job.title.lower() or "server" in job.title.lower()
        or "engineer" in job.title.lower()
    )),
    ("artificial intelligence model developer", lambda job: (
        "PyTorch" in job.skills or "TensorFlow" in job.skills
        or "machine learning" in job.skills
    )),
    ("person who creates smooth user-facing web screens", lambda job: (
        "React" in job.skills or "TypeScript" in job.skills
        or "frontend" in job.title.lower()
    )),
    ("company infrastructure and deployment specialist", lambda job: (
        "AWS" in job.skills or "Terraform" in job.skills
        or "DevOps" in job.title or "Kubernetes" in job.skills
    )),
    ("systems programmer working close to the operating system", lambda job: (
        "Rust" in job.skills or "C++" in job.skills or "Linux" in job.skills
    )),
    # Remote and salary filters.
    ("remote backend engineer", lambda job: job.remote),
    ("remote Python engineer salary 100k plus", lambda job: (
        job.remote and "Python" in job.skills and job.salary >= 100000
    )),
    ("Python developer salary 150k minimum", lambda job: (
        "Python" in job.skills and job.salary >= 150000
    )),
    ("Kubernetes engineer remote", lambda job: (
        job.remote and "Kubernetes" in job.skills
    )),
    ("Python infrastructure role salary 120k+", lambda job: (
        "Python" in job.skills and job.salary >= 120000
    )),
]


def metrics(
    retriever: HybridRetriever, llm: JobSenseLLM, hybrid: bool
) -> tuple[float, float]:
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    for query, matches in TEST_QUERIES:
        if hybrid:
            filters = llm.extract_filters(query, {})
            results = retriever.hybrid_search(query, filters, top_k=5)
        else:
            results = retriever.vector_search(query, top_k=5)
        rank = next(
            (index for index, job in enumerate(results, start=1) if matches(job)),
            0,
        )
        recall_hits += 1.0 if rank else 0.0
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    recall = recall_hits / len(TEST_QUERIES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall, mrr


def main() -> None:
    retriever = HybridRetriever()
    llm = JobSenseLLM()
    vector_recall, vector_mrr = metrics(retriever, llm, hybrid=False)
    hybrid_recall, hybrid_mrr = metrics(retriever, llm, hybrid=True)

    print("JobSense Retrieval Evaluation")
    print("+----------------+-----------+-----------+")
    print("| Method         | Recall@5  | MRR       |")
    print("+----------------+-----------+-----------+")
    print(f"| Vector-only    | {vector_recall:.3f}     | {vector_mrr:.3f}     |")
    print(f"| Hybrid RRF     | {hybrid_recall:.3f}     | {hybrid_mrr:.3f}     |")
    print("+----------------+-----------+-----------+")


if __name__ == "__main__":
    main()
