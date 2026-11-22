import argparse
import concurrent.futures
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "jobs.json"

GREENHOUSE_SLUGS = [
    "stripe",
    "cloudflare",
    "databricks",
    "dropbox",
    "reddit",
    "figma",
    "coinbase",
    "discord",
    "airbnb",
    "scaleai",
    "anthropic",
    "robinhood",
    "mongodb",
    "gitlab",
    "twilio",
    "elastic",
    "lyft",
    "spacex",
    "block",
    "affirm",
    "brex",
    "chime",
    "instacart",
    "pinterest",
    "asana",
    "duolingo",
    "cockroachlabs",
    "toast",
    "fastly",
    "okta",
    "braze",
    "datadog",
    "confluent",
    "redis",
    "clickhouse",
    "squarespace",
    "deliveroo",
    "mozilla",
    "canonical",
    "calendly",
    "crunchyroll",
]


def curl_json(url: str) -> dict | list | None:
    try:
        result = subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "20",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = html.unescape(value)
    return " ".join(value.split())


def infer_salary(text: str) -> int:
    patterns = [
        r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*[kK]",
        r"\$\s*(\d+(?:,\d{3})*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            values = [float(match.replace(",", "")) for match in matches]
            value = max(values)
            return int(value * 1000 if "k" in text.lower() and value < 1000 else value)
    return 0


def infer_remote(location: str, text: str) -> bool:
    haystack = f"{location} {text}".lower()
    return "remote" in haystack or "work from home" in haystack


def infer_skills(title: str, description: str) -> list[str]:
    text = f"{title} {description}"
    candidates = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "React",
        "Node.js",
        "Go",
        "C++",
        "Rust",
        "Ruby",
        "PHP",
        "Swift",
        "Kotlin",
        "SQL",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Terraform",
        "FastAPI",
        "Django",
        "Flask",
        "Spring",
        "Kafka",
        "Spark",
        "Airflow",
        "PyTorch",
        "TensorFlow",
        "Machine Learning",
        "Data Analysis",
        "Tableau",
        "Selenium",
        "Figma",
        "Salesforce",
        "Cybersecurity",
        "Linux",
    ]
    lower = text.lower()
    found = [
        skill
        for skill in candidates
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(skill.lower())}(?![A-Za-z0-9])", lower
        )
    ]
    return found[:12]


def fetch_remoteok() -> list[dict]:
    payload = curl_json("https://remoteok.com/api")
    if not isinstance(payload, list):
        return []
    jobs = []
    for posting in payload:
        if not isinstance(posting, dict) or not posting.get("id"):
            continue
        title = clean_text(str(posting.get("position") or posting.get("title") or ""))
        company = clean_text(str(posting.get("company") or "RemoteOK"))
        description = clean_text(str(posting.get("description") or title))
        location = clean_text(str(posting.get("location") or "Remote"))
        salary = int(posting.get("salary_max") or posting.get("salary_min") or 0)
        url = str(posting.get("url") or posting.get("apply_url") or "")
        if not title:
            continue
        jobs.append(
            {
                "id": f"RO{posting['id']}",
                "title": title,
                "company": company,
                "location": location or "Remote",
                "remote": True,
                "salary": salary,
                "skills": infer_skills(title, description),
                "description": description[:5000],
                "source": "RemoteOK",
                "source_url": url,
            }
        )
    return jobs


def fetch_greenhouse_slug(slug: str) -> list[dict]:
    payload = curl_json(
        f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    )
    if not isinstance(payload, dict):
        return []
    jobs = []
    for posting in payload.get("jobs", []):
        title = clean_text(str(posting.get("title") or ""))
        location = clean_text(
            str((posting.get("location") or {}).get("name") or "Remote")
        )
        description = clean_text(str(posting.get("content") or ""))
        url = str(posting.get("absolute_url") or "")
        if not title:
            continue
        text = f"{title} {description}"
        jobs.append(
            {
                "id": f"GH{posting.get('id')}",
                "title": title,
                "company": slug,
                "location": location,
                "remote": infer_remote(location, text),
                "salary": infer_salary(text),
                "skills": infer_skills(title, description),
                "description": description[:5000],
                "source": "Greenhouse",
                "source_url": url,
            }
        )
    return jobs


def deduplicate(jobs: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for job in jobs:
        key = (
            job["title"].lower(),
            job["company"].lower(),
            job["location"].lower(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public live job postings")
    parser.add_argument("--limit", type=int, default=800)
    args = parser.parse_args()
    if not 500 <= args.limit <= 1000:
        parser.error("--limit must be between 500 and 1000")

    jobs = fetch_remoteok()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        greenhouse_batches = list(executor.map(fetch_greenhouse_slug, GREENHOUSE_SLUGS))
    for batch in greenhouse_batches:
        jobs.extend(batch)

    jobs = deduplicate(jobs)
    if len(jobs) < 500:
        raise RuntimeError(
            f"Only fetched {len(jobs)} unique postings; at least 500 are required"
        )
    jobs = jobs[: args.limit]
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n")
    print(f"Fetched {len(jobs)} unique real postings")
    print(f"Wrote {DATA_FILE}")


if __name__ == "__main__":
    main()
