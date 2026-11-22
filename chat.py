import json
import re
from pathlib import Path

from llm import JobSenseLLM
from retriever import HybridRetriever
from schemas import ChatTurn

ROOT = Path(__file__).resolve().parent
PREFERENCES_FILE = ROOT / "preferences.json"


class PreferenceStore:
    def __init__(self, path: Path = PREFERENCES_FILE) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=True), encoding="utf-8"
        )

    def update_from_query(self, query: str) -> bool:
        changed = False
        text = query.lower()
        if re.search(
            r"\b(i )?(prefer|want|preferably).*remote\b|remote.*\b(preference|prefer)\b",
            text,
        ):
            changed = self.data.get("remote") is not True
            self.data["remote"] = True
        if re.search(r"\b(i )?(prefer|want).*on[- ]?site\b", text):
            changed = self.data.get("remote") is not False
            self.data["remote"] = False
        location_match = re.search(
            r"(?:prefer|want).*?(?:in|at) ([a-zA-Z ]{2,30})(?:\b|$)", text
        )
        if location_match:
            location = location_match.group(1).strip().title()
            changed = self.data.get("location") != location
            self.data["location"] = location
        if changed:
            self.save()
        return changed


def main() -> None:
    print("JobSense CLI - conversational job finder")
    print(
        "Type 'exit' to quit, '/prefs' for preferences, or '/ingest' to rebuild the index."
    )

    retriever = None
    try:
        retriever = HybridRetriever()
    except Exception as exc:
        print(f"Index unavailable: {exc}")
        print("Run 'python ingest.py --reset' first.")

    llm = JobSenseLLM()
    preferences = PreferenceStore()
    history: list[ChatTurn] = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if query.lower() == "/prefs":
            print(
                json.dumps(preferences.data, indent=2)
                if preferences.data
                else "No saved preferences."
            )
            continue
        if query.lower() == "/ingest":
            from ingest import ingest

            try:
                ingest(reset=True)
                retriever = HybridRetriever()
                print("Index rebuilt.")
            except Exception as exc:
                print(f"Ingestion failed: {exc}")
            continue

        preference_updated = preferences.update_from_query(query)
        if preference_updated:
            print("Saved preference.")
        if retriever is None:
            print("No good match: the local index is not available.")
            continue

        filters = llm.extract_filters(query, preferences.data)
        try:
            jobs = retriever.hybrid_search(query, filters, top_k=5)
        except Exception as exc:
            print(f"No good match: retrieval failed ({exc})")
            continue

        weak = not jobs or (jobs[0].score < 0.020 and len(jobs) < 3)
        if weak:
            answer = "No good match found in the current job data."
        else:
            answer = llm.generate_answer(query, jobs, history[-6:])

        print(f"\nJobSense: {answer}")
        history.append(ChatTurn(role="user", content=query))
        history.append(ChatTurn(role="assistant", content=answer))


if __name__ == "__main__":
    main()
