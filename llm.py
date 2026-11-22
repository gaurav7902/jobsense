import json
import os
from typing import Iterable

from dotenv import load_dotenv

from schemas import ChatTurn, JobFilters, RetrievedJob

load_dotenv()


class JobSenseLLM:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._client = None
        if self.api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                print(f"Warning: could not initialize Gemini client: {exc}")

    def extract_filters(self, query: str, preferences: dict) -> JobFilters:
        if self._client:
            preference_text = json.dumps(preferences, ensure_ascii=True)
            prompt = f"""Extract concise search filters from the query and preferences.
Return only valid JSON matching this shape:
{{"role": string|null, "location": string|null, "remote": boolean|null,
 "min_salary": integer|null, "skills": string[]}}

Preferences: {preference_text}
Query: {query}
"""
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.rsplit("```", 1)[0].split("```", 1)[-1].strip()
                    text = text.removeprefix("json").strip()
                return JobFilters.model_validate_json(text)
            except Exception as exc:
                print(
                    f"Warning: filter extraction failed, using fallback parser: {exc}"
                )
        return self._fallback_filters(query, preferences)

    @staticmethod
    def _fallback_filters(query: str, preferences: dict) -> JobFilters:
        text = f"{query} {json.dumps(preferences)}".lower()
        min_salary = None
        import re

        explicit = re.search(r"(?:\d+(?:,\d{3})*(?:\.\d+)?)\s*[k]", text)
        if explicit:
            min_salary = int(float(explicit.group(0).replace(",", "").replace("k", "")) * 1000)
        else:
            match = re.search(r"(?:minimum|min(?:imum)?|over|plus)\s*(?:of\s*)?\$?\s*(\d+(?:,\d{3})*)", text)
            if match:
                min_salary = int(match.group(1).replace(",", ""))
        skills = []
        candidates = (
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "node.js",
            "go",
            "golang",
            "aws",
            "azure",
            "gcp",
            "sql",
            "postgresql",
            "docker",
            "kubernetes",
            "fastapi",
            "flask",
            "django",
            "spring",
            "cybersecurity",
            "machine learning",
            "pandas",
            "pytorch",
            "tensorflow",
            "c++",
            "rust",
            "php",
            "data analysis",
            "excel",
            "figma",
            "salesforce",
            "accounting",
        )
        skills = [skill for skill in candidates if skill in text]

        role = None
        for title in (
            "intern",
            "software engineer",
            "backend",
            "frontend",
            "data analyst",
            "devops",
            "qa",
            "designer",
            "sales",
            "accountant",
        ):
            if title in text:
                role = title
                break

        remote = None
        if "remote" in text:
            remote = True
        elif "on-site" in text or "onsite" in text:
            remote = False

        location = preferences.get("location")
        return JobFilters(
            role=role,
            location=location,
            remote=remote,
            min_salary=min_salary,
            skills=skills,
        )

    def generate_answer(
        self, query: str, jobs: list[RetrievedJob], history: Iterable[ChatTurn]
    ) -> str:
        if not jobs:
            return "No good match found in the current job data."

        if not self._client:
            lines = ["No good match" if not jobs else "Top matches:"]
            if jobs:
                for job in jobs[:5]:
                    lines.append(
                        f"- [{job.id}] {job.title} at {job.company}: matches the requested role or skills. Salary {job.salary:,}; {job.location}."
                    )
            lines.append(
                "Set GEMINI_API_KEY to receive natural-language explanations."
            )
            return "\n".join(lines)

        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in history)
        job_text = json.dumps([job.model_dump() for job in jobs], default=str)
        prompt = f"""You are JobSense CLI, a job recommendation assistant.
Use only the retrieved jobs below. Rank the top 3-5. Explain why each fits the user's query.
Cite every job-specific claim with its job id in brackets, for example [JOB001].
If retrieval is weak or none fits, say exactly: no good match
Do not invent jobs, skills, salaries, locations, or details not present in the retrieved data.

Conversation history:
{history_text}

User query:
{query}

Retrieved jobs:
{job_text}
"""
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            return f"Could not generate the answer: {exc}"
