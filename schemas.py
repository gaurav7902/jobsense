from pydantic import BaseModel, Field


class JobFilters(BaseModel):
    """Structured constraints extracted from a natural-language job query."""

    role: str | None = None
    location: str | None = None
    remote: bool | None = None
    min_salary: int | None = Field(default=None, ge=0)
    skills: list[str] = Field(default_factory=list)


class RetrievedJob(BaseModel):
    id: str
    title: str
    company: str
    location: str
    remote: bool
    salary: int
    skills: list[str]
    description: str
    chunk_text: str
    score: float


class ChatTurn(BaseModel):
    role: str
    content: str
