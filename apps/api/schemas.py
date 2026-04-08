from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    role: str = Field(default="freshman_student")
    session_id: str | None = None


class Citation(BaseModel):
    id: str
    category: str
    intent: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    status: str
    action: str
    citations: list[Citation]
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrieveDebugResponse(BaseModel):
    query: str
    hits: list[dict[str, Any]]
