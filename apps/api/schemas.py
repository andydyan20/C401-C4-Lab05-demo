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


class SetupQuestion(BaseModel):
    id: str
    text: str
    type: str  # select, text
    options: list[str] | None = None


class OnboardingSetupResponse(BaseModel):
    questions: list[SetupQuestion]


class OnboardingInitRequest(BaseModel):
    answers: dict[str, str]


class ChecklistTask(BaseModel):
    id: str
    title: str
    status: str  # pending, completed
    link: str | None = None


class OnboardingInitResponse(BaseModel):
    session_id: str
    checklist: list[ChecklistTask]


class ChecklistStatusResponse(BaseModel):
    checklist: list[ChecklistTask]
    progress: float


class ActionFeedbackRequest(BaseModel):
    session_id: str
    action: str
    feedback: str | None = None


class RetrieveDebugResponse(BaseModel):
    query: str
    hits: list[dict[str, Any]]
