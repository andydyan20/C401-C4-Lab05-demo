from pathlib import Path

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from apps.api.schemas import (
    ActionFeedbackRequest,
    ChatRequest,
    ChatResponse,
    ChecklistStatusResponse,
    OnboardingInitRequest,
    OnboardingInitResponse,
    OnboardingSetupResponse,
    RetrieveDebugResponse,
    SetupQuestion,
)
from packages.rag.onboarding_config import SETUP_QUESTIONS
from packages.rag.service import RagChatService

BASE_DIR = Path(__file__).resolve().parents[2]
SEED_PATH = BASE_DIR / "data" / "vinuni_freshman_faq_seed.json"

app = FastAPI(title="VinUni Onboarding Agent API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For prototype, allow all. In production, specify exact origins.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RagChatService(seed_path=SEED_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/onboarding/setup-questions", response_model=OnboardingSetupResponse)
def get_setup_questions() -> OnboardingSetupResponse:
    return OnboardingSetupResponse(questions=[SetupQuestion(**q) for q in SETUP_QUESTIONS])


@app.post("/api/onboarding/initialize", response_model=OnboardingInitResponse)
def initialize_onboarding(payload: OnboardingInitRequest) -> OnboardingInitResponse:
    session_id = rag_service.create_session(payload.answers)
    session = rag_service.get_session(session_id)
    return OnboardingInitResponse(session_id=session_id, checklist=session["checklist"])


@app.get("/api/onboarding/checklist/{session_id}", response_model=ChecklistStatusResponse)
def get_checklist(session_id: str) -> ChecklistStatusResponse:
    session = rag_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    checklist = session["checklist"]
    completed = len([t for t in checklist if t["status"] == "completed"])
    progress = (completed / len(checklist)) * 100 if checklist else 0
    return ChecklistStatusResponse(checklist=checklist, progress=progress)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    result = rag_service.chat(question=payload.question, role=payload.role, session_id=payload.session_id)
    return ChatResponse(**result)


@app.post("/api/actions/report-error")
def report_error(payload: ActionFeedbackRequest):
    # In a real app, log to DB. For prototype, just return OK.
    print(f"REPORT ERROR: Session {payload.session_id}, Feedback: {payload.feedback}")
    return {"status": "received"}


@app.post("/api/actions/transfer")
def transfer_department(payload: ActionFeedbackRequest):
    # In a real app, trigger escalation workflow.
    print(f"ESCALATE: Session {payload.session_id}, Reason: {payload.feedback}")
    return {"status": "escalated"}


@app.post("/api/retrieve/debug", response_model=RetrieveDebugResponse)
def retrieve_debug(payload: ChatRequest) -> RetrieveDebugResponse:
    hits = rag_service.retrieve_debug(payload.question, top_k=5)
    return RetrieveDebugResponse(query=payload.question, hits=hits)
