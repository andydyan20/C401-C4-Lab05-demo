from pathlib import Path

from fastapi import FastAPI, HTTPException

from apps.api.schemas import ChatRequest, ChatResponse, RetrieveDebugResponse
from packages.rag.service import RagChatService

BASE_DIR = Path(__file__).resolve().parents[2]
SEED_PATH = BASE_DIR / "data" / "vinuni_freshman_faq_seed.json"

app = FastAPI(title="VinUni Onboarding Agent API", version="0.1.0")
rag_service = RagChatService(seed_path=SEED_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    result = rag_service.chat(question=payload.question, role=payload.role)
    return ChatResponse(**result)


@app.post("/api/retrieve/debug", response_model=RetrieveDebugResponse)
def retrieve_debug(payload: ChatRequest) -> RetrieveDebugResponse:
    hits = rag_service.retrieve_debug(payload.question, top_k=5)
    return RetrieveDebugResponse(query=payload.question, hits=hits)
