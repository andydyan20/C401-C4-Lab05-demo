import uuid
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from packages.rag.langgraph_flow import compile_chat_graph
from packages.rag.retriever import SeedRetriever
from packages.rag.seed_loader import load_seed
from packages.rag.onboarding_config import get_checklist_for_answers


class RagChatService:
    def __init__(self, seed_path: str | Path):
        self.seed_path = Path(seed_path)
        self.items = load_seed(self.seed_path)
        self.retriever = SeedRetriever(self.items)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.graph = compile_chat_graph(self.retriever, self.llm)
        self.sessions: dict[str, dict[str, Any]] = {}

    def reload(self) -> None:
        self.items = load_seed(self.seed_path)
        self.retriever = SeedRetriever(self.items)
        self.graph = compile_chat_graph(self.retriever, self.llm)

    def create_session(self, answers: dict[str, str]) -> str:
        session_id = str(uuid.uuid4())
        checklist = get_checklist_for_answers(answers)
        self.sessions[session_id] = {
            "answers": answers,
            "checklist": [{"id": t["id"], "title": t["title"], "status": "pending", "link": t.get("link")} for t in checklist],
            "role_context": answers.get("role", "freshman_student"),
            "unit": answers.get("unit", "CAS")
        }
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def chat(self, question: str, role: str = "freshman_student", session_id: str | None = None) -> dict[str, Any]:
        context_role = role
        unit = "N/A"
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            context_role = session["role_context"]
            unit = session["unit"]

        try:
            state = self.graph.invoke({
                "question": question, 
                "role": context_role,
                "unit": unit
            })
        except Exception as e:
            print(f"LLM ERROR: {e}")
            # Fallback to pure retrieval if LLM fails
            hits = self.retriever.search(question, role=context_role)
            if hits:
                best = hits[0].item
                return {
                    "answer": best["answer"],
                    "intent": best["intent"],
                    "status": "normal",
                    "action": "answer",
                    "citations": [{"id": best["id"], "category": best["category"], "intent": best["intent"]}],
                    "meta": {"error": str(e), "fallback": True}
                }
            raise e
        
        return {
            "answer": state.get("answer", ""),
            "intent": state.get("intent", "unknown"),
            "status": state.get("status", "normal"),
            "action": state.get("action", "answer"),
            "citations": state.get("citations", []),
            "meta": {
                "seed_count": len(self.items),
                "session_id": session_id,
                "unit": unit
            },
        }

    def retrieve_debug(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        hits = self.retriever.search(query, top_k=top_k)
        return [
            {
                "id": hit.item["id"],
                "intent": hit.item["intent"],
                "category": hit.item["category"],
                "question": hit.item["question"],
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
