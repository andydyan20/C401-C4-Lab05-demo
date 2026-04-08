from pathlib import Path
from typing import Any

from packages.rag.langgraph_flow import compile_chat_graph
from packages.rag.retriever import SeedRetriever
from packages.rag.seed_loader import load_seed


class RagChatService:
    def __init__(self, seed_path: str | Path):
        self.seed_path = Path(seed_path)
        self.items = load_seed(self.seed_path)
        self.retriever = SeedRetriever(self.items)
        self.graph = compile_chat_graph(self.retriever)

    def reload(self) -> None:
        self.items = load_seed(self.seed_path)
        self.retriever = SeedRetriever(self.items)
        self.graph = compile_chat_graph(self.retriever)

    def chat(self, question: str, role: str = "freshman_student") -> dict[str, Any]:
        state = self.graph.invoke({"question": question, "role": role})
        return {
            "answer": state.get("answer", ""),
            "intent": state.get("intent", "unknown"),
            "status": state.get("status", "normal"),
            "action": state.get("action", "answer"),
            "citations": state.get("citations", []),
            "meta": {"seed_count": len(self.items)},
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
