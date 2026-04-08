from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from packages.rag.retriever import SeedRetriever


OUT_OF_SCOPE_HINTS = {
    "part-time",
    "viec lam",
    "crypto",
    "chung khoan",
    "game",
    "du lich",
    "booking",
}


class GraphState(TypedDict, total=False):
    question: str
    role: str
    status: str
    action: str
    intent: str
    answer: str
    hits: list[dict[str, Any]]
    citations: list[dict[str, str]]


def _detect_node(state: GraphState) -> GraphState:
    q = state["question"].lower()
    if any(hint in q for hint in OUT_OF_SCOPE_HINTS):
        return {"status": "out_of_scope", "action": "escalate"}
    if len(q.strip().split()) < 3:
        return {"status": "ambiguous", "action": "clarify"}
    return {"status": "normal", "action": "answer"}


def _build_retrieve_node(retriever: SeedRetriever):
    def _retrieve_node(state: GraphState) -> GraphState:
        if state.get("status") != "normal":
            return {}
        hits = retriever.search(state["question"], role=state.get("role", "freshman_student"))
        return {"hits": [h.item for h in hits]}

    return _retrieve_node


def _respond_node(state: GraphState) -> GraphState:
    status = state.get("status", "normal")
    if status == "ambiguous":
        return {
            "intent": "clarify_context",
            "answer": "Minh can them thong tin de tra loi chinh xac. Ban la tan sinh vien hay nhan vien moi?",
            "citations": [],
        }
    if status == "out_of_scope":
        return {
            "intent": "out_of_scope",
            "answer": "Cau hoi nay hien ngoai pham vi onboarding. Minh co the ho tro: hoc vu, thu tuc hanh chinh, portal.",
            "citations": [],
        }

    hits = state.get("hits", [])
    if not hits:
        return {
            "status": "no_context",
            "action": "escalate",
            "intent": "no_context",
            "answer": "Minh chua tim duoc nguon phu hop. Ban vui long dien dat ro hon hoac chuyen bo phan ho tro.",
            "citations": [],
        }

    best = hits[0]
    citations = [{"id": best["id"], "category": best["category"], "intent": best["intent"]}]
    return {
        "intent": best["intent"],
        "answer": best["answer"],
        "citations": citations,
    }


def compile_chat_graph(retriever: SeedRetriever):
    graph = StateGraph(GraphState)
    graph.add_node("detect", _detect_node)
    graph.add_node("retrieve", _build_retrieve_node(retriever))
    graph.add_node("respond", _respond_node)

    graph.set_entry_point("detect")
    graph.add_edge("detect", "retrieve")
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    return graph.compile()
