import os
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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

SYSTEM_PROMPT = """Bạn là trợ lý onboarding chính thức của VinUni (Vingroup University).
Nhiệm vụ: trả lời chính xác các câu hỏi về onboarding cho tân sinh viên và nhân viên mới.

Quy tắc:
1. CHỈ trả lời dựa trên ngữ cảnh (context) được cung cấp. Không bịa thông tin.
2. Trả lời bằng tiếng Việt, ngắn gọn, thân thiện, dùng "mình/bạn".
3. Nếu context không đủ, nói thẳng rằng mình chưa có thông tin và đề nghị liên hệ bộ phận hỗ trợ.
4. Luôn trích dẫn nguồn (category, intent) cuối câu trả lời khi có thể.
5. Nếu câu hỏi về vai trò cụ thể (sinh viên/nhân viên), ưu tiên thông tin phù hợp vai trò đó.
"""


class GraphState(TypedDict, total=False):
    question: str
    role: str
    unit: str
    status: str
    action: str
    intent: str
    answer: str
    hits: list[dict[str, Any]]
    citations: list[dict[str, str]]


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create Gemini LLM instance. Requires GOOGLE_API_KEY env var."""
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0.3,
        max_output_tokens=1024,
    )


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

    # --- Short-circuit: ambiguous / out-of-scope ---
    if status == "ambiguous":
        return {
            "intent": "clarify_context",
            "answer": "Mình cần thêm thông tin để trả lời chính xác. Bạn là tân sinh viên hay nhân viên mới?",
            "citations": [],
        }
    if status == "out_of_scope":
        return {
            "intent": "out_of_scope",
            "answer": "Câu hỏi này hiện ngoài phạm vi onboarding. Mình có thể hỗ trợ: học vụ, thủ tục hành chính, portal.",
            "citations": [],
        }

    # --- No retrieval hits ---
    hits = state.get("hits", [])
    if not hits:
        return {
            "status": "no_context",
            "action": "escalate",
            "intent": "no_context",
            "answer": "Mình chưa tìm được nguồn phù hợp. Bạn vui lòng diễn đạt rõ hơn hoặc chuyển bộ phận hỗ trợ.",
            "citations": [],
        }

    # --- Build context from retrieved hits ---
    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[{i}] Category: {hit['category']} | Intent: {hit['intent']}\n"
            f"Q: {hit['question']}\nA: {hit['answer']}"
        )
    context_text = "\n\n".join(context_parts)

    role_info = state.get("role", "freshman_student")
    unit_info = state.get("unit", "N/A")

    user_prompt = (
        f"Vai trò người hỏi: {role_info} | Đơn vị: {unit_info}\n\n"
        f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---\n\n"
        f"Câu hỏi: {state['question']}\n\n"
        f"Hãy trả lời dựa trên context trên."
    )

    # --- Call Gemini via LangChain ---
    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        generated_answer = response.content
    except Exception as e:
        # Fallback to deterministic answer if LLM call fails
        best = hits[0]
        generated_answer = f"{best['answer']}\n\n⚠️ (Fallback — LLM unavailable: {type(e).__name__})"

    best = hits[0]
    citations = [{"id": best["id"], "category": best["category"], "intent": best["intent"]}]
    return {
        "intent": best["intent"],
        "answer": generated_answer,
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
