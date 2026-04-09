from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_demo_flow_freshman_groundedness():
    # 1. Setup
    answers = {"role": "Tân sinh viên", "unit": "CECS", "term": "Fall 2024", "housing": "Có"}
    resp = client.post("/api/onboarding/initialize", json={"answers": answers})
    session_id = resp.json()["session_id"]
    
    # 2. Test Groundedness (Citations)
    questions = [
        "Em cần làm gì trong tuần đầu tiên?",
        "Lịch orientation xem ở đâu?",
        "Không đăng nhập được LMS thì làm gì?",
        "Đăng ký ký túc xá như thế nào?"
    ]
    
    cited_count = 0
    for q in questions:
        res = client.post("/api/chat", json={"question": q, "session_id": session_id})
        data = res.json()
        if len(data["citations"]) >= 1:
            cited_count += 1
            
    grounded_rate = cited_count / len(questions)
    print(f"Grounded Rate: {grounded_rate * 100}%")
    assert grounded_rate >= 0.75 # High groundedness for seeded questions

def test_demo_flow_staff_checklist():
    # 1. Setup for Staff
    answers = {"role": "Nhân viên mới", "unit": "Phòng IT", "term": "Đợt Q3/2024", "housing": "Không"}
    resp = client.post("/api/onboarding/initialize", json={"answers": answers})
    data = resp.json()
    checklist = data["checklist"]
    
    # 2. Verify Task Success (Correct generation)
    titles = [t["title"] for t in checklist]
    assert "Ký hợp đồng lao động" in titles
    assert "Hoàn thành thủ tục Ký túc xá" not in titles

def test_demo_flow_escalation_and_out_of_scope():
    # 1. Out of Scope
    resp = client.post("/api/chat", json={"question": "Mua crypto o dau?"})
    assert resp.json()["status"] == "out_of_scope"
    assert resp.json()["action"] == "escalate"

    # 2. Ambiguous query
    resp = client.post("/api/chat", json={"question": "Checklist"})
    assert resp.json()["intent"] == "clarify_context"
