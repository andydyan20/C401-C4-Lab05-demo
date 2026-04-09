from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_onboarding_wizard_flow():
    # 1. Get setup questions
    resp = client.get("/api/onboarding/setup-questions")
    assert resp.status_code == 200
    questions = resp.json()["questions"]
    assert len(questions) >= 3
    
    # 2. Initialize onboarding
    answers = {
        "role": "Tân sinh viên",
        "unit": "CECS",
        "term": "Fall 2024",
        "housing": "Có"
    }
    resp = client.post("/api/onboarding/initialize", json={"answers": answers})
    assert resp.status_code == 200
    data = resp.json()
    session_id = data["session_id"]
    checklist = data["checklist"]
    assert session_id
    assert len(checklist) > 0
    # Check if housing task is added
    assert any(t["id"] == "dorm" for t in checklist)

    # 3. Get checklist status
    resp = client.get(f"/api/onboarding/checklist/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["progress"] == 0

    # 4. Chat with session context
    payload = {
        "question": "Lich orientation xem o dau?",
        "session_id": session_id
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    chat_data = resp.json()
    assert "portal sinh vien" in chat_data["answer"].lower()
    assert chat_data["meta"]["session_id"] == session_id

def test_onboarding_feedback_actions():
    payload = {
        "session_id": "test-session",
        "action": "report_error",
        "feedback": "Thông tin về học phí bị sai"
    }
    resp = client.post("/api/actions/report-error", json=payload)
    assert resp.status_code == 200

    resp = client.post("/api/actions/transfer", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "escalated"
