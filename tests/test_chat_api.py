from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_known_question_returns_citation() -> None:
    payload = {"question": "Em la tan sinh vien, tuan dau tien can lam gi?"}
    response = client.post("/api/chat", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["intent"] != "out_of_scope"
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["id"].startswith("FV-")


def test_chat_out_of_scope() -> None:
    payload = {"question": "Cho minh hoi cach mua crypto an toan?"}
    response = client.post("/api/chat", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "out_of_scope"
    assert data["action"] == "escalate"


def test_retrieve_debug() -> None:
    payload = {"question": "dang ky ky tuc xa nhu the nao"}
    response = client.post("/api/retrieve/debug", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["hits"]
    assert any(hit["category"] == "housing" for hit in data["hits"])
