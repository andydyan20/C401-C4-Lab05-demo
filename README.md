# C401-C4 - VinUni Onboarding Agent (MVP)

MVP backend theo ke hoach: FastAPI + LangChain/LangGraph + seed data FAQ cho tan sinh vien.

## 1) Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Nap data seed

```bash
python scripts/ingest_seed.py
```

## 3) Chay API

```bash
uvicorn apps.api.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

## 4) Test nhanh

```bash
pytest -q
```

## 5) Endpoint chinh

- `POST /api/chat`: hoi dap onboarding.
- `POST /api/retrieve/debug`: xem ket qua retrieve (chi dung cho dev).
- `GET /health`: kiem tra he thong.

## 6) Cau tra loi khi user hoi sai

Flow LangGraph:
1. detect (`ambiguous` / `out_of_scope` / `normal`)
2. retrieve
3. respond (safe answer + citation + suggest/escalate)

