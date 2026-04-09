# VinUni Onboarding Agent (Prototype)

Integrated AI Assistant for role-based onboarding (New Students / New Staff). Features a RAG-powered chat, setup wizard, and dynamic checklist.

## 1) Setup & Installation

### Backend

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```bash
cd apps/web
npm install
```

## 2) Running the Project

You need to run both the backend and frontend simultaneously.

### Start Backend

**macOS / Linux:**
```bash
source venv/bin/activate
PYTHONPATH=. python3 -m uvicorn apps.api.main:app --reload
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
$env:PYTHONPATH="."
python -m uvicorn apps.api.main:app --reload
```

API docs available at: `http://127.0.0.1:8000/docs`

### Start Frontend
```bash
cd apps/web
npm run dev
```
Access the UI at: `http://localhost:5173`

## 3) Testing

### Automated Tests

**macOS / Linux:**
```bash
PYTHONPATH=. python3 -m pytest tests/test_onboarding_flow.py
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."
python -m pytest tests/test_onboarding_flow.py
```

### Manual Verification
1. Open the UI at `http://localhost:5173`.
2. Complete the **Setup Wizard** (4 questions).
3. Interact with the **AI Agent** (e.g., "Lịch orientation xem ở đâu?").
4. Verify the **Checklist** on the left panel updates based on your role.
5. Use "Báo sai" or "Chuyển bộ phận" buttons to test feedback/escalation.

## 4) Key Features
- **Role-based Setup**: Tailored onboarding for Students vs. Staff.
- **Dynamic Checklist**: Tasks generated based on role, department, and housing status.
- **RAG with Citations**: Answers are grounded in VinUni FAQ data with clear source IDs.
- **Context injection**: Agent knows your role and unit throughout the session.

## 5) API Structure
- `GET /api/onboarding/setup-questions`: Initialize wizard.
- `POST /api/onboarding/initialize`: Create session & checklist.
- `GET /api/onboarding/checklist/{session_id}`: Track progress.
- `POST /api/chat`: Context-aware chat.
- `POST /api/actions/*`: Feedback & Escalation.

