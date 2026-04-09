from typing import Any

SETUP_QUESTIONS = [
    {
        "id": "role",
        "text": "Bạn là tân sinh viên hay nhân viên mới?",
        "type": "select",
        "options": ["Tân sinh viên", "Nhân viên mới"]
    },
    {
        "id": "unit",
        "text": "Khoa hoặc Đơn vị của bạn là gì?",
        "type": "select",
        "options": ["CAS", "COB", "CECS", "CHS", "Phòng Hành chính", "Phòng IT"]
    },
    {
        "id": "term",
        "text": "Kỳ học / Đợt tuyển dụng của bạn?",
        "type": "select",
        "options": ["Fall 2024", "Spring 2025", "Đợt Q3/2024"]
    },
    {
        "id": "housing",
        "text": "Bạn có nhu cầu ở Ký túc xá không?",
        "type": "select",
        "options": ["Có", "Không"]
    }
]

CHECKLIST_TEMPLATES = {
    "Tân sinh viên": [
        {"id": "it_account", "title": "Kích hoạt tài khoản IT", "link": "https://it.vinuni.edu.vn/activate"},
        {"id": "tuition", "title": "Hoàn thành học phí", "link": "https://portal.vinuni.edu.vn/finance"},
        {"id": "insurance", "title": "Đăng ký bảo hiểm y tế", "link": "https://portal.vinuni.edu.vn/insurance"},
        {"id": "orientation", "title": "Tham gia Orientation Day", "link": "https://vinuni.edu.vn/orientation"},
    ],
    "Nhân viên mới": [
        {"id": "it_account", "title": "Thiết lập tài khoản nhân viên", "link": "https://it.vinuni.edu.vn/staff-setup"},
        {"id": "contract", "title": "Ký hợp đồng lao động", "link": "#"},
        {"id": "training", "title": "Hoàn thành đào tạo hội nhập", "link": "https://lms.vinuni.edu.vn/staff-onboarding"},
        {"id": "health_check", "title": "Khám sức khỏe đầu vào", "link": "#"},
    ]
}

HOUSING_TASK = {"id": "dorm", "title": "Hoàn thành thủ tục Ký túc xá", "link": "https://housing.vinuni.edu.vn"}

def get_checklist_for_answers(answers: dict[str, str]) -> list[dict[str, Any]]:
    role = answers.get("role")
    tasks = CHECKLIST_TEMPLATES.get(role, []).copy()
    
    if answers.get("housing") == "Có":
        tasks.append(HOUSING_TASK)
        
    return tasks
