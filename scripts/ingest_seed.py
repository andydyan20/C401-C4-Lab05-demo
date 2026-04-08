import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from packages.rag.seed_loader import load_seed


def main() -> None:
    seed_path = BASE_DIR / "data" / "vinuni_freshman_faq_seed.json"
    items = load_seed(seed_path)

    intents = sorted({item["intent"] for item in items})
    categories = sorted({item["category"] for item in items})

    print(f"[OK] Loaded {len(items)} seed items from: {seed_path}")
    print(f"[OK] Intents: {len(intents)} | Categories: {len(categories)}")

    output_summary = {
        "total_items": len(items),
        "intents": intents,
        "categories": categories,
    }
    summary_path = BASE_DIR / "data" / "seed_summary.json"
    summary_path.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote summary file: {summary_path}")


if __name__ == "__main__":
    main()
