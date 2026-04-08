import re
from dataclasses import dataclass
from typing import Any


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


@dataclass
class RetrievalHit:
    item: dict[str, Any]
    score: float


class SeedRetriever:
    """Simple deterministic retriever for local MVP tests."""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        self._indexed = []
        for item in items:
            source_text = " ".join(
                [
                    item.get("question", ""),
                    " ".join(item.get("keywords", [])),
                    item.get("intent", ""),
                    item.get("category", ""),
                ]
            )
            self._indexed.append((item, _tokenize(source_text)))

    def search(self, query: str, role: str = "freshman_student", top_k: int = 3) -> list[RetrievalHit]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        hits: list[RetrievalHit] = []
        for item, item_tokens in self._indexed:
            overlap = len(q_tokens.intersection(item_tokens))
            if overlap == 0:
                continue

            priority_weight = {"high": 0.3, "medium": 0.15, "low": 0.05}.get(item.get("priority", "low"), 0.05)
            score = overlap / max(len(q_tokens), 1) + priority_weight
            hits.append(RetrievalHit(item=item, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
