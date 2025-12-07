from typing import Any
import math
import numpy as np
from sentence_transformers import SentenceTransformer


class Page:
    """Information about the page and its context."""

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self, name: str, depth: int) -> None:
        self.name = name
        self.depth = depth
        self.content: dict[str, Any] | None = None
        self.novelty: float = 1

        emb = self.model.encode(name, convert_to_numpy=True, show_progress_bar=False)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        self.name_emb_normalized = emb.astype(np.float32)

    def retrieve(self, api) -> None:
        """Request the page carefully from API."""
        try:
            self.content = api.fetch_page(self.name)
        except Exception:
            return None

    def get_name_rank(self) -> float:
        """Predict the page's quality by its name."""
        name = self.name.lower()

        # name lenght score (the shorter - the better)
        score = 200 / (len(name) + 5)

        # bonuses for important words
        important_keywords = {
            "history": 20,
            "science": 20,
            "mathematics": 20,
            "physics": 18,
            "biology": 18,
            "technology": 18,
            "philosophy": 15,
            "art": 12,
            "culture": 12,
            "music": 10,
            "computer": 10,
        }

        for key, bonus in important_keywords.items():
            if key in name:
                score += bonus

        # bonus for a single-word titles
        if len(name.split()) == 1:
            score += 15

        # bonus for the lack of parenthesis
        if "(" not in name:
            score += 10

        # penalty for the deep page source
        score *= max(0.5, 1 - (self.depth - 1) * 0.1)

        return max(score, 1)

    def get_rank(self) -> float:
        """Estimate the quality of the page."""
        if not self.content:
            return 0
        score = 0
        score += math.log(len(self.content["content"]) + 10)
        score += math.sqrt(len(self.content["links"]))
        score += max(0, 20 - 3 * self.depth)
        score += self.get_name_rank() / 5
        return score
