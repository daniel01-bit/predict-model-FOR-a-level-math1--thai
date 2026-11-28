"""Thread-safe Exact & Near-Exact Anti-Duplicate Registry for generated exams.

Allows conceptually similar questions (different numbers/context) while strictly
blocking exact or near-identical clones across:
1. Slots inside the same exam session (including parallel workers)
2. Previously generated exams persisted in data/reports/generated_history.jsonl

Uses Thai-safe character 5-gram Jaccard + normalized SHA-1 fingerprints because
whitespace tokenization fails on unspaced Thai sentences.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "reports" / "generated_history.jsonl"


def normalize_stem_for_dedup(text: str) -> str:
    """Normalize whitespace, dollar spacing, and punctuation while preserving all math numbers."""
    s = (text or "").lower().strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("$", "").replace(",", "").replace(";", "")
    return s


def stem_fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_stem_for_dedup(text).encode("utf-8")).hexdigest()[:20]


def char_ngrams(text: str, n: int = 5) -> set[str]:
    norm = normalize_stem_for_dedup(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def thai_char_similarity(stem_a: str, stem_b: str, n: int = 5) -> float:
    ga, gb = char_ngrams(stem_a, n=n), char_ngrams(stem_b, n=n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


class GeneratedStemRegistry:
    """Persistent + in-memory registry preventing exact/clone questions across runs."""

    def __init__(self, history_path: Path = DEFAULT_HISTORY_PATH, max_history: int = 500) -> None:
        self.history_path = history_path
        self.max_history = max_history
        self._lock = threading.Lock()
        self._fingerprints: set[str] = set()
        self._recent_grams: list[tuple[str, set[str]]] = []
        self._loaded = False

    def _ensure_loaded_unlocked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.history_path.exists():
            return
        lines = [ln.strip() for ln in self.history_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines[-self.max_history :]:
            try:
                rec = json.loads(ln)
                stem = rec.get("stem_latex", "")
                fp = rec.get("fp") or stem_fingerprint(stem)
                if stem:
                    self._fingerprints.add(fp)
                    self._recent_grams.append((fp, char_ngrams(stem)))
            except Exception:
                continue

    def check_duplicate(
        self,
        stem_latex: str,
        session_stems: list[str] | None = None,
        exact_history_threshold: float = 0.82,
        session_threshold: float = 0.72,
    ) -> str | None:
        """Return an error reason if stem is an exact/near-exact duplicate, else None."""
        fp = stem_fingerprint(stem_latex)
        grams = char_ngrams(stem_latex)

        # 1) Check against questions already accepted in THIS exam session
        if session_stems:
            for prev in session_stems:
                if stem_fingerprint(prev) == fp:
                    return "Exact duplicate of another question in this same exam"
                sim = thai_char_similarity(stem_latex, prev)
                if sim >= session_threshold:
                    return f"Near-exact duplicate inside same exam (char-5gram={sim:.2f} >= {session_threshold})"

        # 2) Check against previously generated exams on disk (similar allowed, exact/clone forbidden)
        with self._lock:
            self._ensure_loaded_unlocked()
            if fp in self._fingerprints:
                return "Exact duplicate of a previously generated question"
            for _, old_grams in self._recent_grams:
                if not grams or not old_grams:
                    continue
                sim = len(grams & old_grams) / len(grams | old_grams)
                if sim >= exact_history_threshold:
                    return f"Near-exact clone of a previously generated question (char-5gram={sim:.2f} >= {exact_history_threshold})"
        return None

    def register(self, stem_latex: str, chapter_id: str = "", sub_skill_id: str = "") -> None:
        """Atomically record a newly accepted question stem so future slots/runs never repeat it."""
        fp = stem_fingerprint(stem_latex)
        grams = char_ngrams(stem_latex)
        with self._lock:
            self._ensure_loaded_unlocked()
            if fp in self._fingerprints:
                return
            self._fingerprints.add(fp)
            self._recent_grams.append((fp, grams))
            if len(self._recent_grams) > self.max_history:
                self._recent_grams = self._recent_grams[-self.max_history :]
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"fp": fp, "chapter_id": chapter_id, "sub_skill_id": sub_skill_id, "stem_latex": stem_latex}
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


GLOBAL_STEM_REGISTRY = GeneratedStemRegistry()
