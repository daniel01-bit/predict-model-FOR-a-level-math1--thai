"""Persistent SymPy-verified question pool for instant exam assembly (Option C)."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL_PATH = PROJECT_ROOT / "data" / "verified_pool.jsonl"


def _question_fingerprint(item: dict) -> str:
    stem = (item.get("stem_latex") or "").strip().lower()
    raw = f"{item.get('sub_skill_id','')}|{item.get('question_format','')}|{stem}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class VerifiedQuestionPool:
    """Thread-safe JSONL store of SymPy-verified questions with non-repeating checkout."""

    def __init__(self, pool_path: Path = DEFAULT_POOL_PATH) -> None:
        self.pool_path = pool_path
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._served_ids: set[str] = set()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.pool_path.exists():
            return
        for line in self.pool_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not obj.get("verified_sympy"):
                    continue
                qid = obj.get("pool_id") or _question_fingerprint(obj)
                obj["pool_id"] = qid
                self._items[qid] = obj
            except Exception:
                continue

    def add(self, item: dict) -> str:
        """Persist a verified question into the pool (idempotent by stem fingerprint)."""
        with self._lock:
            self._ensure_loaded()
            rec = dict(item)
            qid = rec.get("pool_id") or _question_fingerprint(rec)
            rec["pool_id"] = qid
            is_new = qid not in self._items
            self._items[qid] = rec
            if is_new:
                self.pool_path.parent.mkdir(parents=True, exist_ok=True)
                with self.pool_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return qid

    def add_many(self, items: Iterable[dict]) -> int:
        added = 0
        for item in items:
            if item.get("verified_sympy"):
                self.add(item)
                added += 1
        return added

    def checkout(
        self,
        sub_skill_id: str,
        question_format: str,
        slot_number: int,
        exclude_ids: set[str] | None = None,
    ) -> dict | None:
        """Return an unused verified question matching (sub_skill_id, question_format), or None."""
        with self._lock:
            self._ensure_loaded()
            excluded = (exclude_ids or set()) | self._served_ids
            candidates = [
                q
                for qid, q in self._items.items()
                if qid not in excluded
                and q.get("sub_skill_id") == sub_skill_id
                and q.get("question_format") == question_format
            ]
            if not candidates:
                return None
            chosen = random.choice(candidates)
            self._served_ids.add(chosen["pool_id"])
            out = dict(chosen)
            out["slot_number"] = slot_number
            out["from_pool"] = True
            return out

    def count_available(self, sub_skill_id: str | None = None, question_format: str | None = None) -> int:
        with self._lock:
            self._ensure_loaded()
            return sum(
                1
                for qid, q in self._items.items()
                if qid not in self._served_ids
                and (sub_skill_id is None or q.get("sub_skill_id") == sub_skill_id)
                and (question_format is None or q.get("question_format") == question_format)
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._ensure_loaded()
            total = len(self._items)
            unused = sum(1 for qid in self._items if qid not in self._served_ids)
            return {"total": total, "unused": unused, "served": total - unused}

    def seed_from_jobs_dir(self, jobs_dir: Path) -> int:
        """Import previously SymPy-verified questions from completed job outputs into the pool."""
        if not jobs_dir.exists():
            return 0
        imported = 0
        for exam_json in sorted(jobs_dir.glob("*/exam.json")):
            try:
                items = json.loads(exam_json.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    imported += self.add_many(items)
            except Exception:
                continue
        return imported


GLOBAL_POOL = VerifiedQuestionPool()
