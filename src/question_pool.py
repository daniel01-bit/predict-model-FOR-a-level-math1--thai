"""Persistent SymPy-verified question pool for instant exam assembly (Option C)."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from pathlib import Path
from typing import Iterable

from src.schema import QuestionFormat
from src.verifier import verify_generated_question


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL_PATH = PROJECT_ROOT / "data" / "verified_pool.jsonl"


def _is_strictly_valid_pool_item(item: dict) -> bool:
    """Re-verify every cached/imported question against the latest SymPy + LaTeX + Red-Flag gates."""
    if not isinstance(item, dict) or not item.get("verified_sympy"):
        return False
    sympy_code = str(item.get("sympy_code") or "").strip()
    stem = str(item.get("stem_latex") or "").strip()
    answer = str(item.get("answer") or "").strip()
    fmt_raw = item.get("question_format")
    if not sympy_code or not stem or not answer or fmt_raw not in ("mcq_5", "numeric"):
        return False
    try:
        res = verify_generated_question(
            stem_latex=stem,
            question_format=QuestionFormat(fmt_raw),
            choices_latex=list(item.get("choices_latex") or []),
            expected_answer=answer,
            sympy_code=sympy_code,
            historical_stems=None,
            solution_steps_th=str(item.get("solution_steps_th") or ""),
            chapter_id=item.get("chapter_id"),
        )
        return res.passed
    except Exception:
        return False


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

    def _rewrite_disk_unlocked(self) -> None:
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(rec, ensure_ascii=False) for rec in self._items.values()]
        self.pool_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.pool_path.exists():
            return
        pruned = 0
        for line in self.pool_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not _is_strictly_valid_pool_item(obj):
                    pruned += 1
                    continue
                qid = obj.get("pool_id") or _question_fingerprint(obj)
                obj["pool_id"] = qid
                self._items[qid] = obj
            except Exception:
                pruned += 1
                continue
        if pruned > 0:
            print(f"[pool] auto-pruned {pruned} legacy/invalid items from {self.pool_path.name} (kept {len(self._items)})", flush=True)
            self._rewrite_disk_unlocked()

    def add(self, item: dict) -> str:
        """Persist a strictly verified question into the pool (idempotent by stem fingerprint)."""
        if not _is_strictly_valid_pool_item(item):
            return ""
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
            if self.add(item):
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
