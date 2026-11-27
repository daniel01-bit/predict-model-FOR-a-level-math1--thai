"""3-Tier Weighted Multi-Source Forecasting Model for A-Level Math 1."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.schema import DatasetTier, ExamQuestion, QuestionFormat, load_syllabus


CURRICULUM_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "curriculum" / "ipst_2560_curriculum.json"
)

ModelMode = Literal[
    "naive_uniform",
    "model_a_tier1",
    "model_b_tier1_tier2",
    "model_c_full_3tier",
]


@dataclass(frozen=True)
class ModelHyperparams:
    tier1_recency_decay: float = 0.15       # Mild decay within Tier 1 (64-69)
    rotation_gap_boost: float = 0.25        # Boost if seen in Tier 1 before, skipped last year
    tier2_uncovered_boost: float = 0.35     # Boost for IPST 2560 syllabus blind spots
    tier2_exercise_weight: float = 0.20     # Weight for parsed Tier 2 IPST exercises
    tier3_time_decay: float = 0.18          # Exponential decay lambda for legacy Saman/Ent
    tier3_base_weight: float = 0.35         # Base scale of Tier 3 relative to Tier 1
    smoothing_alpha: float = 0.10           # Laplace smoothing to prevent zero probabilities


@dataclass(frozen=True)
class ExamSlot:
    slot_number: int
    strand_id: str
    chapter_id: str
    sub_skill_id: str
    secondary_chapter: str | None
    question_format: QuestionFormat
    predicted_prob: float
    rationale: str


def load_curriculum(path: Path = CURRICULUM_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_no_leakage(questions: list[ExamQuestion], target_year_be: int) -> list[ExamQuestion]:
    """Strict chronological guard: drop any exam item from target_year_be or later."""
    filtered: list[ExamQuestion] = []
    for q in questions:
        if q.exam_year_be is not None and q.exam_year_be >= target_year_be:
            continue
        filtered.append(q)
    return filtered


def predict_subskill_distribution(
    questions: list[ExamQuestion],
    target_year_be: int,
    mode: ModelMode = "model_c_full_3tier",
    params: ModelHyperparams = ModelHyperparams(),
) -> dict[str, float]:
    """Return normalized probability distribution over all Syllabus 2560 sub-skills."""
    syllabus = load_syllabus()
    all_skills = sorted(syllabus.valid_sub_skill_ids())

    if mode == "naive_uniform":
        uniform_p = 1.0 / len(all_skills)
        return {sk: uniform_p for sk in all_skills}

    safe_qs = _filter_no_leakage(questions, target_year_be)
    tier1_qs = [q for q in safe_qs if q.tier == DatasetTier.TIER_1_TARGET]
    tier2_qs = [q for q in safe_qs if q.tier == DatasetTier.TIER_2_IPST]
    tier3_qs = [q for q in safe_qs if q.tier == DatasetTier.TIER_3_LEGACY]

    scores: dict[str, float] = {sk: params.smoothing_alpha for sk in all_skills}

    # --- Tier 1: Target distribution + recency + rotation gap ---
    tier1_skill_years: dict[str, set[int]] = defaultdict(set)
    latest_tier1_year = max((q.exam_year_be for q in tier1_qs if q.exam_year_be), default=None)

    for q in tier1_qs:
        assert q.exam_year_be is not None
        age = max(1, target_year_be - q.exam_year_be)
        w = math.exp(-params.tier1_recency_decay * (age - 1))
        per_skill = w / len(q.sub_skills)
        for sk in q.sub_skills:
            scores[sk] += per_skill
            tier1_skill_years[sk].add(q.exam_year_be)

    if latest_tier1_year is not None:
        for sk in all_skills:
            years_seen = tier1_skill_years.get(sk, set())
            # Appeared in earlier Tier 1 exams, skipped in the immediately preceding year -> rotation candidate
            if years_seen and latest_tier1_year not in years_seen:
                scores[sk] += params.rotation_gap_boost

    # --- Tier 2: IPST 2560 Curriculum Universe & Uncovered Syllabus Blind Spots ---
    if mode in ("model_b_tier1_tier2", "model_c_full_3tier"):
        # Uncovered syllabus signal: in IPST 2560 curriculum but 0 appearances in Tier 1 so far
        for sk in all_skills:
            if not tier1_skill_years.get(sk):
                scores[sk] += params.tier2_uncovered_boost

        # Parsed Tier 2 IPST textbook exercises (if ingested)
        if tier2_qs:
            t2_counts = Counter(sk for q in tier2_qs for sk in q.sub_skills)
            max_t2 = max(t2_counts.values(), default=1)
            for sk, cnt in t2_counts.items():
                scores[sk] += params.tier2_exercise_weight * (cnt / max_t2)

    # --- Tier 3: Legacy Saman 55-63 & Ent (Syllabus-2560 filtered + Time-Decay) ---
    if mode == "model_c_full_3tier" and tier3_qs:
        for q in tier3_qs:
            year = q.exam_year_be or (target_year_be - 10)
            age = max(1, target_year_be - year)
            w = params.tier3_base_weight * math.exp(-params.tier3_time_decay * age)
            per_skill = w / len(q.sub_skills)
            for sk in q.sub_skills:
                scores[sk] += per_skill

    total = sum(scores.values())
    return {sk: scores[sk] / total for sk in all_skills}


def compute_cross_topic_pairs(
    questions: list[ExamQuestion],
    target_year_be: int,
) -> dict[str, list[tuple[str, int]]]:
    """Return top co-occurring secondary chapters for each primary chapter."""
    safe_qs = _filter_no_leakage(questions, target_year_be)
    cooccur: dict[str, Counter[str]] = defaultdict(Counter)
    for q in safe_qs:
        weight = 3 if q.tier == DatasetTier.TIER_1_TARGET else 1
        for sec in q.secondary_chapters:
            if sec != q.primary_chapter:
                cooccur[q.primary_chapter][sec] += weight
    return {ch: cnt.most_common(3) for ch, cnt in cooccur.items()}


def build_trap_bank(
    questions: list[ExamQuestion],
    target_year_be: int,
) -> dict[str, list[str]]:
    """Combine historical key_trap observations with IPST 2560 classic_traps per sub-skill."""
    safe_qs = _filter_no_leakage(questions, target_year_be)
    traps: dict[str, list[str]] = defaultdict(list)

    for q in safe_qs:
        if q.key_trap:
            for sk in q.sub_skills:
                if q.key_trap not in traps[sk]:
                    traps[sk].append(q.key_trap)

    if CURRICULUM_PATH.exists():
        curr = load_curriculum()
        for ch in curr.get("chapters", []):
            for sd in ch.get("sub_skill_details", []):
                sk = sd["sub_skill_id"]
                for trap in sd.get("classic_traps", []):
                    if trap not in traps[sk]:
                        traps[sk].append(trap)
    return dict(traps)


def allocate_blueprint_slots(
    subskill_probs: dict[str, float],
    cross_topic_pairs: dict[str, list[tuple[str, int]]] | None = None,
) -> list[ExamSlot]:
    """Allocate exactly 30 A-Level Math 1 exam slots respecting MyTCAS strand bounds."""
    syllabus = load_syllabus()
    cross_topic_pairs = cross_topic_pairs or {}

    skill_to_chapter = {
        sk: ch.id for s in syllabus.strands for ch in s.chapters for sk in ch.sub_skills
    }
    strand_skills = {
        s.id: [sk for ch in s.chapters for sk in ch.sub_skills] for s in syllabus.strands
    }

    # Start each strand at its official minimum, then distribute remaining slots by strand probability mass
    strand_counts = {s.id: s.min_questions for s in syllabus.strands}
    strand_max = {s.id: s.max_questions for s in syllabus.strands}
    remaining = syllabus.total_questions - sum(strand_counts.values())

    strand_mass = {
        s.id: sum(subskill_probs.get(sk, 0.0) for sk in strand_skills[s.id])
        for s in syllabus.strands
    }

    while remaining > 0:
        candidates = [
            s_id for s_id in strand_counts if strand_counts[s_id] < strand_max[s_id]
        ]
        # Pick strand with highest marginal probability density per allocated slot
        best_strand = max(candidates, key=lambda sid: strand_mass[sid] / (strand_counts[sid] + 1))
        strand_counts[best_strand] += 1
        remaining -= 1

    # Inside each strand, pick top sub-skills (allowing repeat only if strand needs more slots than skills)
    selected: list[tuple[str, str, str, float]] = []
    for s in syllabus.strands:
        quota = strand_counts[s.id]
        ranked = sorted(
            strand_skills[s.id],
            key=lambda sk: subskill_probs.get(sk, 0.0),
            reverse=True,
        )
        # Encourage chapter diversity on first pass, then fill by highest prob
        chosen_skills: list[str] = []
        seen_chapters: set[str] = set()
        for sk in ranked:
            ch_id = skill_to_chapter[sk]
            if ch_id not in seen_chapters and len(chosen_skills) < quota:
                chosen_skills.append(sk)
                seen_chapters.add(ch_id)
        idx = 0
        while len(chosen_skills) < quota:
            cand = ranked[idx % len(ranked)]
            if cand not in chosen_skills or len(chosen_skills) >= len(ranked):
                chosen_skills.append(cand)
            idx += 1

        for sk in chosen_skills:
            selected.append((s.id, skill_to_chapter[sk], sk, subskill_probs.get(sk, 0.0)))

    # Order slots: 25 MCQ first (1-25), 5 Numeric last (26-30) choosing high-calculation sub-skills for numeric
    numeric_preferred_strands = {"calculus", "statistics_probability", "measurement_geometry", "number_algebra"}
    # Pick 5 numeric slots distributed across strands (at most 2 per strand)
    numeric_indices: list[int] = []
    strand_num_cnt: Counter[str] = Counter()
    for idx in sorted(range(len(selected)), key=lambda i: selected[i][3], reverse=True):
        sid = selected[idx][0]
        if len(numeric_indices) < 5 and sid in numeric_preferred_strands and strand_num_cnt[sid] < 2:
            numeric_indices.append(idx)
            strand_num_cnt[sid] += 1

    numeric_set = set(numeric_indices)
    mcq_items = [selected[i] for i in range(len(selected)) if i not in numeric_set]
    num_items = [selected[i] for i in numeric_indices]

    slots: list[ExamSlot] = []
    for slot_idx, (sid, ch_id, sk, prob) in enumerate(mcq_items + num_items, start=1):
        fmt = QuestionFormat.MCQ_5 if slot_idx <= 25 else QuestionFormat.NUMERIC
        sec_list = cross_topic_pairs.get(ch_id, [])
        sec_ch = sec_list[0][0] if sec_list else None
        slots.append(
            ExamSlot(
                slot_number=slot_idx,
                strand_id=sid,
                chapter_id=ch_id,
                sub_skill_id=sk,
                secondary_chapter=sec_ch,
                question_format=fmt,
                predicted_prob=round(prob, 5),
                rationale=f"Predicted p={prob:.4f} within MyTCAS strand '{sid}'",
            )
        )
    return slots
