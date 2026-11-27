"""Walk-Forward Ablation Backtester (Zero Leakage) for A-Level Math 1."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from src.model import ModelHyperparams, ModelMode, predict_subskill_distribution
from src.schema import DatasetTier, ExamQuestion, load_syllabus


@dataclass(frozen=True)
class FoldMetrics:
    test_year_be: int
    mode: ModelMode
    recall_at_10: float
    recall_at_15: float
    brier_score: float
    log_loss: float
    actual_unique_skills: int


def evaluate_prediction(
    predicted_probs: dict[str, float],
    actual_questions: list[ExamQuestion],
    expected_hits: float = 25.0,
) -> tuple[float, float, float, float, int]:
    """Compute Recall@10, Recall@15, Brier Score, and Log-Loss against a test year's exam.

    `expected_hits` MUST come strictly from training folds (or a fixed prior), never from the test set.
    """
    syllabus = load_syllabus()
    all_skills = sorted(syllabus.valid_sub_skill_ids())
    actual_skills = {sk for q in actual_questions for sk in q.sub_skills}
    if not actual_skills:
        raise ValueError("actual_questions must contain at least one sub-skill")

    ranked = sorted(all_skills, key=lambda sk: predicted_probs.get(sk, 0.0), reverse=True)
    top10 = set(ranked[:10])
    top15 = set(ranked[:15])

    recall_10 = len(top10 & actual_skills) / min(10, len(actual_skills))
    recall_15 = len(top15 & actual_skills) / min(15, len(actual_skills))

    # Convert probability distribution to marginal presence probability using training-only expected_hits
    brier_sum = 0.0
    logloss_sum = 0.0
    eps = 1e-9
    for sk in all_skills:
        y = 1.0 if sk in actual_skills else 0.0
        p = min(1.0 - eps, max(eps, predicted_probs.get(sk, 0.0) * expected_hits))
        brier_sum += (p - y) ** 2
        logloss_sum += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

    n = len(all_skills)
    return (
        round(recall_10, 4),
        round(recall_15, 4),
        round(brier_sum / n, 5),
        round(logloss_sum / n, 5),
        len(actual_skills),
    )


def run_walk_forward_ablation(
    questions: list[ExamQuestion],
    test_years_be: list[int] | None = None,
    params: ModelHyperparams = ModelHyperparams(),
) -> dict:
    """Compare Naive Baseline vs Model A vs Model B vs Model C chronologically."""
    tier1_years = sorted(
        {
            q.exam_year_be
            for q in questions
            if q.tier == DatasetTier.TIER_1_TARGET and q.exam_year_be is not None
        }
    )
    if test_years_be is None:
        # Default: evaluate on the last 2 Tier 1 years (e.g. 2568, 2569)
        test_years_be = tier1_years[-2:] if len(tier1_years) >= 3 else tier1_years[-1:]

    modes: list[ModelMode] = [
        "naive_uniform",
        "model_a_tier1",
        "model_b_tier1_tier2",
        "model_c_full_3tier",
    ]

    folds: list[FoldMetrics] = []
    for yr in test_years_be:
        actual_qs = [
            q for q in questions if q.tier == DatasetTier.TIER_1_TARGET and q.exam_year_be == yr
        ]
        if not actual_qs:
            continue
        # Estimate expected active sub-skills strictly from past Tier 1 years (zero evaluation leakage)
        past_t1_years = sorted(
            {
                q.exam_year_be
                for q in questions
                if q.tier == DatasetTier.TIER_1_TARGET
                and q.exam_year_be is not None
                and q.exam_year_be < yr
            }
        )
        if past_t1_years:
            past_counts = [
                len(
                    {
                        sk
                        for q in questions
                        if q.tier == DatasetTier.TIER_1_TARGET and q.exam_year_be == py
                        for sk in q.sub_skills
                    }
                )
                for py in past_t1_years
            ]
            train_expected_hits = sum(past_counts) / len(past_counts)
        else:
            train_expected_hits = 25.0

        for mode in modes:
            probs = predict_subskill_distribution(
                questions=questions,
                target_year_be=yr,
                mode=mode,
                params=params,
            )
            r10, r15, brier, ll, n_act = evaluate_prediction(
                probs, actual_qs, expected_hits=train_expected_hits
            )
            folds.append(
                FoldMetrics(
                    test_year_be=yr,
                    mode=mode,
                    recall_at_10=r10,
                    recall_at_15=r15,
                    brier_score=brier,
                    log_loss=ll,
                    actual_unique_skills=n_act,
                )
            )

    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        m_folds = [f for f in folds if f.mode == mode]
        if not m_folds:
            continue
        summary[mode] = {
            "mean_recall_at_10": round(sum(f.recall_at_10 for f in m_folds) / len(m_folds), 4),
            "mean_recall_at_15": round(sum(f.recall_at_15 for f in m_folds) / len(m_folds), 4),
            "mean_brier_score": round(sum(f.brier_score for f in m_folds) / len(m_folds), 5),
            "mean_log_loss": round(sum(f.log_loss for f in m_folds) / len(m_folds), 5),
        }

    # Winner selection: lowest Brier Score (tie-break by highest Recall@15)
    winner = min(
        summary.keys(),
        key=lambda m: (summary[m]["mean_brier_score"], -summary[m]["mean_recall_at_15"]),
    ) if summary else "model_b_tier1_tier2"

    beats_baseline = (
        winner != "naive_uniform"
        and "naive_uniform" in summary
        and summary[winner]["mean_brier_score"] < summary["naive_uniform"]["mean_brier_score"]
    )

    return {
        "test_years_be": test_years_be,
        "winner_mode": winner,
        "beats_naive_baseline": beats_baseline,
        "summary": summary,
        "folds": [asdict(f) for f in folds],
    }


def load_jsonl_dataset(paths: list[Path]) -> list[ExamQuestion]:
    questions: list[ExamQuestion] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                questions.append(ExamQuestion.model_validate_json(line))
    return questions
