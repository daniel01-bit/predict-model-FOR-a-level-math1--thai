"""Unified CLI entrypoint for Predict A-Level Math 1 (Backtest, Forecast, and v3.0 Mock Generation)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.backtest import load_jsonl_dataset, run_walk_forward_ablation
from src.generator import generate_mock_exam
from src.model import (
    ModelMode,
    allocate_blueprint_slots,
    build_trap_bank,
    compute_cross_topic_pairs,
    predict_subskill_distribution,
)
from src.schema import DatasetTier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def _default_dataset_paths() -> list[Path]:
    return sorted(PARSED_DIR.glob("*.jsonl"))


def render_forecast_markdown(
    target_year_be: int,
    mode: ModelMode,
    probs: dict[str, float],
    slots: list,
    cross_pairs: dict[str, list[tuple[str, int]]],
    n_questions: int,
) -> str:
    lines = [
        f"# A-Level Math 1 (ปี {target_year_be}) Forecast Report",
        f"- **Model Mode:** `{mode}`",
        f"- **Training Questions Loaded:** `{n_questions}`",
        "",
        "## Top 15 High-Probability Sub-Skills",
        "| Rank | Sub-Skill ID | Predicted Prob |",
        "| :--- | :--- | :--- |",
    ]
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:15]
    for idx, (sk, p) in enumerate(ranked, start=1):
        lines.append(f"| {idx} | `{sk}` | `{p:.4f}` |")

    lines.extend(
        [
            "",
            "## Predicted 30-Slot Blueprint Allocation (MyTCAS-Compliant)",
            "| ข้อ | สาระ | บทหลัก | ทักษะย่อย | บทบูรณาการ | รูปแบบ | Prob |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for s in slots:
        lines.append(
            f"| {s.slot_number} | `{s.strand_id}` | `{s.chapter_id}` | `{s.sub_skill_id}` | `{s.secondary_chapter or '-'}` | `{s.question_format.value}` | `{s.predicted_prob:.4f}` |"
        )

    if cross_pairs:
        lines.extend(["", "## Top Cross-Topic Fusion Pairs"])
        for ch, pairs in sorted(cross_pairs.items()):
            pair_str = ", ".join(f"`{sec}` ({w})" for sec, w in pairs)
            lines.append(f"- **`{ch}`** + {pair_str}")

    return "\n".join(lines) + "\n"


def cmd_backtest(args: argparse.Namespace) -> int:
    questions = load_jsonl_dataset(_default_dataset_paths())
    if not questions:
        print("No parsed JSONL data found in data/parsed/. Ingest Tier 1 PDFs first.")
        return 1
    report = run_walk_forward_ablation(questions)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ablation_backtest.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Winner: {report['winner_mode']} | Beats Baseline: {report['beats_naive_baseline']}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    questions = load_jsonl_dataset(_default_dataset_paths())
    mode: ModelMode = args.mode
    if mode == "auto":
        tier1_years = {
            q.exam_year_be
            for q in questions
            if q.tier == DatasetTier.TIER_1_TARGET and q.exam_year_be is not None
        }
        if len(tier1_years) >= 2:
            ablation = run_walk_forward_ablation(questions)
            mode = ablation["winner_mode"]
        else:
            mode = "model_b_tier1_tier2"

    probs = predict_subskill_distribution(questions, target_year_be=args.target_year, mode=mode)
    cross_pairs = compute_cross_topic_pairs(questions, target_year_be=args.target_year)
    slots = allocate_blueprint_slots(probs, cross_topic_pairs=cross_pairs)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / f"forecast_{args.target_year}.json"
    out_md = REPORTS_DIR / f"forecast_{args.target_year}.md"

    payload = {
        "target_year_be": args.target_year,
        "mode": mode,
        "training_questions_count": len(questions),
        "subskill_probs": probs,
        "cross_topic_pairs": cross_pairs,
        "slots": [asdict(s) for s in slots],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(
        render_forecast_markdown(args.target_year, mode, probs, slots, cross_pairs, len(questions)),
        encoding="utf-8",
    )
    print(f"Saved forecast ({mode}) -> {out_json} and {out_md}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    questions = load_jsonl_dataset(_default_dataset_paths())
    mode: ModelMode = args.mode if args.mode != "auto" else "model_b_tier1_tier2"
    probs = predict_subskill_distribution(questions, target_year_be=args.target_year, mode=mode)
    cross_pairs = compute_cross_topic_pairs(questions, target_year_be=args.target_year)
    slots = allocate_blueprint_slots(probs, cross_topic_pairs=cross_pairs)
    traps = build_trap_bank(questions, target_year_be=args.target_year)
    historical_stems = [q.stem_latex for q in questions]

    out_json = REPORTS_DIR / f"alevel_{args.target_year}_mock.json"
    out_md = REPORTS_DIR / f"alevel_{args.target_year}_mock.md"
    items = generate_mock_exam(
        slots=slots,
        trap_bank=traps,
        historical_stems=historical_stems,
        output_json=out_json,
        output_md=out_md,
        limit=args.limit,
        model_name=args.model,
    )
    print(f"Generated & SymPy-verified {len(items)} questions -> {out_md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict A-Level Math 1 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("backtest", help="Run walk-forward ablation backtest")
    p_bt.set_defaults(func=cmd_backtest)

    p_fc = sub.add_parser("forecast", help="Generate A-Level forecast & 30-slot Blueprint plan")
    p_fc.add_argument("--target-year", type=int, default=2570)
    p_fc.add_argument(
        "--mode",
        choices=["auto", "naive_uniform", "model_a_tier1", "model_b_tier1_tier2", "model_c_full_3tier"],
        default="auto",
    )
    p_fc.set_defaults(func=cmd_forecast)

    p_gen = sub.add_parser("generate", help="Generate v3.0 SymPy-verified mock exam")
    p_gen.add_argument("--target-year", type=int, default=2570)
    p_gen.add_argument("--limit", type=int, default=30, help="Number of questions to generate (default 30)")
    p_gen.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="Primary Gemini model")
    p_gen.add_argument(
        "--mode",
        choices=["auto", "model_a_tier1", "model_b_tier1_tier2", "model_c_full_3tier"],
        default="auto",
    )
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
