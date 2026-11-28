"""Unit tests for curriculum alignment, 3-Tier model, walk-forward backtest, and SymPy verifier."""

from collections import Counter

from src.backtest import run_walk_forward_ablation
from src.model import (
    allocate_blueprint_slots,
    load_curriculum,
    predict_subskill_distribution,
)
from src.schema import (
    DatasetTier,
    DifficultyLevel,
    ExamQuestion,
    QuestionFormat,
    QuestionStyle,
    load_syllabus,
)
from src.verifier import sanitize_choice_text, sanitize_latex_text, verify_generated_question


def _make_q(qid: str, tier: DatasetTier, year: int | None, ch: str, skill: str) -> ExamQuestion:
    return ExamQuestion(
        question_id=qid,
        tier=tier,
        source_name=f"Synthetic {year}",
        exam_year_be=year,
        question_number=1,
        question_format=QuestionFormat.MCQ_5,
        question_style=QuestionStyle.WORD_PROBLEM,
        difficulty=DifficultyLevel.MEDIUM,
        primary_chapter=ch,
        secondary_chapters=[],
        sub_skills=[skill],
        stem_latex=f"โจทย์ทดสอบสำหรับ {qid} ในบท {ch}",
        choices_latex=["1", "2", "3", "4", "5"],
        answer="2",
    )


def test_curriculum_json_matches_closed_taxonomy_100_percent():
    syllabus = load_syllabus()
    curr = load_curriculum()
    curr_chapters = {c["chapter_id"] for c in curr["chapters"]}
    curr_skills = {
        sd["sub_skill_id"] for c in curr["chapters"] for sd in c["sub_skill_details"]
    }
    assert curr_chapters == syllabus.valid_chapter_ids()
    assert curr_skills == syllabus.valid_sub_skill_ids()


def test_zero_leakage_excludes_target_and_future_years():
    q_past = _make_q("t1_67", DatasetTier.TIER_1_TARGET, 2567, "calculus", "limit_continuity")
    q_future = _make_q("t1_69", DatasetTier.TIER_1_TARGET, 2569, "sets", "power_set_subset")

    p_without_future = predict_subskill_distribution([q_past], target_year_be=2568, mode="model_a_tier1")
    p_with_future = predict_subskill_distribution([q_past, q_future], target_year_be=2568, mode="model_a_tier1")
    assert p_without_future == p_with_future


def test_blueprint_allocator_respects_exact_mytcas_bounds():
    syllabus = load_syllabus()
    probs = predict_subskill_distribution([], target_year_be=2570, mode="model_b_tier1_tier2")
    slots = allocate_blueprint_slots(probs)

    assert len(slots) == 30
    assert sum(1 for s in slots if s.question_format == QuestionFormat.MCQ_5) == 25
    assert sum(1 for s in slots if s.question_format == QuestionFormat.NUMERIC) == 5

    strand_counts = Counter(s.strand_id for s in slots)
    for strand in syllabus.strands:
        assert strand.min_questions <= strand_counts[strand.id] <= strand.max_questions


def test_walk_forward_ablation_runs_all_four_models():
    qs = [
        _make_q("t1_66_1", DatasetTier.TIER_1_TARGET, 2566, "calculus", "extrema_optimization_word_problem"),
        _make_q("t1_67_1", DatasetTier.TIER_1_TARGET, 2567, "calculus", "extrema_optimization_word_problem"),
        _make_q("t1_68_1", DatasetTier.TIER_1_TARGET, 2568, "calculus", "extrema_optimization_word_problem"),
        _make_q("t1_69_1", DatasetTier.TIER_1_TARGET, 2569, "calculus", "extrema_optimization_word_problem"),
        _make_q("t3_60_1", DatasetTier.TIER_3_LEGACY, 2560, "calculus", "extrema_optimization_word_problem"),
    ]
    report = run_walk_forward_ablation(qs, test_years_be=[2568, 2569])
    assert set(report["summary"].keys()) == {
        "naive_uniform",
        "model_a_tier1",
        "model_b_tier1_tier2",
        "model_c_full_3tier",
    }
    assert report["beats_naive_baseline"] is True


def test_sympy_verifier_accepts_correct_and_rejects_wrong_or_duplicate():
    ok_res = verify_generated_question(
        stem_latex="จงหาค่าของ $x$ เมื่อ $2x + 6 = 14$",
        question_format=QuestionFormat.MCQ_5,
        choices_latex=["2", "4", "6", "8", "10"],
        expected_answer="2",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(2*x + 6 - 14, x)[0]",
    )
    assert ok_res.passed is True
    assert float(ok_res.computed_answer) == 4.0

    bad_math = verify_generated_question(
        stem_latex="จงหาค่าของ $x$ เมื่อ $2x + 6 = 14$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="99",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(2*x + 6 - 14, x)[0]",
    )
    assert bad_math.passed is False

    dup_stem = verify_generated_question(
        stem_latex="จงหาค่าของ $x$ เมื่อ $2x + 6 = 14$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="4",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(2*x + 6 - 14, x)[0]",
        historical_stems=["จงหาค่าของ $x$ เมื่อ $2x + 6 = 14$"],
    )
    assert dup_stem.passed is False
    assert "similar" in (dup_stem.error or "").lower()

    latex_frac_res = verify_generated_question(
        stem_latex="จงหาค่าของ $\\frac{3}{2}$",
        question_format=QuestionFormat.MCQ_5,
        choices_latex=["$1$", "$\\frac{3}{2}$", "$2$", "$\\frac{5}{2}$", "$3$"],
        expected_answer="2",
        sympy_code="ANSWER = sp.Rational(3, 2)",
    )
    assert latex_frac_res.passed is True


def test_audit_regression_latex_sanitizer_and_quality_gates():
    # JSON-eaten \times (\t + imes) and \frac (\f + rac) must be repaired cleanly
    assert sanitize_latex_text("เมทริกซ์ขนาด 3\times3 และ \frac{1}{6}") == r"เมทริกซ์ขนาด 3\times3 และ \frac{1}{6}"
    assert sanitize_choice_text(r"\mathbf{1}\text{.) } 12") == "12"

    # Proposition stem paired with bare 1..5 choices must be rejected
    bad_prop = verify_generated_question(
        stem_latex="ประพจน์ใดต่อไปนี้สมมูลกับ $p \\to q$",
        question_format=QuestionFormat.MCQ_5,
        choices_latex=["1", "2", "3", "4", "5"],
        expected_answer="2",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(x - 2, x)[0]",
    )
    assert bad_prop.passed is False
    assert "proposition" in (bad_prop.error or "").lower()

    # Hardcoded ANSWER without computation must be rejected
    hardcoded = verify_generated_question(
        stem_latex="จงหาค่า $x$ ที่ทำให้ $2x = 10$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="5",
        sympy_code="ANSWER = 5",
    )
    assert hardcoded.passed is False
    assert "hardcoded" in (hardcoded.error or "").lower()

    # Unbalanced $ delimiter must be rejected
    bad_dollar = verify_generated_question(
        stem_latex="กำหนดให้ $x = 5 จงหาค่า $2x$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="10",
        sympy_code="x = sp.Integer(5)\nANSWER = 2 * x",
    )
    assert bad_dollar.passed is False
    assert "unbalanced" in (bad_dollar.error or "").lower()

    # Interval, set, boolean, and mathematically duplicate choices must be rejected
    for bad_choices, expected_kw in [
        (["$(1, 2)$", "$(2, \\infty)$", "$(3, 4)$", "$(4, 5)$", "$(5, 6)$"], "interval"),
        (["$\\{-2, 1\\}$", "$\\{0, 1\\}$", "$\\{1, 2\\}$", "$\\{2, 3\\}$", "$\\{3, 4\\}$"], "set notation"),
        (["จริง", "เท็จ", "1", "2", "3"], "boolean"),
        (["$0.5$", "$\\frac{1}{2}$", "$1$", "$2$", "$3$"], "mathematically equal"),
    ]:
        res = verify_generated_question(
            stem_latex="จงหาคำตอบที่ถูกต้องของสมการต่อไปนี้",
            question_format=QuestionFormat.MCQ_5,
            choices_latex=bad_choices,
            expected_answer="1",
            sympy_code="x = sp.Rational(1, 2)\nANSWER = x",
        )
        assert res.passed is False
        assert expected_kw in (res.error or "").lower()

    # AI hedging red-flag phrase and 3imes3 / rationalization leaks must be rejected
    red_flag_res = verify_generated_question(
        stem_latex="จงหาค่า $x$ ที่ทำให้ $2x = 10$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="5",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(2*x - 10, x)[0]",
        solution_steps_th="ปรับเวกเตอร์ใหม่เพื่อให้ได้ค่า 5",
    )
    assert red_flag_res.passed is False
    assert "red-flag" in (red_flag_res.error or "").lower()

    corrupted_times = verify_generated_question(
        stem_latex="เมทริกซ์ขนาด 3imes3 ที่มีค่า det เท่ากับ 5",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="5",
        sympy_code="x = sp.Integer(5)\nANSWER = x * 1",
    )
    assert corrupted_times.passed is False
    assert "garbled" in (corrupted_times.error or "").lower()

    # Broken \leq -> eq0, step-scaffolding in stem, and LLM repetition loop must be rejected
    corrupted_leq = verify_generated_question(
        stem_latex="จงหาจำนวนเต็มที่สอดคล้อง $x^2 - x - 2eq0$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="2",
        sympy_code="x = sp.Integer(2)\nANSWER = x * 1",
    )
    assert corrupted_leq.passed is False
    assert "garbled" in (corrupted_leq.error or "").lower()

    step_leak = verify_generated_question(
        stem_latex="ขั้นที่ 1 หาค่า $k$ จากสมการ ขั้นที่ 2 นำไปคำนวณ $2k$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="4",
        sympy_code="k = sp.Integer(2)\nANSWER = 2 * k",
    )
    assert step_leak.passed is False
    assert "scaffolding" in (step_leak.error or "").lower()

    loop_stem = "อสมการหลักคือ $x^2 - x - 2 \\le 0$ ตรงกับเงื่อนไข " * 5
    rep_res = verify_generated_question(
        stem_latex=loop_stem,
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="2",
        sympy_code="k = sp.Integer(2)\nANSWER = k * 1",
    )
    assert rep_res.passed is False
    assert "repetition loop" in (rep_res.error or "").lower()


def test_all_sympy_first_seeds_verify_cleanly():
    from src.sympy_seeds import _SEEDERS, generate_verified_seed
    from src.verifier import _parse_numeric, run_sympy_code

    syllabus = load_syllabus()
    assert set(_SEEDERS.keys()) == syllabus.valid_chapter_ids()

    for ch_id in _SEEDERS:
        for lv in (1, 2, 3):
            seed = generate_verified_seed(ch_id, seed=42, difficulty_level=lv)
            assert seed is not None
            ok, computed = run_sympy_code(seed.sympy_code)
            assert ok, f"Seed {ch_id} lv={lv} failed: {computed}"
            expected_num = _parse_numeric(seed.exact_answer)
            assert expected_num is not None
            assert abs(float(computed) - expected_num) < 1e-4

    # Verify non-repeating seed archetypes inside the same exam for calculus & trigonometry
    used_calc: set[str] = set()
    c_archs = []
    for i in range(3):
        s = generate_verified_seed("calculus", seed=10 + i, exclude_archetypes=used_calc)
        assert s is not None and s.archetype_id
        used_calc.add(s.archetype_id)
        c_archs.append(s.archetype_id)
    assert len(set(c_archs)) == 3

    # Verify new audit regression gates: Lao script, empty arctan(), '1.) ' choice prefix
    assert sanitize_choice_text("1.) 2.0") == "2.0"
    lao_res = verify_generated_question(
        stem_latex="ຄ້າ กำหนดให้ $x = 5$ จงหา $2x$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="10",
        sympy_code="x = sp.Integer(5)\nANSWER = 2 * x",
    )
    assert lao_res.passed is False
    assert "lao" in (lao_res.error or "").lower()

    empty_arg = verify_generated_question(
        stem_latex="กำหนดให้ $\\theta = \\arctan( ) + \\arctan(1/3)$ จงหา $\\tan(\\theta)$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="1",
        sympy_code="ANSWER = sp.Integer(1) + sp.Integer(0)",
    )
    assert empty_arg.passed is False
    assert "garbled" in (empty_arg.error or "").lower()


def test_global_sympy_assertions_block_underdetermined_infinite_and_invalid_domain():
    # 1) Under-determined free symbol (like Q27 polynomial) must be blocked
    free_sym = verify_generated_question(
        stem_latex="จงหาค่า $P(4)$ ของพหุนามดีกรีสาม",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="12",
        sympy_code="a = sp.Symbol('a')\nANSWER = 6*a + 12",
    )
    assert free_sym.passed is False
    assert "free symbols" in (free_sym.error or "").lower()

    # 2) Empty solution / infinity / ambiguous multi-root must be blocked
    empty_sol = verify_generated_question(
        stem_latex="จงหาค่า $x$ ที่สอดคล้องระบบสมการ",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="1",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve([sp.Eq(x, 8), sp.Eq(x, 11)], x)",
    )
    assert empty_sol.passed is False

    inf_sol = verify_generated_question(
        stem_latex="จงหาค่าลิมิตของฟังก์ชัน",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="0",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.limit(1/x, x, 0, dir='+')",
    )
    assert inf_sol.passed is False
    assert "non-finite" in (inf_sol.error or "").lower()

    # 3) Negative counting / negative area must be blocked
    neg_count = verify_generated_question(
        stem_latex="จงหาจำนวนวิธีจัดเรียงทั้งหมด",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="-5",
        sympy_code="ANSWER = sp.Integer(2) - sp.Integer(7)",
        chapter_id="counting",
    )
    assert neg_count.passed is False
    assert "non-negative integer" in (neg_count.error or "").lower()

    neg_area = verify_generated_question(
        stem_latex="จงหาพื้นที่ปิดล้อมด้วยเส้นโค้งนี้",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="-4",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.integrate(-2, (x, 0, 2))",
        chapter_id="calculus",
    )
    assert neg_area.passed is False
    assert "cannot be negative" in (neg_area.error or "").lower()

    # 4) Part 2 (NUMERIC grid-in) strict A-Level bubble-sheet bounds: 0 <= ans <= 9999.99, <= 2 decimals, no '/'
    neg_part2 = verify_generated_question(
        stem_latex="จงหาค่าของ $x$ จากสมการ $x + 7 = 0$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="-7",
        sympy_code="x = sp.Symbol('x')\nANSWER = sp.solve(x + 7, x)[0]",
    )
    assert neg_part2.passed is False
    assert "negative" in (neg_part2.error or "").lower()

    frac_part2 = verify_generated_question(
        stem_latex="จงหาความน่าจะเป็นของเหตุการณ์นี้",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="3/16",
        sympy_code="ANSWER = sp.Rational(3, 16) * sp.Integer(1)",
    )
    assert frac_part2.passed is False
    assert "fraction" in (frac_part2.error or "").lower()

    dec3_part2 = verify_generated_question(
        stem_latex="จงหาค่าของ $1/8$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="0.125",
        sympy_code="ANSWER = sp.Rational(1, 8) + sp.Integer(0)",
    )
    assert dec3_part2.passed is False
    assert "2 decimal" in (dec3_part2.error or "").lower()

    over_part2 = verify_generated_question(
        stem_latex="จงหาค่าของ $10000 + 5$",
        question_format=QuestionFormat.NUMERIC,
        choices_latex=[],
        expected_answer="10005",
        sympy_code="ANSWER = sp.Integer(10000) + sp.Integer(5)",
    )
    assert over_part2.passed is False
    assert "9999.99" in (over_part2.error or "")

    from src.diversity import load_few_shot_examples, sample_diversity_directive

    for ch in ("sequences_series", "functions", "analytic_geometry", "vectors_3d"):
        assert len(load_few_shot_examples(ch, k=2)) >= 2
    used: set[str] = set()
    a1, _ = sample_diversity_directive("functions", used_archetypes=used)
    a2, _ = sample_diversity_directive("functions", used_archetypes=used)
    assert a1 != a2

    # 5) Thai char-5gram & exact-hash Anti-Duplicate Registry: similar allowed, exact/clone blocked
    import tempfile
    from pathlib import Path
    from src.dedup import GeneratedStemRegistry

    with tempfile.TemporaryDirectory() as td:
        reg = GeneratedStemRegistry(history_path=Path(td) / "hist.jsonl")
        q1 = "กำหนดวงรีมีสมการ $(x - 2)^2 / 25 + (y + 1)^2 / 16 = 1$ จงหาความยาวรอบรูปของสามเหลี่ยม $PF_1F_2$"
        q_exact = "กำหนดวงรีมีสมการ $(x-2)^2/25 + (y+1)^2/16 = 1$ จงหาความยาวรอบรูปของสามเหลี่ยม $PF_1F_2$"
        q_similar = "โรงงานออกแบบหลังคาทรงวงรี $(x + 3)^2 / 169 + (y - 4)^2 / 144 = 1$ จงหาระยะห่างระหว่างจุดโฟกัสและแกนเอก"
        assert reg.check_duplicate(q1) is None
        reg.register(q1, "analytic_geometry", "conics")
        assert reg.check_duplicate(q_exact) is not None  # exact clone blocked!
        assert reg.check_duplicate(q_similar) is None     # conceptually similar with different context/numbers allowed!
