"""Tests for QUESTION/MINI-TEST/MOCK-TEST slot builders, A-Level 61 PDF renderer, and FastAPI endpoints."""

from pathlib import Path

from src.generator import build_slots_for_mode
from src.llm_router import (
    build_openai_payload,
    clear_cooldowns,
    is_model_in_cooldown,
    mark_model_cooldown,
    recommend_batch_size,
    recommend_workers,
)
from src.model import predict_subskill_distribution
from src.pdf_renderer import render_exam_bundle
from src.schema import QuestionFormat
from src.server import api_providers, serve_index


def test_build_slots_for_all_three_modes():
    probs = predict_subskill_distribution([], target_year_be=2570, mode="model_b_tier1_tier2")

    q_slots = build_slots_for_mode("question", probs, seed=42)
    assert len(q_slots) == 1

    mini_slots = build_slots_for_mode("mini_test", probs, seed=42)
    assert len(mini_slots) == 15
    assert len({s.chapter_id for s in mini_slots}) == 15
    assert sum(1 for s in mini_slots if s.question_format == QuestionFormat.MCQ_5) == 12
    assert sum(1 for s in mini_slots if s.question_format == QuestionFormat.NUMERIC) == 3

    mock_slots = build_slots_for_mode("mock_test", probs, seed=42)
    assert len(mock_slots) == 30
    assert sum(1 for s in mock_slots if s.question_format == QuestionFormat.MCQ_5) == 25
    assert sum(1 for s in mock_slots if s.question_format == QuestionFormat.NUMERIC) == 5


def test_render_authentic_alevel_61_pdf_bundle(tmp_path: Path):
    sample_items = [
        {
            "slot_number": 1,
            "chapter_id": "sequences_series",
            "chapter_th": "ลำดับและอนุกรม",
            "sub_skill_id": "financial_math_compound_annuity",
            "question_format": "mcq_5",
            "difficulty_stars": 4,
            "stem_latex": "ฝากเงินต้น $10{,}000$ บาท อัตราดอกเบี้ย $2\\%$ ต่อปี คิดดอกเบี้ยทบต้นทุกปี ครบ $2$ ปีจะมีเงินรวมกี่บาท",
            "choices_latex": ["$10{,}200$", "$10{,}404$", "$10{,}600$", "$10{,}800$", "$11{,}000$"],
            "answer": "2",
            "sympy_computed_answer": "10404.0",
            "solution_steps_th": "ใช้สูตรดอกเบี้ยทบต้น $S = P(1+r)^n = 10000(1.02)^2 = 10404$",
            "key_trap": "ใช้ดอกเบี้ยเชิงเดียวแทนดอกเบี้ยทบต้น",
        },
        {
            "slot_number": 2,
            "chapter_id": "calculus",
            "chapter_th": "แคลคูลัสเบื้องต้น",
            "sub_skill_id": "differentiation_tangent_rate",
            "question_format": "numeric",
            "difficulty_stars": 5,
            "stem_latex": "กำหนด $f(x) = x^3 - 3x^2 + 5$ จงหาความชันของเส้นสัมผัสที่ $x = 3$",
            "choices_latex": [],
            "answer": "9",
            "sympy_computed_answer": "9.0",
            "solution_steps_th": "$f'(x) = 3x^2 - 6x$ ดังนั้น $f'(3) = 27 - 18 = 9$",
            "key_trap": "แทนค่า $x=3$ ลงใน $f(x)$ แทน $f'(x)$",
        },
    ]
    bundle = render_exam_bundle(sample_items, mode="mini_test", target_year_be=2570, out_dir=tmp_path)

    for key in ("exam_pdf", "solutions_pdf"):
        pdf_path = bundle[key]
        assert pdf_path.exists()
        raw = pdf_path.read_bytes()
        assert raw.startswith(b"%PDF-")
        assert len(raw) > 5000

    exam_html = bundle["exam_html"].read_text(encoding="utf-8")
    assert "รหัสวิชา 61" in exam_html
    assert "★★★★☆" in exam_html
    assert "★★★★★" in exam_html


def test_server_index_and_providers_contract():
    html_resp = serve_index()
    assert html_resp.status_code == 200
    body = html_resp.body.decode("utf-8")
    assert "QUESTION" in body
    assert "MINI-TEST" in body
    assert "MOCK-TEST" in body

    providers = api_providers()["providers"]
    ids = {p["id"] for p in providers}
    assert {"gemini", "openai", "anthropic", "openrouter", "ollama"} <= ids


def test_verified_question_pool_non_repeating_checkout(tmp_path: Path):
    from src.question_pool import VerifiedQuestionPool

    pool = VerifiedQuestionPool(pool_path=tmp_path / "pool.jsonl")
    sample = {
        "slot_number": 9,
        "chapter_id": "calculus",
        "sub_skill_id": "differentiation_tangent_rate",
        "question_format": "mcq_5",
        "verified_sympy": True,
        "stem_latex": "จงหาความชันของเส้นโค้ง $y = x^2$ ที่ $x = 3$",
        "choices_latex": ["2", "4", "6", "8", "10"],
        "answer": "3",
        "sympy_code": "x = sp.Symbol('x')\nANSWER = sp.diff(x**2, x).subs(x, 3)",
        "solution_steps_th": "หาอนุพันธ์ $y' = 2x$ แทน $x=3$ ได้ $6$",
    }
    poisoned = dict(sample, stem_latex="เมทริกซ์ขนาด 3imes3 จงหาคำตอบ", solution_steps_th="ขออภัย ปรับเวกเตอร์ใหม่เพื่อให้ลงตัว")
    assert pool.add(poisoned) == ""
    pool.add(sample)
    pool.add(sample)  # idempotent duplicate
    assert pool.stats() == {"total": 1, "unused": 1, "served": 0}

    hit = pool.checkout("differentiation_tangent_rate", "mcq_5", slot_number=1)
    assert hit is not None
    assert hit["slot_number"] == 1
    assert hit["from_pool"] is True
    assert pool.checkout("differentiation_tangent_rate", "mcq_5", slot_number=2) is None
    assert pool.stats() == {"total": 1, "unused": 0, "served": 1}


def test_job_cancellation_immediate_signal():
    import threading
    import pytest
    from src.llm_router import JobCancelledError, check_cancelled
    from src.server import JOBS, JobState, api_cancel_job, serve_index

    job = JobState(
        job_id="test_cancel_1",
        mode="question",
        provider="ollama",
        model="qwen3.5:4b",
        target_year_be=2570,
        status="running",
    )
    JOBS[job.job_id] = job
    try:
        res = api_cancel_job(job.job_id)
        assert res["ok"] is True
        assert res["status"] == "cancelled"
        assert job.cancel_event.is_set()
        with pytest.raises(JobCancelledError):
            check_cancelled(job.cancel_event)
    finally:
        JOBS.pop(job.job_id, None)

    body = serve_index().body.decode("utf-8")
    assert "cancelCurrentJob()" in body


def test_sympy_error_formatter_includes_chapter_and_reason():
    from src.server import _format_user_friendly_error

    err = RuntimeError("Slot 6 (เมทริกซ์ / matrix_operations_transpose) failed SymPy verification: SymPy='32.0' does not match choice #2 ('30')")
    msg = _format_user_friendly_error(err, provider="ollama", model="qwen3.5:4b")
    assert "เมทริกซ์ / matrix_operations_transpose" in msg
    assert "SymPy='32.0' does not match choice #2 ('30')" in msg


def test_deterministic_mcq_choices_and_stem_integrity_all_16_chapters():
    import random
    from src.sympy_seeds import (
        _SEEDERS,
        build_deterministic_mcq_choices,
        generate_verified_seed,
        verify_seed_numbers_in_stem,
    )
    from src.verifier import verify_generated_question

    for ch_id in _SEEDERS:
        seed = generate_verified_seed(ch_id, seed=123)
        assert seed is not None
        choices, ans_idx = build_deterministic_mcq_choices(seed.exact_answer, ch_id, rng=random.Random(42))
        assert len(choices) == 5
        assert ans_idx in {"1", "2", "3", "4", "5"}

        ver = verify_generated_question(
            stem_latex=f"โจทย์ทดสอบบท {ch_id}: {seed.params_description_th}",
            question_format=QuestionFormat.MCQ_5,
            choices_latex=choices,
            expected_answer=ans_idx,
            sympy_code=seed.sympy_code,
            chapter_id=ch_id,
        )
        assert ver.passed, f"{ch_id} failed deterministic SymPy gate: {ver.error}"

        ok, err = verify_seed_numbers_in_stem(seed, seed.params_description_th)
        assert ok, f"{ch_id} stem integrity false negative: {err}"

    # Verify Stem Integrity Gate rejects hallucinated/altered numbers
    calc_seed = generate_verified_seed("sets", seed=123)
    assert calc_seed is not None
    ok_bad, err_bad = verify_seed_numbers_in_stem(calc_seed, "กลุ่มตัวอย่างทั้งหมด 999 คน ชอบ A 1 คน")
    assert not ok_bad and err_bad is not None


def test_verifier_rounding_latex_and_sympy_edge_cases():
    import math
    from src.verifier import _numbers_match, _parse_numeric, run_sympy_code

    # 1. Exact A-Level 2-decimal rounding equivalence (23.4375 -> 23.44 passes, 23.45 fails)
    assert _numbers_match(23.4375, 23.44, "23.44") is True
    assert _numbers_match(23.4375, 23.45, "23.45") is False
    assert _numbers_match(0.25, 25.0, "25%") is True

    # 2. Advanced LaTeX expressions (\dfrac, nth root, log base, e^x, degrees)
    assert math.isclose(_parse_numeric(r"$\dfrac{3}{2}$"), 1.5, rel_tol=1e-6)
    assert math.isclose(_parse_numeric(r"$\sqrt[3]{27}$"), 3.0, rel_tol=1e-6)
    assert math.isclose(_parse_numeric(r"$\log_{2}{8}$"), 3.0, rel_tol=1e-6)
    assert math.isclose(_parse_numeric(r"$e^2$"), math.e**2, rel_tol=1e-6)
    assert math.isclose(_parse_numeric(r"$45^\circ$"), 45.0, rel_tol=1e-6)

    # 3. SymPy worker convenience aliases (sp.comb, sp.perm, single-key dict, statistics)
    ok_comb, val_comb = run_sympy_code("ANSWER = sp.comb(5, 2) + sp.perm(4, 2)")
    assert ok_comb and math.isclose(float(val_comb), 22.0, rel_tol=1e-6)

    ok_dict, val_dict = run_sympy_code("x = sp.Symbol('x')\nANSWER = sp.solve([sp.Eq(2*x - 6, 0)], [x])")
    assert ok_dict and math.isclose(float(val_dict), 3.0, rel_tol=1e-6)


def test_llm_router_circuit_breaker_and_provider_guards():
    from src.llm_router import _is_daily_quota_exhausted, _parse_retry_after_seconds

    assert recommend_workers("ollama") == 1
    assert recommend_batch_size("ollama") == 1
    assert recommend_workers("gemini") == 2
    assert recommend_batch_size("gemini") == 5
    assert recommend_workers("openrouter") == 4

    assert _parse_retry_after_seconds("Please retry in 22.5s.") == 23.5
    assert _parse_retry_after_seconds("{'retryDelay': '18s'}") == 19.0
    assert _parse_retry_after_seconds("Please retry in 2h40m58.6s.") is None
    assert _is_daily_quota_exhausted("GenerateRequestsPerDayPerProjectPerModel-FreeTier") is True
    assert _is_daily_quota_exhausted("Generate content requests per day limit exceeded on Free Tier. Please retry in 2h40m58s.") is True
    assert _is_daily_quota_exhausted("GenerateRequestsPerMinutePerProjectPerModel-FreeTier") is False

    clear_cooldowns()
    assert is_model_in_cooldown("gemini:test-model") is False
    mark_model_cooldown("gemini:test-model", 30.0)
    assert is_model_in_cooldown("gemini:test-model") is True
    clear_cooldowns()

    o4_payload = build_openai_payload("o4-mini", "hello", temperature=0.4)
    assert "temperature" not in o4_payload
    gpt_payload = build_openai_payload("gpt-4.1-mini", "hello", temperature=0.4)
    assert gpt_payload["temperature"] == 0.4

    from src.server import _format_user_friendly_error

    msg_rpd = _format_user_friendly_error(
        RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay"), "gemini", "gemini-3.8-flash"
    )
    assert "โควตาฟรีรายวัน" in msg_rpd
    assert "RESOURCE_EXHAUSTED" not in msg_rpd

    msg_rpm = _format_user_friendly_error(
        RuntimeError("429 RESOURCE_EXHAUSTED"), "gemini", "gemini-3.8-flash"
    )
    assert "ถี่เกินโควตาฟรี" in msg_rpm
    assert "RESOURCE_EXHAUSTED" not in msg_rpm

    from src.llm_router import (
        GEMINI_FREE_QUOTA_DEFAULTS,
        _QUOTA_LOCK,
        _QUOTA_STATE,
        _save_quota_state_unlocked,
        get_quota_status,
        record_quota_exhausted,
    )

    assert GEMINI_FREE_QUOTA_DEFAULTS["gemini-3.5-flash-lite"]["rpd"] == 500
    ollama_q = get_quota_status("ollama", "qwen3.5:4b")
    assert ollama_q["unlimited"] is True
    assert ollama_q["pct"] == 100

    record_quota_exhausted(
        "gemini",
        "gemini-test-quota-check",
        "429 limit: 20 Generate content requests per day limit exceeded. Please retry in 2h30m0s.",
        is_daily=True,
    )
    try:
        exhausted_q = get_quota_status("gemini", "gemini-test-quota-check")
        assert exhausted_q["status"] == "exhausted_rpd"
        assert exhausted_q["rpd_remaining"] == 0
        assert exhausted_q["pct"] == 0
        assert exhausted_q["reset_in_seconds"] > 8000
    finally:
        with _QUOTA_LOCK:
            _QUOTA_STATE.pop("gemini:gemini-test-quota-check", None)
            _save_quota_state_unlocked()
