"""v3.0 A-Level 70 Exam Generator (QUESTION / MINI-TEST / MOCK-TEST) with SymPy Gate & 1-5 Star Difficulty."""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from src.dedup import GLOBAL_STEM_REGISTRY
from src.diversity import format_few_shot_block, sample_diversity_directive
from src.llm_router import (
    JobCancelledError,
    check_cancelled,
    generate_structured_json,
    recommend_batch_size,
    recommend_workers,
    set_ai_activity,
)
from src.model import (
    ExamSlot,
    allocate_blueprint_slots,
    load_curriculum,
)
from src.question_pool import GLOBAL_POOL, VerifiedQuestionPool
from src.schema import DifficultyLevel, QuestionFormat, QuestionStyle, load_syllabus
from src.sympy_seeds import (
    build_deterministic_mcq_choices,
    generate_verified_seed,
    verify_seed_numbers_in_stem,
)
from src.verifier import (
    sanitize_choice_text,
    sanitize_latex_text,
    sort_mcq_choices_ascending,
    verify_generated_question,
)


GenerationMode = Literal["question", "mini_test", "mock_test"]
DifficultyLevelTier = Literal[1, 2, 3]

DIFFICULTY_LEVEL_PROFILES: dict[int, dict] = {
    1: {
        "label": "Level 1 • A-Level",
        "target_stars": "3–4 ดาว",
        "min_stars": 3,
        "thinking_budget": 128,
        "directive": (
            "ระดับความยาก Level 1 (มาตรฐาน A-Level สสวท. จริง — 3–4 ดาว): "
            "เชื่อมโยง 2 แนวคิด คิด 2 ขั้นตอน งดโจทย์ 1 ดาวที่แทนสูตรบรรทัดเดียว"
        ),
    },
    2: {
        "label": "Level 2 • Hard (+50%)",
        "target_stars": "4 ดาว",
        "min_stars": 4,
        "thinking_budget": 256,
        "directive": (
            "ระดับความยาก Level 2 (ยากขึ้นกว่า A-Level ปกติ +50% — เป้าหมาย 4 ดาว): "
            "ต้องเพิ่มความซับซ้อนของการคำนวณและการประยุกต์อย่างชัดเจน ใช้การคิดต่อเนื่องอย่างน้อย 3 ขั้นตอน "
            "(เช่น แก้เงื่อนไขแรกหาพารามิเตอร์ → แทนในฟังก์ชัน/สมการขั้นที่สอง → เชื่อมโยงเงื่อนไขประยุกต์หรือเช็กโดเมนซ้อน) "
            "ใช้เวลาคิดและคำนวณนานขึ้นกว่าข้อสอบ A-Level ปกติ แต่ยังต้องอยู่ในหลักสูตร สสวท. 2560 และคิดเลขลงตัว 100%"
        ),
    },
    3: {
        "label": "Level 3 • Extreme (+125%)",
        "target_stars": "4–5 ดาว",
        "min_stars": 4,
        "thinking_budget": 384,
        "directive": (
            "ระดับความยาก Level 3 (ยากขึ้นจาก Level 2 อีก +50% / ระดับ PAT1-Elite — เป้าหมาย 4–5 ดาว): "
            "บูรณาการข้ามบทหรือซ้อนเงื่อนไขลึกอย่างน้อย 3–4 ขั้นตอนต่อเนื่อง ต้องจัดรูปพีชคณิต/แคลคูลัส/เรขาคณิตหลายชั้น "
            "แฝงกับดักโดเมน-ขอบเขต-เครื่องหมายที่ต้องวิเคราะห์อย่างละเอียด ใช้เวลาคำนวณนานที่สุดใน 3 ระดับ "
            "(แต่ยังคงอยู่ในหลักสูตร สสวท. 2560, คำตอบเดียวแน่นอน, และผ่าน SymPy Verification 100%)"
        ),
    },
}


def get_difficulty_profile(level: int) -> dict:
    return DIFFICULTY_LEVEL_PROFILES.get(int(level), DIFFICULTY_LEVEL_PROFILES[1])
_STYLE_GUIDE_PATH = Path(__file__).resolve().parent.parent / "config" / "exam_style_guide.json"
_STYLE_GUIDE_CACHE: dict | None = None


def load_exam_style_guide() -> dict:
    """Load the compact IPST A-Level 61 exam style guide JSON (cached in-process)."""
    global _STYLE_GUIDE_CACHE
    if _STYLE_GUIDE_CACHE is None:
        if _STYLE_GUIDE_PATH.exists():
            _STYLE_GUIDE_CACHE = json.loads(_STYLE_GUIDE_PATH.read_text(encoding="utf-8"))
        else:
            _STYLE_GUIDE_CACHE = {}
    return _STYLE_GUIDE_CACHE


def _style_guide_prompt_block(question_format: QuestionFormat) -> str:
    guide = load_exam_style_guide()
    if not guide:
        return ""
    rules = list(guide.get("common_rules_th", []))
    if question_format == QuestionFormat.MCQ_5:
        rules.extend(guide.get("mcq_5_rules_th", []))
    else:
        rules.extend(guide.get("numeric_rules_th", []))
    if not rules:
        return ""
    return "\nมาตรฐาน Format ข้อสอบ สสวท. (Exam Style Guide):\n" + "\n".join(
        f"- {r}" for r in rules
    )


def _pick_seed_for_slot(
    chapter_id: str,
    question_format: QuestionFormat,
    difficulty_level: int = 1,
    exclude_archetypes: set[str] | None = None,
):
    """Pick a SymPy-First seed respecting difficulty_level, non-repeating archetypes, and Part 2 bounds."""
    import sympy as sp

    for _ in range(10):
        s = generate_verified_seed(
            chapter_id,
            difficulty_level=difficulty_level,
            exclude_archetypes=exclude_archetypes,
        )
        if s is None or question_format == QuestionFormat.MCQ_5:
            if s is not None and s.archetype_id and exclude_archetypes is not None:
                exclude_archetypes.add(s.archetype_id)
            return s
        try:
            val = float(sp.N(sp.sympify(s.exact_answer)))
            if 0.0 <= val <= 9999.99 and abs(val * 100.0 - round(val * 100.0)) <= 1e-4:
                if s.archetype_id and exclude_archetypes is not None:
                    exclude_archetypes.add(s.archetype_id)
                return s
        except Exception:
            continue
    return None


class LogicAuditItem(BaseModel):
    slot_number: int
    logically_valid: bool = Field(
        ...,
        description="True ONLY if all mathematical conditions are non-contradictory, solvable, realistic, and choices match what the stem asks.",
    )
    issue: str = Field(default="", description="Short reason if logically_valid is False")


class LogicAuditBatch(BaseModel):
    audits: list[LogicAuditItem] = Field(default_factory=list)


def _sanitize_draft(draft: "GeneratedQuestionDraft") -> "GeneratedQuestionDraft":
    return draft.model_copy(
        update={
            "stem_latex": sanitize_latex_text(draft.stem_latex),
            "choices_latex": [sanitize_choice_text(c) for c in draft.choices_latex],
            "solution_steps_th": sanitize_latex_text(draft.solution_steps_th),
            "key_trap": sanitize_latex_text(draft.key_trap),
        }
    )


def audit_math_logic_batch(
    candidates: list[tuple[ExamSlot, "GeneratedQuestionDraft"]],
    provider: str,
    model_name: str,
    cancel_event: threading.Event | None = None,
) -> dict[int, tuple[bool, str]]:
    """Independent LLM mathematical logic & consistency auditor (checks contradictions, solvability, realism)."""
    check_cancelled(cancel_event)
    if not candidates:
        return {}
    blocks = []
    for slot, d in candidates:
        blocks.append(
            f"[ข้อที่ {slot.slot_number}] รูปแบบ={slot.question_format.value}\n"
            f"โจทย์: {d.stem_latex}\n"
            f"ตัวเลือก: {d.choices_latex}\n"
            f"คำตอบที่ระบุ: {d.answer}\n"
            f"วิธีทำ: {d.solution_steps_th}"
        )
    audit_prompt = (
        "คุณคือกรรมการตรวจความถูกต้องทางคณิตศาสตร์ระดับ สสวท. (เข้มงวดสูงสุด)\n"
        "ตรวจสอบโจทย์ทุกข้อด้านล่างว่ามีข้อบกพร่องเหล่านี้หรือไม่:\n"
        "1. เงื่อนไขขัดแย้งกันเองหรือเป็นไปไม่ได้ทางคณิตศาสตร์ (เช่น กำหนด x!=0 แต่สมการบังคับ x=0, กำหนดค่าเฉลี่ยและมัธยฐานที่ขัดแย้งกัน, กำหนดลิมิตไม่เท่ากับค่าฟังก์ชันแต่ให้หาความต่อเนื่อง, จุดบนไฮเพอร์โบลาตัดเส้นกำกับ)\n"
        "2. หาคำตอบไม่ได้ หรือมีหลายคำตอบที่ขัดกับโจทย์ หรือเงื่อนไขไม่ครบ (เช่น ขาดชื่อรูป/สามเหลี่ยม)\n"
        "3. ตัวเลือกไม่สอดคล้องกับสิ่งที่โจทย์ถาม (เช่น โจทย์ถามประพจน์ใดสมมูล แต่ตัวเลือกเป็นตัวเลข 1,2,3,4,5)\n"
        "4. มีภาษาวิบัติหรือโค้ด LaTeX แตกอ่านไม่รู้เรื่อง\n"
        "ให้คืนผลตรวจครบทุกข้อในฟิลด์ `audits` โดยตั้ง `logically_valid=false` ทันทีถ้าพบข้อบกพร่องแม้แต่ข้อเดียว\n\n"
        + "\n---\n".join(blocks)
    )
    try:
        set_ai_activity(
            "working",
            f"กำลังตรวจตรรกะภาษาไทย ({len(candidates)} ข้อ)",
            f"ให้ {model_name} ตรวจเช็กเงื่อนไขโจทย์และตัวเลือก",
        )
        res = generate_structured_json(
            provider=provider,
            model=model_name,
            prompt=audit_prompt,
            schema_cls=LogicAuditBatch,
            temperature=0.0,
            timeout_seconds=25.0,
            thinking_budget=64,
            cancel_event=cancel_event,
        )
        return {a.slot_number: (a.logically_valid, a.issue) for a in res.audits}
    except JobCancelledError:
        raise
    except Exception as exc:
        print(f"[audit] warning: logic auditor skipped ({exc})", flush=True)
        return {slot.slot_number: (True, "") for slot, _ in candidates}


class GeneratedQuestionDraft(BaseModel):
    math_scratchpad: str = Field(
        default="",
        description="Private brief math setup/verification scratchpad BEFORE writing stem_latex (never shown to students).",
    )
    stem_latex: str = Field(..., min_length=15)
    choices_latex: list[str] = Field(default_factory=list)
    answer: str = Field(..., min_length=1)
    difficulty: DifficultyLevel
    difficulty_stars: int = Field(..., ge=1, le=5, description="Difficulty rating 1 (easiest) to 5 (hardest)")
    question_style: QuestionStyle
    key_trap: str
    solution_steps_th: str
    sympy_code: str = Field(
        ...,
        description="Self-contained Python code using sympy (imported as sp) that assigns the exact numeric/scalar answer to variable ANSWER.",
    )


CHAPTER_LOGIC_INVARIANTS: dict[str, str] = {
    "sets": "จำนวนสมาชิกในทุกบริเวณของแผนภาพเวนน์ต้องเป็นจำนวนเต็ม >= 0 และ n(A∩B) <= min(n(A), n(B)), n(A∪B) <= n(U)",
    "logic": "ห้ามใช้ sp.satisfiability หรือ equivalent ใน SymPy — ให้ใช้ itertools.product([True, False], repeat=n) ตรวจตารางค่าความจริง และตั้งคำถามหาจำนวนข้อที่ถูก/จำนวนกรณี; ต้องใช้สัญลักษณ์ตรรกศาสตร์ LaTeX ที่ถูกต้องครบทุกตัว ($\\land$, $\\lor$, $\\rightarrow$, $\\leftrightarrow$, $\\sim$ หรือ $\\neg$, $\\equiv$, $\\forall x$, $\\exists x$) ห้ามใช้สัญลักษณ์ ASCII แบบ ->, <->, ^, v, ~",
    "real_numbers": "อสมการต้องถามหาค่าตัวเลขเดี่ยว (เช่น ผลบวกจำนวนเต็มที่เป็นคำตอบ หรือ a+b) และต้องหักค่าที่ทำให้ตัวส่วนเป็น 0 ออกเสมอ",
    "functions": "ใช้ Reverse-Engineering Construction: ถ้าหาเรนจ์ของพาราโบลา f(x)=ax^2+bx+c บนช่วงปิด [x1,x2] ต้องคำนวณจุดยอด x_v=-b/(2a) และเช็ก {f(x1),f(x2),f(x_v)} เมื่อ x1<=x_v<=x2; ฟังก์ชันประกอบ f(g(x)) ต้องให้เรนจ์ของ g อยู่ในโดเมนของ f (ไม่หาร 0, รากคู่ >= 0, หลัง log > 0); ฟังก์ชันแยกช่วงต้องให้ lim ซ้าย == f(c) == lim ขวา",
    "expo_log": "ลอการิทึมต้อง assert หลัง log > 0 และฐาน > 0, != 1 ทุกคำตอบที่ได้",
    "trigonometry": "ต้อง assert |sin| <= 1, |cos| <= 1, อสมการสามเหลี่ยม a+b>c และเรนจ์หลักของ arcsin/arccos/arctan",
    "complex_numbers": "ใน SymPy ใช้ sp.I เท่านั้น (ห้ามใช้ i ตัวเล็ก) และทฤษฎีรากสังยุคใช้ได้เฉพาะพหุนามสัมประสิทธิ์จำนวนจริง",
    "matrices": "ถ้าใช้ A^{-1} ต้อง assert det(A) != 0 และห้ามกำหนดเงื่อนไข x != 0 ที่ขัดกับสมการ det = 0",
    "sequences_series": "ใช้ Reverse-Engineering Construction: กำหนดค่าจำนวนเต็มบวกของ a1,d,b1,r และ n เป้าหมายก่อนแล้วค่อยคำนวณ K=a_n+b_n; อนุกรมเรขาคณิตอนันต์ต้อง assert |r| < 1 และระบุพจน์แรกตามดัชนีเริ่ม (n=1 vs n=0) ชัดเจน; อนุกรมเทเลสโคปิก 1/((n+a)(n+b)) ต้องคูณตัวประกอบ 1/(b-a) เสมอ; คณิตศาสตร์การเงินระบุงวดทบต้นและต้นงวด/ปลายงวดชัดเจน",
    "analytic_geometry": "ใช้ Reverse-Engineering Construction ด้วยสามจำนวนพีทาโกรัส (3,4,5)/(5,12,13)/(8,15,17): วงกลม r^2>0; วงรี a^2=b^2+c^2 (a>b>0, a>c>0); ไฮเพอร์โบลา c^2=a^2+b^2 (c>a>0, c>b>0) และห้ามวางจุด P บนทั้งไฮเพอร์โบลาและเส้นกำกับ (ไม่มีวันตัดกัน!); เส้นขนาน Ax+By+C=0 ให้เลือก (A,B) ที่ sqrt(A^2+B^2) เป็นจำนวนเต็มแล้วคำนวณ C2=C1±d*sqrt(A^2+B^2)",
    "vectors_3d": "ใช้ Reverse-Engineering Construction: กำหนดองค์ประกอบจำนวนเต็มของ u,v,w ก่อนแล้วคำนวณพื้นที่ |u×v| หรือปริมาตร |u·(v×w)| ด้วย SymPy ก่อนเขียนโจทย์; ห้ามสุ่มเวกเตอร์ขนานกัน (|u×v|>0) หรืออยู่ระนาบเดียวกัน (|u·(v×w)|>0); ห้ามแก้ตัวเลขเวกเตอร์ใน solution_steps_th; ถ้ามีมุมให้ใช้มุมมาตรฐาน 30,45,60,90,120,135,150 องศาและ assert -1 <= (u·v)/(|u||v|) <= 1",
    "statistics": "ข้อมูลที่เรียงลำดับต้องคงลำดับจริงหลังแก้หาตัวแปร x; ค่าเฉลี่ย มัธยฐาน ฐานนิยม ห้ามขัดแย้งกัน; ระบุชัดว่าเป็นประชากร (หาร N) หรือตัวอย่าง (หาร n-1)",
    "prob_distributions": "ต้อง assert 0 <= p <= 1 และผลรวมความน่าจะเป็น = 1; ถ้าใช้โค้งปกติ (Z-score) ต้องให้ค่าตาราง P(Z<=z) ที่จำเป็นครบในโจทย์",
    "counting": "คำตอบต้องเป็นจำนวนเต็ม >= 0; ใน SymPy ใช้ sp.binomial(n, r) และ sp.factorial(n) (ห้ามใช้ sp.comb)",
    "probability": "ต้อง assert 0 <= P <= 1 และเงื่อนไขการหยิบของต้องเป็นไปได้จริงตามจำนวนของที่มี",
    "calculus": "ความต่อเนื่องที่ x=c ต้อง assert limit ซ้าย == limit ขวา == f(c); พื้นที่ใต้กราฟต้องแบ่งช่วงเมื่อฟังก์ชันเปลี่ยนเครื่องหมาย",
}


def _chapter_guard_hint(chapter_id: str) -> str:
    return CHAPTER_LOGIC_INVARIANTS.get(chapter_id, "ตรวจสอบเงื่อนไขโดเมนทุกข้อให้สอดคล้องกันและมีคำตอบเดียว")


def _lookup_curriculum_context(chapter_id: str, sub_skill_id: str) -> dict:
    curr = load_curriculum()
    for ch in curr.get("chapters", []):
        if ch["chapter_id"] == chapter_id:
            for sd in ch.get("sub_skill_details", []):
                if sd["sub_skill_id"] == sub_skill_id:
                    return {
                        "chapter_th": ch["name_th"],
                        "topics_th": sd.get("topics_th", []),
                        "ipst_applied_contexts": sd.get("ipst_applied_contexts", []),
                        "classic_traps": sd.get("classic_traps", []),
                    }
    return {"chapter_th": chapter_id, "topics_th": [], "ipst_applied_contexts": [], "classic_traps": []}


def build_slots_for_mode(
    mode: GenerationMode,
    subskill_probs: dict[str, float],
    cross_topic_pairs: dict[str, list[tuple[str, int]]] | None = None,
    seed: int | None = None,
) -> list[ExamSlot]:
    """Build target exam slots for QUESTION (1), MINI-TEST (15 diverse), or MOCK-TEST (30 predicted)."""
    if mode == "mock_test":
        return allocate_blueprint_slots(subskill_probs, cross_topic_pairs=cross_topic_pairs)

    rng = random.Random(seed)
    syllabus = load_syllabus()
    cross_topic_pairs = cross_topic_pairs or {}

    chapter_pool: list[tuple[str, str, list[str]]] = [
        (strand.id, ch.id, list(ch.sub_skills))
        for strand in syllabus.strands
        for ch in strand.chapters
    ]

    if mode == "question":
        strand_id, ch_id, skills = rng.choice(chapter_pool)
        sk = rng.choice(skills)
        fmt = rng.choices([QuestionFormat.MCQ_5, QuestionFormat.NUMERIC], weights=[5, 1])[0]
        sec_list = cross_topic_pairs.get(ch_id, [])
        return [
            ExamSlot(
                slot_number=1,
                strand_id=strand_id,
                chapter_id=ch_id,
                sub_skill_id=sk,
                secondary_chapter=sec_list[0][0] if sec_list else None,
                question_format=fmt,
                predicted_prob=round(subskill_probs.get(sk, 0.02), 5),
                rationale=f"Random single question from chapter '{ch_id}'",
            )
        ]

    # MINI-TEST: 15 questions spanning 15 distinct chapters (maximal diversity across all 16 chapters)
    # 12 MCQ (slots 1-12) + 3 Numeric (slots 13-15)
    rng.shuffle(chapter_pool)
    chosen_chapters = chapter_pool[:15]
    slots: list[ExamSlot] = []
    for idx, (strand_id, ch_id, skills) in enumerate(chosen_chapters, start=1):
        sk = rng.choice(skills)
        fmt = QuestionFormat.MCQ_5 if idx <= 12 else QuestionFormat.NUMERIC
        sec_list = cross_topic_pairs.get(ch_id, [])
        slots.append(
            ExamSlot(
                slot_number=idx,
                strand_id=strand_id,
                chapter_id=ch_id,
                sub_skill_id=sk,
                secondary_chapter=sec_list[0][0] if sec_list else None,
                question_format=fmt,
                predicted_prob=round(subskill_probs.get(sk, 0.02), 5),
                rationale=f"Diverse mini-test slot from chapter '{ch_id}'",
            )
        )
    return slots


def generate_verified_slot(
    slot: ExamSlot,
    traps: list[str],
    historical_stems: list[str],
    provider: str = "gemini",
    model_name: str = "gemini-3.5-flash-lite",
    max_retries: int = 5,
    _allow_fallback: bool = True,
    cancel_event: threading.Event | None = None,
    used_archetypes: set[str] | None = None,
    difficulty_level: int = 1,
) -> dict:
    """Generate a single original IPST-style A-Level 70 question and verify it with SymPy."""
    check_cancelled(cancel_event)
    diff_prof = get_difficulty_profile(difficulty_level)
    effective_retries = min(max_retries, 2) if provider == "ollama" else max_retries
    ctx = _lookup_curriculum_context(slot.chapter_id, slot.sub_skill_id)
    seed_obj = _pick_seed_for_slot(
        slot.chapter_id,
        slot.question_format,
        difficulty_level=difficulty_level,
        exclude_archetypes=used_archetypes,
    )
    _, diversity_hint = sample_diversity_directive(slot.chapter_id, used_archetypes=used_archetypes)
    few_shot_block = format_few_shot_block(slot.chapter_id, k=1)
    locked_choices: list[str] | None = None
    locked_answer: str | None = None
    if seed_obj and provider == "ollama":
        if slot.question_format == QuestionFormat.MCQ_5:
            locked_choices, locked_answer = build_deterministic_mcq_choices(
                seed_obj.exact_answer, slot.chapter_id
            )
            choice_hint = (
                f"  • ตัวเลือก 5 ข้อที่ล็อกไว้แล้ว: {locked_choices} | หมายเลขคำตอบที่ถูก = '{locked_answer}' (ค่า = `{seed_obj.exact_answer}`)\n"
            )
        else:
            locked_choices = []
            locked_answer = seed_obj.exact_answer
            choice_hint = f"  • คำตอบอัตนัยที่ล็อกไว้แล้ว = `{locked_answer}`\n"
        seed_block = (
            f"\n- **[บังคับ 100% สำหรับ Local Model] นำเงื่อนไขและตัวเลขทุกตัวจาก SymPy-First Seed ด้านล่างไปแต่งเป็นโจทย์ภาษาไทย ห้ามเปลี่ยนตัวเลขแม้แต่ตัวเดียว:**\n"
            f"  {seed_obj.params_description_th}\n"
            f"{choice_hint}"
            f"  • คัดลอก `sympy_code` นี้ไปใส่โดยตรง:\n```python\n{seed_obj.sympy_code}\n```\n"
        )
    elif seed_obj:
        seed_block = (
            f"\n- **ตัวเลขแกนกลางที่ผ่าน SymPy แล้ว (แนะนำให้นำไปแต่งเป็นโจทย์ประยุกต์ภาษาไทยเพื่อให้คำตอบลงตัว 100%):**\n"
            f"  {seed_obj.params_description_th}\n"
            f"  (สามารถใช้ sympy_code นี้ได้โดยตรง: `{seed_obj.sympy_code}`)\n"
        )
    else:
        seed_block = ""
    fmt_instruction = (
        "ปรนัย 5 ตัวเลือก (choices_latex ต้องมี 5 ตัวเลือกที่เป็นตัวเลข/ค่าไม่ซ้ำกัน และ answer เป็น '1'-'5')"
        if slot.question_format == QuestionFormat.MCQ_5
        else (
            "อัตนัยระบายคำตอบตัวเลข Part 2 (choices_latex=[]): กฎเหล็กกระดาษคำตอบ สสวท. — "
            "คำตอบสุดท้ายต้องเป็นจำนวนไม่ติดลบ `0 <= answer <= 9999.99` และทศนิยมไม่เกิน 2 ตำแหน่งเท่านั้น "
            "(ห้ามติดลบ ห้ามมีเครื่องหมาย `-` และห้ามตอบเศษส่วน `/` เช่น `3/16` — ถ้าคิดได้ติดลบหรือเศษส่วน "
            "ให้ปรับคำถามเป็นค่าสัมบูรณ์ `|ans|` หรือคูณสเกลเช่น `16P` เพื่อให้ฝนคำตอบบวกที่ลงตัวได้จริง!)"
        )
    )

    prompt = f"""คุณคือผู้ออกข้อสอบ A-Level คณิตศาสตร์ประยุกต์ 1 รหัสวิชา 61 (สสวท. หลักสูตร 2560) สำหรับปี 2570
จงสร้างโจทย์ใหม่ 100% ข้อที่ {slot.slot_number} ตามโหมดความยาก: {diff_prof['directive']}
สเปกรายละเอียด:
- บทหลัก: {ctx['chapter_th']} ({slot.chapter_id})
- ทักษะย่อยเป้าหมาย: {slot.sub_skill_id}
- หัวข้อในหลักสูตร สสวท.: {', '.join(ctx['topics_th'])}
- บทบูรณาการเสริม (ถ้าเหมาะสม): {slot.secondary_chapter or 'ไม่มี'}
- รูปแบบข้อสอบ: {fmt_instruction}
- บริบทประยุกต์สไตล์ สสวท.: {', '.join(ctx['ipst_applied_contexts'])}
- ความหลากหลายรอบนี้ (ห้ามซ้ำแม่แบบเดิม): {diversity_hint}
- จุดหลอกที่ควรแฝงในตัวเลือกหลอก: {', '.join(traps[:3] or ctx['classic_traps'])}
- กฎลอจิกเฉพาะบทนี้ (ห้ามละเมิด): {_chapter_guard_hint(slot.chapter_id)}{seed_block}{few_shot_block}

กฎเหล็ก:
1. ห้ามใช้เนื้อหานอกหลักสูตร 2560 (ห้ามใช้โลปิตาล, กำหนดการเชิงเส้น, ทฤษฎีจำนวนแบบเก่า)
2. **ห้ามสร้างเงื่อนไขขัดแย้งกันเองเด็ดขาด!** ตรวจสอบก่อนเสมอว่าทุกสมการ เงื่อนไขโดเมน (เช่น $x \neq 0$, ข้อมูลเรียงจากน้อยไปมาก, ลิมิตซ้าย=ขวา=$f(c)$, สมบัติภาคตัดกรวยจริง) เป็นจริงพร้อมกันและมีคำตอบเดียวที่หาได้เสมอ
3. คำตอบต้องเป็น **ค่าตัวเลขเดี่ยวที่แน่นอน** เท่านั้น (เช่น `12`, `-3.5`, `$\\frac{5}{2}$`, `$\\sqrt{3}$`) — **ห้ามใช้ช่วงคำตอบ `(a, b)`, `[a, b]`, `\\infty` หรือเซต `{...}` เป็นตัวเลือก!** สำหรับโจทย์อสมการให้ถามหา *"ผลบวกของจำนวนเต็มที่เป็นคำตอบ"*, *"จำนวนเต็มน้อยที่สุดที่สอดคล้อง"*, หรือ *"ค่าของ $a+b$ เมื่อเซตคำตอบคือ $(a, b)$"* และสำหรับบทตรรกศาสตร์/เซต ให้ถามเป็น **จำนวนข้อที่ถูกต้อง / จำนวนสมาชิก** (ห้ามใส่ข้อความภาษาไทยใน `choices_latex`)
4. ใช้ LaTeX มาตรฐาน KaTeX ที่ถูกต้อง 100% ครบทุกสัญลักษณ์พิเศษทางคณิตศาสตร์ (ใส่เครื่องหมาย `$` เปิด-ปิดให้ครบคู่ และ escape backslash ใน JSON ให้ถูก เช่น `\\neq`, `\\leq`, `\\geq`, `\\times`, `\\cdot`, `\\frac`, `\\sqrt`, `\\land`, `\\lor`, `\\rightarrow`, `\\leftrightarrow`, `\\sim`, `\\neg`, `\\equiv`, `\\forall`, `\\exists`, `\\in`, `\\notin`, `\\subset`, `\\cup`, `\\cap`, `\\emptyset`, `\\vec{{u}}`, `\\sum`, `\\int`, `\\lim`, `\\begin{{bmatrix}}...\\end{{bmatrix}}`)
5. เลือกใช้ตัวเลขในโจทย์ที่ **คิดเลขลงตัวและไม่ซับซ้อนเกินไป** เพื่อให้คำตอบของ `sympy_code`, `choices_latex`, และ `solution_steps_th` ตรงกันเป๊ะ 100% — **ห้ามแอบแก้ไขตัวเลขโจทย์หรือเขียนว่า "ปรับโจทย์/ปรับเวกเตอร์/ขออภัย" ใน `solution_steps_th` เด็ดขาด!**
6. ประเมินระดับความยากเป็น `difficulty_stars` ตั้งแต่ 1 ถึง 5 ดาว (1=พื้นฐานมาก, 2=ค่อนข้างง่าย, 3=มาตรฐาน A-Level, 4=ยาก/บูรณาการ, 5=ยากมาก/คัดแยกคะแนน)
7. ในฟิลด์ `sympy_code` ต้องเขียนโค้ด Python ที่ใช้ `sp` (sympy) คำนวณหาคำตอบจริงจากเงื่อนไขโจทย์ + ใส่ `assert` ครบ 4 ด้าน: (ก) กรองโดเมนแล้ว `assert len(valid_sols) == 1` (ข) แก้ทุกเงื่อนไขเป็นระบบสมการเดียวกันแล้ว `assert` ว่าไม่ขัดแย้งกัน (ค) `assert` นิยามคณิตศาสตร์ (เช่น ลิมิตซ้าย=ขวา=f(c), det!=0) (ง) `assert len(ANSWER.free_symbols) == 0` ห้ามติดตัวแปรอิสระ (ห้าม hardcode `ANSWER = ...` และห้ามใส่หมายเลขตัวเลือก 1–5 ลงใน `ANSWER`)
{_style_guide_prompt_block(slot.question_format)}
"""

    last_error = ""
    last_computed: str | None = None
    last_draft_summary = ""
    for attempt in range(effective_retries + 1):
        check_cancelled(cancel_event)
        set_ai_activity(
            "working",
            f"แต่งโจทย์ข้อ {slot.slot_number} • {ctx['chapter_th']}",
            f"โมเดล {model_name} (รอบที่ {attempt + 1}/{effective_retries + 1})"
            + (" • ใช้ SymPy-First Seed" if seed_obj else ""),
        )
        print(
            f"[gen] slot={slot.slot_number} ({slot.sub_skill_id}) provider={provider}:{model_name} attempt={attempt + 1}/{effective_retries + 1}...",
            flush=True,
        )
        if last_error:
            if last_computed and attempt <= 2:
                locked_hint = ""
                if slot.question_format == QuestionFormat.MCQ_5:
                    det_ch, det_ans = build_deterministic_mcq_choices(last_computed, slot.chapter_id)
                    locked_hint = f"\nแนะนำให้ใช้ตัวเลือกที่ตรงกับค่าจริงนี้เลย: choices_latex={det_ch}, answer='{det_ans}'"
                retry_hint = (
                    f"\n\nรอบก่อนไม่ผ่านการตรวจ SymPy เพราะ: {last_error}\n"
                    f"โจทย์และโค้ดเดิมของคุณคำนวณด้วย SymPy ได้ค่าจริงคือ `{last_computed}`:\n"
                    f"{last_draft_summary}\n"
                    f"จงแก้ `choices_latex`, `answer`, และ `solution_steps_th` ให้ตรงกับค่าจริง `{last_computed}` "
                    f"(หรือปรับตัวเลขในโจทย์ให้คิดลงตัวง่ายๆ แล้วให้ `ANSWER` ใน `sympy_code` ตรงกับตัวเลือกที่ถูกเป๊ะ 100%){locked_hint}"
                )
            else:
                retry_hint = (
                    f"\n\nรอบก่อนไม่ผ่านการตรวจ SymPy เพราะ: {last_error}\n"
                    f"จงสร้างโจทย์ใหม่ที่ใช้ตัวเลขจำนวนเต็มขนาดเล็ก คิดเลขลงตัวง่ายๆ "
                    f"และตรวจให้แน่ใจว่า `ANSWER` จาก `sympy_code` ตรงกับคำตอบและตัวเลือกที่ถูกต้อง 100%"
                )
        else:
            retry_hint = ""
        try:
            draft = _sanitize_draft(
                generate_structured_json(
                    provider=provider,
                    model=model_name,
                    prompt=prompt + retry_hint,
                    schema_cls=GeneratedQuestionDraft,
                    temperature=0.45 if attempt == 0 else 0.15,
                    thinking_budget=diff_prof["thinking_budget"] if attempt == 0 else 512,
                    cancel_event=cancel_event,
                )
            )
        except JobCancelledError:
            raise
        except ValidationError as val_exc:
            check_cancelled(cancel_event)
            last_error = f"Invalid/truncated JSON output: {str(val_exc)[:180]}"
            set_ai_activity(
                "working",
                f"ข้อ {slot.slot_number} JSON ไม่ครบ • กำลังให้แก้ใหม่",
                f"รอบที่ {attempt + 1}/{effective_retries + 1} • {ctx['chapter_th']}",
            )
            print(f"[gen] slot={slot.slot_number} RETRY (JSON): {last_error}", flush=True)
            time.sleep(0.1)
            continue
        check_cancelled(cancel_event)
        # True Neuro-Symbolic Lock for Local Ollama: Python/SymPy owns sympy_code, choices_latex,
        # and answer 100%, while Stem Integrity Gate guarantees the LLM kept every seed number intact.
        if provider == "ollama" and seed_obj is not None and locked_answer is not None and locked_choices is not None:
            stem_ok, stem_err = verify_seed_numbers_in_stem(seed_obj, draft.stem_latex)
            if not stem_ok:
                last_error = stem_err or "Stem altered seed parameters"
                print(f"[gen] slot={slot.slot_number} RETRY (StemGate): {last_error}", flush=True)
                time.sleep(0.1)
                continue
            draft = draft.model_copy(
                update={
                    "sympy_code": seed_obj.sympy_code,
                    "choices_latex": list(locked_choices),
                    "answer": locked_answer,
                }
            )
        set_ai_activity(
            "working",
            f"รัน SymPy ตรวจคำตอบข้อ {slot.slot_number}",
            f"เช็กสมการ + 5 Global Gates + LaTeX ({ctx['chapter_th']})",
        )
        ver = verify_generated_question(
            stem_latex=draft.stem_latex,
            question_format=slot.question_format,
            choices_latex=draft.choices_latex,
            expected_answer=draft.answer,
            sympy_code=draft.sympy_code,
            historical_stems=historical_stems,
            solution_steps_th=draft.solution_steps_th,
            chapter_id=slot.chapter_id,
            check_generated_history=True,
        )
        if ver.passed:
            # Skip redundant LLM Logic Audit on Local Ollama (SymPy + 5 Global Gates already enforce correctness)
            needs_llm_audit = (provider != "ollama") and ((seed_obj is None) or (attempt > 0))
            if needs_llm_audit:
                audit_map = audit_math_logic_batch(
                    [(slot, draft)],
                    provider=provider,
                    model_name=model_name,
                    cancel_event=cancel_event,
                )
                is_logical, logic_issue = audit_map.get(slot.slot_number, (True, ""))
            else:
                is_logical, logic_issue = True, ""
            if is_logical:
                GLOBAL_STEM_REGISTRY.register(draft.stem_latex, slot.chapter_id, slot.sub_skill_id)
                print(
                    f"[gen] slot={slot.slot_number} PASSED SymPy{' + Logic Audit' if needs_llm_audit else ' (SymPy-First Fast Gate)'} (ans={ver.computed_answer}, stars={draft.difficulty_stars})",
                    flush=True,
                )
                return _format_verified_item(slot, ctx, draft, ver)
            last_error = f"Logic audit rejected: {logic_issue}"
            print(f"[gen] slot={slot.slot_number} RETRY (Logic): {logic_issue}", flush=True)
            time.sleep(0.1)
            continue
        last_error = ver.error or "Unknown verification failure"
        set_ai_activity(
            "working",
            f"ข้อ {slot.slot_number} ไม่ผ่าน SymPy • ส่งแก้ใหม่",
            f"สาเหตุ: {last_error[:70]}",
        )
        last_computed = ver.computed_answer
        last_draft_summary = (
            f"- stem_latex: {draft.stem_latex[:300]}\n"
            f"- choices_latex: {draft.choices_latex}\n"
            f"- answer: {draft.answer}"
        )
        print(f"[gen] slot={slot.slot_number} RETRY: {last_error}", flush=True)
        time.sleep(0.1)

    # Fallback: if a stubborn sub-skill still failed, try an alternate sub-skill in the SAME strand so the exam never crashes
    check_cancelled(cancel_event)
    if _allow_fallback:
        syllabus = load_syllabus()
        for strand in syllabus.strands:
            if strand.id == slot.strand_id:
                alt_candidates = [
                    (ch.id, sk)
                    for ch in strand.chapters
                    for sk in ch.sub_skills
                    if sk != slot.sub_skill_id
                ]
                if alt_candidates:
                    alt_ch, alt_sk = random.choice(alt_candidates)
                    print(
                        f"[gen] slot={slot.slot_number} switching stubborn skill '{slot.sub_skill_id}' -> '{alt_sk}' (same strand '{slot.strand_id}')",
                        flush=True,
                    )
                    alt_slot = ExamSlot(
                        slot_number=slot.slot_number,
                        strand_id=slot.strand_id,
                        chapter_id=alt_ch,
                        sub_skill_id=alt_sk,
                        secondary_chapter=None,
                        question_format=slot.question_format,
                        predicted_prob=slot.predicted_prob,
                        rationale=f"Fallback verified skill in strand '{slot.strand_id}'",
                    )
                    return generate_verified_slot(
                        slot=alt_slot,
                        traps=[],
                        historical_stems=historical_stems,
                        provider=provider,
                        model_name=model_name,
                        max_retries=2,
                        _allow_fallback=False,
                        cancel_event=cancel_event,
                    )

    raise RuntimeError(
        f"Slot {slot.slot_number} ({ctx['chapter_th']} / {slot.sub_skill_id}) failed SymPy verification: {last_error}"
    )


def render_mock_exam_markdown(
    exam_items: list[dict],
    title: str = "A-Level Math 1 (ปี 70) Predicted Mock Exam",
) -> str:
    lines = [f"# {title}\n", f"จำนวนข้อที่ผ่าน SymPy Verification: **{len(exam_items)} ข้อ**\n", "---\n"]
    for item in exam_items:
        stars = "★" * int(item.get("difficulty_stars", 3)) + "☆" * (5 - int(item.get("difficulty_stars", 3)))
        lines.append(
            f"### ข้อที่ {item['slot_number']} {stars} (`{item['chapter_id']}` / `{item['sub_skill_id']}` — {item['question_format']})"
        )
        lines.append(item["stem_latex"])
        if item["choices_latex"]:
            for idx, ch in enumerate(item["choices_latex"], start=1):
                lines.append(f"{idx}. {ch}")
        lines.append("")

    lines.extend(["---", "## เฉลยและแนวคิด (SymPy-Verified)\n"])
    for item in exam_items:
        stars = "★" * int(item.get("difficulty_stars", 3)) + "☆" * (5 - int(item.get("difficulty_stars", 3)))
        lines.append(
            f"- **ข้อ {item['slot_number']}** ({stars}): คำตอบ `{item['answer']}` (SymPy=`{item['sympy_computed_answer']}`) | จุดหลอก: {item['key_trap']}"
        )
        lines.append(f"  - แนวคิด: {item['solution_steps_th']}")
    return "\n".join(lines) + "\n"


class GeneratedBatchDraft(BaseModel):
    """Batch of up to 5 questions generated in a single LLM call for 3x-5x higher throughput."""

    items: list[GeneratedQuestionDraft] = Field(..., min_length=1)


def _format_verified_item(slot: ExamSlot, ctx: dict, draft: GeneratedQuestionDraft, ver) -> dict:
    payload = draft.model_dump()
    if slot.question_format == QuestionFormat.MCQ_5 and payload.get("choices_latex"):
        sorted_choices, remapped_ans = sort_mcq_choices_ascending(
            payload["choices_latex"], str(payload["answer"])
        )
        payload["choices_latex"] = sorted_choices
        payload["answer"] = remapped_ans
    return {
        "slot_number": slot.slot_number,
        "strand_id": slot.strand_id,
        "chapter_id": slot.chapter_id,
        "chapter_th": ctx["chapter_th"],
        "sub_skill_id": slot.sub_skill_id,
        "secondary_chapter": slot.secondary_chapter,
        "question_format": slot.question_format.value,
        "predicted_prob": slot.predicted_prob,
        "verified_sympy": True,
        "sympy_computed_answer": ver.computed_answer,
        "max_stem_similarity": ver.max_stem_similarity,
        **payload,
    }


def generate_verified_batch(
    batch_slots: list[ExamSlot],
    trap_bank: dict[str, list[str]],
    historical_stems: list[str],
    provider: str = "gemini",
    model_name: str = "gemini-3.5-flash-lite",
    cancel_event: threading.Event | None = None,
    used_archetypes: set[str] | None = None,
    difficulty_level: int = 1,
) -> list[dict]:
    """Generate a batch of up to 5 questions in ONE API call, verify each with SymPy, and repair only failed slots."""
    check_cancelled(cancel_event)
    diff_prof = get_difficulty_profile(difficulty_level)
    spec_blocks = []
    ctxs = {}
    seeded_slots: set[int] = set()
    batch_used = used_archetypes if used_archetypes is not None else set()
    for idx, slot in enumerate(batch_slots, start=1):
        ctx = _lookup_curriculum_context(slot.chapter_id, slot.sub_skill_id)
        ctxs[slot.slot_number] = ctx
        traps = trap_bank.get(slot.sub_skill_id, []) or ctx["classic_traps"]
        fmt_desc = (
            "mcq_5 (5 ตัวเลือกเป็นตัวเลข/นิพจน์คณิตล้วน ห้ามมีคำภาษาไทยในตัวเลือก, answer='1'-'5')"
            if slot.question_format == QuestionFormat.MCQ_5
            else "numeric Part 2 (choices_latex=[], คำตอบต้องเป็นจำนวนไม่ติดลบ 0<=ans<=9999.99 ทศนิยม<=2ตำแหน่ง ห้ามติดลบ ห้ามเศษส่วน /)"
        )
        seed_obj = _pick_seed_for_slot(
            slot.chapter_id,
            slot.question_format,
            difficulty_level=difficulty_level,
            exclude_archetypes=batch_used,
        )
        if seed_obj is not None:
            seeded_slots.add(slot.slot_number)
        _, div_hint = sample_diversity_directive(slot.chapter_id, used_archetypes=batch_used)
        seed_hint = f" | SymPy-Seed (ใช้ตัวเลขนี้ได้ตรงๆ): {seed_obj.params_description_th}" if seed_obj else ""
        spec_blocks.append(
            f"{idx}) ข้อที่ {slot.slot_number} | บท: {ctx['chapter_th']} ({slot.chapter_id}) | ทักษะ: {slot.sub_skill_id} "
            f"| บทเสริม: {slot.secondary_chapter or '-'} | รูปแบบ: {fmt_desc} | จุดหลอก: {', '.join(traps[:2])} "
            f"| {div_hint} | กฎลอจิกบทนี้: {_chapter_guard_hint(slot.chapter_id)}{seed_hint}"
        )

    prompt = rf"""คุณคือผู้ออกข้อสอบ A-Level คณิตศาสตร์ประยุกต์ 1 รหัสวิชา 61 (สสวท. หลักสูตร 2560) สำหรับปี 2570
จงสร้างโจทย์ใหม่ 100% จำนวน {len(batch_slots)} ข้อ ตามโหมดความยาก: {diff_prof['directive']}
เรียงตามลำดับสเปกด้านล่างในฟิลด์ `items` (ต้องครบ {len(batch_slots)} ข้อ):
{chr(10).join(spec_blocks)}

กฎเหล็กทุกข้อ:
1. ห้ามใช้เนื้อหานอกหลักสูตร 2560 (ห้ามใช้โลปิตาล, กำหนดการเชิงเส้น, ทฤษฎีจำนวนแบบเก่า)
2. คำตอบต้องเป็นค่าตัวเลขเดี่ยวที่แน่นอน ห้ามใช้ช่วง `(a, b)` / `\\infty` หรือข้อความภาษาไทยในตัวเลือก; และสำหรับข้อ `numeric` (Part 2 ระบายตัวเลข) คำตอบต้องอยู่ในช่วง `0 <= ans <= 9999.99` ทศนิยมไม่เกิน 2 ตำแหน่ง ห้ามติดลบและห้ามเป็นเศษส่วน `/` เด็ดขาด
3. ประเมิน `difficulty_stars` 1–5 ดาวให้ตรงความยากจริง (เป้าหมาย {diff_prof['target_stars']})
4. `sympy_code` ต้องใช้ `sp` (sympy) คำนวณค่าคำตอบทางคณิตศาสตร์จริงแล้วกำหนดลงตัวแปร `ANSWER` (ห้ามใส่หมายเลขตัวเลือก 1–5 ลงใน `ANSWER`) และ **ห้ามแก้ตัวเลขโจทย์หรือเขียน "ขออภัย/ปรับโจทย์" ใน `solution_steps_th` เด็ดขาด!**
"""

    verified_by_slot: dict[int, dict] = {}
    try:
        set_ai_activity(
            "working",
            f"สร้างชุดย่อย {len(batch_slots)} ข้อพร้อมกัน (ข้อ {batch_slots[0].slot_number}–{batch_slots[-1].slot_number})",
            f"ส่งคำสั่ง Batch ไปยัง {model_name}...",
        )
        print(
            f"[gen-batch] slots={[s.slot_number for s in batch_slots]} provider={provider}:{model_name} (1 call for {len(batch_slots)} Qs)...",
            flush=True,
        )
        batch_draft = generate_structured_json(
            provider=provider,
            model=model_name,
            prompt=prompt,
            schema_cls=GeneratedBatchDraft,
            temperature=0.4,
            timeout_seconds=90.0 if difficulty_level == 1 else 115.0,
            thinking_budget=diff_prof["thinking_budget"],
            cancel_event=cancel_event,
        )
        check_cancelled(cancel_event)
        set_ai_activity(
            "working",
            f"กำลังตรวจ SymPy ทั้ง {len(batch_slots)} ข้อ",
            "เช็กคำตอบจริง + โดเมน + 5 ตัวเลือกไม่ซ้ำ + LaTeX",
        )
        sympy_passed: list[tuple[ExamSlot, GeneratedQuestionDraft, object]] = []
        batch_seen_stems: list[str] = []
        for slot, raw_draft in zip(batch_slots, batch_draft.items):
            draft = _sanitize_draft(raw_draft)
            ver = verify_generated_question(
                stem_latex=draft.stem_latex,
                question_format=slot.question_format,
                choices_latex=draft.choices_latex,
                expected_answer=draft.answer,
                sympy_code=draft.sympy_code,
                historical_stems=historical_stems,
                solution_steps_th=draft.solution_steps_th,
                chapter_id=slot.chapter_id,
                session_stems=batch_seen_stems,
                check_generated_history=True,
            )
            if ver.passed:
                batch_seen_stems.append(draft.stem_latex)
                sympy_passed.append((slot, draft, ver))
            else:
                print(
                    f"[gen-batch] slot={slot.slot_number} failed SymPy/Quality ({ver.error}) -> repairing solo",
                    flush=True,
                )

        # Option A (Selective Audit): SymPy-First items that passed all 5 Global SymPy Gates +
        # Quality Gates are accepted directly (~50% lower API quota/cost). Only unseeded items
        # require an extra LLM Logic Audit call; failed items still enter solo repair + audit.
        audit_candidates: list[tuple[ExamSlot, GeneratedQuestionDraft, object]] = []
        for slot, draft, ver in sympy_passed:
            if slot.slot_number in seeded_slots:
                GLOBAL_STEM_REGISTRY.register(draft.stem_latex, slot.chapter_id, slot.sub_skill_id)
                verified_by_slot[slot.slot_number] = _format_verified_item(
                    slot, ctxs[slot.slot_number], draft, ver
                )
            else:
                audit_candidates.append((slot, draft, ver))

        if audit_candidates:
            audit_map = audit_math_logic_batch(
                [(s, d) for s, d, _ in audit_candidates],
                provider=provider,
                model_name=model_name,
                cancel_event=cancel_event,
            )
            for slot, draft, ver in audit_candidates:
                ok_logic, issue = audit_map.get(slot.slot_number, (True, ""))
                if ok_logic:
                    GLOBAL_STEM_REGISTRY.register(draft.stem_latex, slot.chapter_id, slot.sub_skill_id)
                    verified_by_slot[slot.slot_number] = _format_verified_item(
                        slot, ctxs[slot.slot_number], draft, ver
                    )
                else:
                    print(
                        f"[gen-batch] slot={slot.slot_number} rejected by Logic Audit ({issue}) -> repairing solo",
                        flush=True,
                    )
    except JobCancelledError:
        raise
    except Exception as exc:
        err_str = str(exc)
        if any(tok in err_str for tok in ("429", "RESOURCE_EXHAUSTED", "โควตารายวัน", "unavailable")):
            raise
        print(f"[gen-batch] batch call failed ({exc}) -> falling back to solo slots", flush=True)

    check_cancelled(cancel_event)
    missing_slots = [s for s in batch_slots if s.slot_number not in verified_by_slot]
    if len(missing_slots) == 1 or provider in ("ollama", "gemini"):
        for slot in missing_slots:
            check_cancelled(cancel_event)
            verified_by_slot[slot.slot_number] = generate_verified_slot(
                slot=slot,
                traps=trap_bank.get(slot.sub_skill_id, []),
                historical_stems=historical_stems,
                provider=provider,
                model_name=model_name,
                cancel_event=cancel_event,
                difficulty_level=difficulty_level,
            )
    elif missing_slots:
        with ThreadPoolExecutor(max_workers=min(4, len(missing_slots))) as repair_pool:
            repair_futs = {
                repair_pool.submit(
                    generate_verified_slot,
                    slot,
                    trap_bank.get(slot.sub_skill_id, []),
                    historical_stems,
                    provider,
                    model_name,
                    5,
                    True,
                    cancel_event,
                    None,
                    difficulty_level,
                ): slot
                for slot in missing_slots
            }
            for fut in as_completed(repair_futs):
                check_cancelled(cancel_event)
                slot = repair_futs[fut]
                verified_by_slot[slot.slot_number] = fut.result()

    return [verified_by_slot[s.slot_number] for s in batch_slots]


def generate_mock_exam(
    slots: list[ExamSlot],
    trap_bank: dict[str, list[str]],
    historical_stems: list[str],
    output_json: Path,
    output_md: Path,
    limit: int | None = None,
    provider: str = "gemini",
    model_name: str = "gemini-3.5-flash-lite",
    max_workers: int | None = None,
    batch_size: int | None = None,
    on_progress: Callable[[int, int, ExamSlot], None] | None = None,
    use_pool: bool = True,
    pool: VerifiedQuestionPool | None = None,
    cancel_event: threading.Event | None = None,
    difficulty_level: int = 1,
) -> list[dict]:
    check_cancelled(cancel_event)
    target_slots = slots[:limit] if limit else slots
    total = len(target_slots)
    eff_workers = max_workers if max_workers is not None else recommend_workers(provider)
    if batch_size is not None:
        eff_batch = batch_size
    else:
        base_batch = recommend_batch_size(provider)
        # Level 2 & 3 questions have longer multi-step scratchpads/solutions -> use 3 Qs/batch to prevent timeouts
        eff_batch = min(base_batch, 3) if (difficulty_level >= 2 and base_batch > 1) else base_batch
    active_pool = pool if pool is not None else GLOBAL_POOL
    results: list[dict] = []
    remaining_slots: list[ExamSlot] = []

    if use_pool:
        used_pool_ids: set[str] = set()
        for slot in target_slots:
            cached = active_pool.checkout(
                sub_skill_id=slot.sub_skill_id,
                question_format=slot.question_format.value,
                slot_number=slot.slot_number,
                exclude_ids=used_pool_ids,
            )
            if cached is not None:
                used_pool_ids.add(cached["pool_id"])
                results.append(cached)
            else:
                remaining_slots.append(slot)
        if results and on_progress:
            on_progress(len(results), total, target_slots[min(len(results) - 1, total - 1)])
    else:
        remaining_slots = list(target_slots)

    rem_total = len(remaining_slots)
    if rem_total == 0:
        pass
    elif rem_total <= 1 or eff_batch <= 1:
        completed = len(results)
        for slot in remaining_slots:
            check_cancelled(cancel_event)
            if on_progress:
                on_progress(completed, total, slot)
            item = generate_verified_slot(
                slot=slot,
                traps=trap_bank.get(slot.sub_skill_id, []),
                historical_stems=historical_stems,
                provider=provider,
                model_name=model_name,
                cancel_event=cancel_event,
                difficulty_level=difficulty_level,
            )
            check_cancelled(cancel_event)
            if use_pool:
                active_pool.add(item)
            results.append(item)
            completed += 1
            if on_progress:
                on_progress(completed, total, slot)
    else:
        chunks = [remaining_slots[i : i + eff_batch] for i in range(0, rem_total, eff_batch)]
        completed = len(results)
        lock = threading.Lock()
        shared_used_archetypes: set[str] = set()
        workers = min(eff_workers, len(chunks))
        if on_progress:
            on_progress(completed, total, remaining_slots[0])
        with ThreadPoolExecutor(max_workers=workers) as pool_exec:
            future_map = {
                pool_exec.submit(
                    generate_verified_batch,
                    chunk,
                    trap_bank,
                    historical_stems,
                    provider,
                    model_name,
                    cancel_event,
                    shared_used_archetypes,
                    difficulty_level,
                ): chunk
                for chunk in chunks
            }
            try:
                for fut in as_completed(future_map):
                    check_cancelled(cancel_event)
                    chunk = future_map[fut]
                    batch_items = fut.result()
                    check_cancelled(cancel_event)
                    if use_pool:
                        active_pool.add_many(batch_items)
                    with lock:
                        results.extend(batch_items)
                        completed += len(batch_items)
                        done_now = completed
                    if on_progress:
                        on_progress(done_now, total, chunk[-1])
            except JobCancelledError:
                for f in future_map:
                    f.cancel()
                raise

    check_cancelled(cancel_event)
    results.sort(key=lambda x: x["slot_number"])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_mock_exam_markdown(results), encoding="utf-8")
    return results
