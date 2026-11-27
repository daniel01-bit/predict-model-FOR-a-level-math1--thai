"""Multimodal PDF/Image -> Structured ExamQuestion JSONL ingestion pipeline (google-genai)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from src.schema import DatasetTier, ExamQuestion, load_syllabus


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-4-hydrogen"]


class RejectedItem(BaseModel):
    question_ref: str = Field(..., description="Question number or short reference")
    reason: str = Field(..., description="Deprecated Syllabus 2551 topic reason")


class RawExtractedQuestion(BaseModel):
    """Flat, Gemini-Developer-API-compatible extraction item before strict ExamQuestion validation."""

    question_id: str
    exam_year_be: int | None = None
    question_number: int | None = None
    question_format: str = Field(..., description="'mcq_5' or 'numeric'")
    question_style: str = Field(..., description="'pure_math', 'word_problem', or 'visual_graph'")
    difficulty: str = Field(..., description="'easy', 'medium', or 'hard'")
    primary_chapter: str
    secondary_chapters: list[str] = Field(default_factory=list)
    sub_skills: list[str]
    stem_latex: str
    choices_latex: list[str] = Field(default_factory=list)
    answer: str
    key_trap: str = ""
    has_diagram: bool = False


class ExtractedBatch(BaseModel):
    """Wrapper schema for structured LLM extraction of multiple questions per PDF/chunk."""

    questions: list[RawExtractedQuestion] = Field(default_factory=list)
    rejected_out_of_syllabus: list[RejectedItem] = Field(default_factory=list)


def resolve_gemini_api_key() -> str:
    """Load GEMINI_API_KEY from environment or local .env without printing secrets."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it in your environment or local .env file (see .env.example)."
        )
    return key


def build_extraction_prompt(
    tier: DatasetTier,
    source_name: str,
    exam_year_be: int | None,
) -> str:
    syllabus = load_syllabus()
    taxonomy_lines = []
    for strand in syllabus.strands:
        for ch in strand.chapters:
            taxonomy_lines.append(f"- {ch.id}: {', '.join(ch.sub_skills)}")
    deprecated_str = ", ".join(syllabus.deprecated_topics_hard_filter)

    return f"""คุณคือผู้เชี่ยวชาญคณิตศาสตร์ ม.ปลาย หลักสูตร สสวท. 2560 (A-Level คณิตศาสตร์ประยุกต์ 1)
งานของคุณ: สกัดข้อสอบทุกข้อจากไฟล์ PDF/รูปภาพนี้ให้เป็น JSON ที่ตรงตาม Schema อย่างเคร่งครัด

ข้อมูลเมตาดาต้าของไฟล์นี้:
- tier: "{tier.value}"
- source_name: "{source_name}"
- exam_year_be: {exam_year_be if exam_year_be is not None else "null"}

กฎเหล็ก (ห้ามละเมิด):
1. เขียนโจทย์ (stem_latex) และตัวเลือก (choices_latex) เป็นภาษาไทย + LaTeX ที่ถูกต้อง 100%
2. primary_chapter และ secondary_chapters ต้องเลือกจากรายการ Chapter ID ด้านล่างเท่านั้น
3. sub_skills ต้องเลือกจาก Closed Taxonomy ด้านล่างเท่านั้น (ห้ามแต่งชื่อใหม่เอง)
4. ถ้าข้อใดเป็นเนื้อหาที่ถูกถอดออกจากหลักสูตร สสวท. 2560 ({deprecated_str}) ห้ามใส่ใน questions ให้แยกไปใส่ใน rejected_out_of_syllabus พร้อมระบุเหตุผล
5. สำหรับ mcq_5 ต้องมี choices_latex ครบ 5 ตัวเลือก และ answer เป็น '1' ถึง '5'
6. สำหรับ numeric ต้องให้ choices_latex เป็น [] และ answer เป็นตัวเลขที่ parse เป็น float ได้
7. ระบุ key_trap (จุดหลอกสำคัญที่ข้อสอบวัด) ให้กระชับใน 1 ประโยค

Closed Taxonomy (chapter_id: sub_skills):
{chr(10).join(taxonomy_lines)}
"""


def load_existing_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    ids: set[str] = set()
    for line in output_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "question_id" in obj:
                ids.add(str(obj["question_id"]))
        except json.JSONDecodeError:
            continue
    return ids


def append_questions_jsonl(output_jsonl: Path, questions: Iterable[ExamQuestion]) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_ids(output_jsonl)
    written = 0
    with output_jsonl.open("a", encoding="utf-8") as fh:
        for q in questions:
            if q.question_id in existing:
                continue
            fh.write(q.model_dump_json() + "\n")
            existing.add(q.question_id)
            written += 1
    return written


def ingest_pdf_file(
    pdf_path: Path,
    tier: DatasetTier,
    source_name: str,
    exam_year_be: int | None,
    output_jsonl: Path,
    model_name: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> dict:
    """Upload a PDF to Gemini Files API, extract structured ExamQuestions, validate, and append."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    api_key = resolve_gemini_api_key()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=90000),
    )
    prompt = build_extraction_prompt(tier=tier, source_name=source_name, exam_year_be=exam_year_be)

    pdf_part = types.Part.from_bytes(
        data=pdf_path.read_bytes(),
        mime_type="application/pdf",
    )
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        m = candidates[attempt % len(candidates)]
        try:
            print(f"[ingest] {pdf_path.name} (yr={exam_year_be}, tier={tier.value}) model={m} attempt={attempt + 1}...", flush=True)
            response = client.models.generate_content(
                model=m,
                contents=[pdf_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedBatch,
                    temperature=0.0,
                    max_output_tokens=16384,
                ),
            )
            batch = ExtractedBatch.model_validate_json(response.text)
            valid_questions: list[ExamQuestion] = []
            dropped_invalid = 0
            import hashlib

            file_tag = hashlib.md5(source_name.encode("utf-8")).hexdigest()[:6]
            for idx, raw_q in enumerate(batch.questions, start=1):
                try:
                    q_year = raw_q.exam_year_be or exam_year_be
                    # For Tier 1, ensure year falls in [2564, 2569]
                    if tier == DatasetTier.TIER_1_TARGET and exam_year_be:
                        q_year = exam_year_be
                    q_num = raw_q.question_number or idx
                    stable_qid = f"{tier.value}_{q_year or 'na'}_{file_tag}_q{q_num:02d}_{idx:02d}"
                    eq = ExamQuestion(
                        question_id=stable_qid,
                        tier=tier,
                        source_name=source_name,
                        exam_year_be=q_year,
                        question_number=raw_q.question_number,
                        question_format=raw_q.question_format,
                        question_style=raw_q.question_style,
                        difficulty=raw_q.difficulty,
                        primary_chapter=raw_q.primary_chapter,
                        secondary_chapters=raw_q.secondary_chapters,
                        sub_skills=raw_q.sub_skills,
                        stem_latex=raw_q.stem_latex,
                        choices_latex=raw_q.choices_latex if raw_q.question_format == "mcq_5" else [],
                        answer=raw_q.answer,
                        key_trap=raw_q.key_trap or None,
                        has_diagram=raw_q.has_diagram,
                    )
                    valid_questions.append(eq)
                except ValidationError:
                    dropped_invalid += 1

            written = append_questions_jsonl(output_jsonl, valid_questions)
            print(
                f"[ingest] OK {pdf_path.name}: extracted={len(batch.questions)}, valid={len(valid_questions)}, new={written}, dropped_invalid={dropped_invalid}, rejected_deprecated={len(batch.rejected_out_of_syllabus)}",
                flush=True,
            )
            return {
                "pdf": str(pdf_path),
                "extracted": len(batch.questions),
                "valid": len(valid_questions),
                "written_new": written,
                "dropped_invalid": dropped_invalid,
                "rejected_out_of_syllabus": len(batch.rejected_out_of_syllabus),
            }
        except (ValidationError, Exception) as exc:
            last_err = exc
            print(f"[ingest] RETRY {pdf_path.name} ({type(exc).__name__}): {str(exc)[:180]}", flush=True)
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to ingest {pdf_path} after {max_retries + 1} attempts: {last_err}")


def batch_ingest_directory(
    raw_dir: Path,
    tier: DatasetTier,
    output_jsonl: Path,
    model_name: str = DEFAULT_MODEL,
) -> list[dict]:
    """Ingest all PDFs in a directory idempotently (skips years already well-populated in JSONL)."""
    import re

    existing_sources: set[str] = set()
    if output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing_sources.add(json.loads(line).get("source_name", ""))
            except json.JSONDecodeError:
                continue

    pdfs = sorted(raw_dir.glob("*.pdf"))
    reports: list[dict] = []
    for pdf in pdfs:
        source_name = pdf.stem
        if source_name in existing_sources:
            print(f"[ingest] SKIP already ingested: {pdf.name}", flush=True)
            continue
        m4 = re.search(r"(25[3-6][0-9])", pdf.name)
        yr = int(m4.group(1)) if m4 else None
        if yr is None:
            m2 = re.search(r"(?:^|[^0-9])(6[4-9]|5[5-9]|6[0-3]|4[1-9]|5[0-4])(?:[^0-9]|$)", pdf.name)
            yr = (2500 + int(m2.group(1))) if m2 else None
        try:
            rep = ingest_pdf_file(
                pdf_path=pdf,
                tier=tier,
                source_name=source_name,
                exam_year_be=yr,
                output_jsonl=output_jsonl,
                model_name=model_name,
            )
            reports.append(rep)
        except Exception as exc:
            print(f"[ingest] FAILED {pdf.name}: {exc}", flush=True)
            reports.append({"pdf": str(pdf), "error": str(exc)})
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Thai Math PDF into validated JSONL.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to single input PDF")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Directory of PDFs to batch ingest")
    parser.add_argument("--tier", required=True, choices=[t.value for t in DatasetTier])
    parser.add_argument("--source-name", type=str, default=None)
    parser.add_argument("--year-be", type=int, default=None, help="Buddhist Era year e.g. 2568")
    parser.add_argument("--out", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if args.batch_dir:
        results = batch_ingest_directory(
            raw_dir=args.batch_dir,
            tier=DatasetTier(args.tier),
            output_jsonl=args.out,
            model_name=args.model,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if not args.pdf or not args.source_name:
        parser.error("Provide either --batch-dir or both --pdf and --source-name")

    result = ingest_pdf_file(
        pdf_path=args.pdf,
        tier=DatasetTier(args.tier),
        source_name=args.source_name,
        exam_year_be=args.year_be,
        output_jsonl=args.out,
        model_name=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
