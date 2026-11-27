"""FastAPI dashboard server for Predict A-Level Math 1 (v3.0)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.backtest import load_jsonl_dataset
from src.generator import (
    GenerationMode,
    build_slots_for_mode,
    generate_mock_exam,
    generate_verified_batch,
)
from src.llm_router import (
    JobCancelledError,
    get_quota_status,
    list_provider_status,
    set_ai_activity,
    set_runtime_api_key,
)
from src.model import (
    build_trap_bank,
    compute_cross_topic_pairs,
    predict_subskill_distribution,
)
from src.pdf_renderer import render_exam_bundle
from src.question_pool import GLOBAL_POOL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
JOBS_DIR = PROJECT_ROOT / "data" / "reports" / "jobs"

app = FastAPI(title="Predict A-Level Math 1 Dashboard", version="3.0")


@dataclass
class JobState:
    job_id: str
    mode: GenerationMode
    provider: str
    model: str
    target_year_be: int
    status: str = "queued"          # queued | running | rendering_pdf | done | error | cancelled
    completed: int = 0
    total: int = 1
    current_label: str = ""
    error: str | None = None
    items: list[dict] = field(default_factory=list)
    exam_pdf: Path | None = None
    solutions_pdf: Path | None = None
    exam_html: Path | None = None
    solutions_html: Path | None = None
    finished_stamp: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)


JOBS: dict[str, JobState] = {}
_FORECAST_CACHE: dict[tuple[int, float], tuple[dict[str, float], dict, dict, list[str]]] = {}
_FORECAST_LOCK = threading.Lock()
_REFILL_LOCK = threading.Lock()
_POOL_SEEDED = False


def _ensure_pool_seeded() -> None:
    global _POOL_SEEDED
    if not _POOL_SEEDED:
        _POOL_SEEDED = True
        GLOBAL_POOL.seed_from_jobs_dir(JOBS_DIR)


def _get_forecast_context(target_year_be: int) -> tuple[dict[str, float], dict, dict, list[str]]:
    parsed_paths = sorted(PARSED_DIR.glob("*.jsonl"))
    latest_mtime = max((p.stat().st_mtime for p in parsed_paths), default=0.0)
    key = (target_year_be, latest_mtime)
    with _FORECAST_LOCK:
        cached = _FORECAST_CACHE.get(key)
        if cached is not None:
            return cached
        questions = load_jsonl_dataset(parsed_paths)
        probs = predict_subskill_distribution(
            questions, target_year_be=target_year_be, mode="model_c_full_3tier"
        )
        cross_pairs = compute_cross_topic_pairs(questions, target_year_be=target_year_be)
        traps = build_trap_bank(questions, target_year_be=target_year_be)
        historical_stems = [q.stem_latex for q in questions]
        ctx = (probs, cross_pairs, traps, historical_stems)
        _FORECAST_CACHE.clear()
        _FORECAST_CACHE[key] = ctx
        return ctx


def _schedule_pool_refill(
    consumed_slots: list,
    traps: dict,
    historical_stems: list[str],
    provider: str,
    model: str,
) -> None:
    """Silently top up consumed pool slots in the background (disabled on Ollama & Gemini Free Tier to prevent hidden quota burn)."""
    if not consumed_slots or provider in ("ollama", "gemini"):
        return
    if not _REFILL_LOCK.acquire(blocking=False):
        return

    def _worker() -> None:
        try:
            refill_slots = consumed_slots[:5]
            new_items = generate_verified_batch(
                batch_slots=refill_slots,
                trap_bank=traps,
                historical_stems=historical_stems,
                provider=provider,
                model_name=model,
            )
            GLOBAL_POOL.add_many(new_items)
        except Exception as exc:
            print(f"[pool-refill] skipped ({exc})", flush=True)
        finally:
            _REFILL_LOCK.release()

    threading.Thread(target=_worker, daemon=True).start()


class GenerateRequest(BaseModel):
    mode: GenerationMode
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    target_year_be: int = 2570


class ApiKeyRequest(BaseModel):
    provider: str
    api_key: str


def _format_user_friendly_error(exc: Exception, provider: str = "", model: str = "") -> str:
    """Translate raw API/system exceptions into clear, actionable Thai warnings."""
    raw = str(exc)
    low = raw.lower()
    model_label = f" ({model})" if model else ""

    if "โควตารายวัน" in raw or ("429" in low and ("per day" in low or "perday" in low or "requestsperday" in low)):
        return (
            f"โควตาฟรีรายวันของโมเดล{model_label} หมดแล้วครับ\n"
            f"แนะนำ: สลับไปใช้รุ่น Gemini อื่นที่เมนูด้านบน (เช่น gemini-3.5-flash-lite) "
            f"หรือสลับไปใช้ Local Ollama ฟรีบนเครื่องได้เลย"
        )
    if "429" in low or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
        return (
            f"ตอนนี้ใช้งานโมเดล{model_label} ถี่เกินโควตาฟรีชั่วคราว\n"
            f"แนะนำ: รอประมาณ 30–60 วินาทีแล้วกดลองใหม่ หรือสลับไปใช้รุ่นอื่นที่เมนูด้านบนครับ"
        )
    if "api_key" in low or "api key" in low or "401" in low or "403" in low or "unauthenticated" in low or "permission_denied" in low:
        return (
            f"API Key ของ {provider or 'ผู้ให้บริการ'} ไม่ถูกต้องหรือยังไม่ได้ตั้งค่า\n"
            f"แนะนำ: กดปุ่ม 'API Keys' ด้านบนเพื่อตรวจสอบหรือใส่คีย์ใหม่อีกครั้งครับ"
        )
    if "404" in low or "not_found" in low or "not found" in low:
        return (
            f"ไม่พบโมเดล{model_label} บนระบบของผู้ให้บริการ\n"
            f"แนะนำ: เลือกโมเดลอื่นจากเมนูด้านบนแล้วลองใหม่ครับ"
        )
    if provider == "ollama" and ("connection" in low or "11434" in low or "refused" in low or "ollama error" in low):
        return (
            f"ไม่สามารถเชื่อมต่อ Local Ollama{model_label} ได้\n"
            f"แนะนำ: ตรวจสอบว่าเปิดโปรแกรม Ollama อยู่บนเครื่อง หรือลองสลับไปใช้โมเดลอื่นครับ"
        )
    if "timed out" in low or "timeout" in low or "deadline" in low:
        return (
            f"ระบบใช้เวลานานเกินกำหนด (Timeout){model_label}\n"
            f"แนะนำ: กดลองใหม่อีกครั้ง หากใช้โหมด MOCK-TEST 30 ข้อ ลองเลือก MINI-TEST 15 ข้อหรือรุ่น Flash-Lite ที่เร็วกว่าครับ"
        )
    if "json" in low or "validation error" in low or "eof while parsing" in low:
        return (
            f"โมเดล{model_label} ส่งคำตอบกลับมาไม่ครบถ้วน\n"
            f"แนะนำ: กดสร้างใหม่อีกครั้ง หรือเปลี่ยนไปใช้โมเดลรุ่นที่ใหญ่ขึ้น/เสถียรกว่าครับ"
        )
    if "500" in low or "502" in low or "503" in low or "504" in low or "unavailable" in low:
        return (
            f"เซิร์ฟเวอร์ของโมเดล{model_label} กำลังขัดข้องชั่วคราว\n"
            f"แนะนำ: รอสักครู่แล้วกดลองใหม่ หรือสลับไปใช้โมเดลอื่นด้านบนครับ"
        )
    if "sympy" in low or "verification" in low:
        import re

        m_slot = re.search(r"Slot\s+\d+\s*\(([^)]+)\)", raw)
        m_reason = re.search(r"failed SymPy verification:\s*(.+)$", raw)
        chapter_info = f"\nบทที่พบปัญหา: {m_slot.group(1)}" if m_slot else ""
        reason_info = f"\nสาเหตุย่อ: {m_reason.group(1)[:140]}" if m_reason else ""
        return (
            "ระบบตรวจความถูกต้องทางคณิตศาสตร์ (SymPy) พบว่าโจทย์ที่สุ่มได้ยังไม่สมบูรณ์"
            f"{chapter_info}{reason_info}\n"
            "แนะนำ: กดลองใหม่อีกครั้ง ระบบจะสุ่มและตรวจคำตอบใหม่ให้อัตโนมัติครับ"
        )
    return (
        "ไม่สามารถสร้างข้อสอบได้ในขณะนี้\n"
        "แนะนำ: กดลองใหม่อีกครั้ง หรือลองสลับโมเดลที่เมนูด้านบนครับ"
    )


def _run_generation_job(job: JobState) -> None:
    try:
        if job.cancel_event.is_set():
            raise JobCancelledError("Job cancelled before start")
        job.status = "running"
        set_ai_activity("working", f"เตรียมโครงสร้างข้อสอบ ({job.model})...")
        _ensure_pool_seeded()
        probs, cross_pairs, traps, historical_stems = _get_forecast_context(job.target_year_be)
        slots = build_slots_for_mode(
            mode=job.mode,
            subskill_probs=probs,
            cross_topic_pairs=cross_pairs,
        )

        job.total = len(slots)
        out_dir = JOBS_DIR / job.job_id

        def on_progress(done: int, total: int, slot) -> None:
            if job.cancel_event.is_set():
                raise JobCancelledError("Job cancelled by user")
            job.completed = done
            job.total = total
            job.current_label = f"ข้อ {slot.slot_number}/{total} • {slot.chapter_id} ({slot.sub_skill_id})"
            set_ai_activity("working", f"สร้าง+ตรวจ SymPy ข้อ {done}/{total}")

        items = generate_mock_exam(
            slots=slots,
            trap_bank=traps,
            historical_stems=historical_stems,
            output_json=out_dir / "exam.json",
            output_md=out_dir / "exam.md",
            provider=job.provider,
            model_name=job.model,
            on_progress=on_progress,
            use_pool=True,
            cancel_event=job.cancel_event,
        )
        if job.cancel_event.is_set():
            raise JobCancelledError("Job cancelled by user")
        pool_hits = [s for s, it in zip(slots, items) if it.get("from_pool")]
        if pool_hits:
            _schedule_pool_refill(pool_hits, traps, historical_stems, job.provider, job.model)
        job.items = items
        if job.mode in ("mini_test", "mock_test"):
            if job.cancel_event.is_set():
                raise JobCancelledError("Job cancelled before PDF render")
            job.status = "rendering_pdf"
            job.current_label = "กำลังจัดหน้า PDF มาตรฐาน A-Level รหัสวิชา 61..."
            set_ai_activity("working", "กำลังจัดหน้า PDF A-Level 61...")

            bundle = render_exam_bundle(
                items=items,
                mode=job.mode,
                target_year_be=job.target_year_be,
                out_dir=out_dir,
            )
            if job.cancel_event.is_set():
                raise JobCancelledError("Job cancelled during PDF render")
            job.exam_pdf = bundle["exam_pdf"]
            job.solutions_pdf = bundle["solutions_pdf"]
            job.exam_html = bundle["exam_html"]
            job.solutions_html = bundle["solutions_html"]
            job.current_label = "สร้างข้อสอบและเฉลย PDF เรียบร้อยแล้ว"
        else:
            job.current_label = "พร้อมทำโจทย์บนหน้าเว็บแล้ว"
        now = datetime.now()
        be_yy = (now.year + 543) % 100
        job.finished_stamp = f"{now:%H.%M-%d-%m}-{be_yy:02d}"
        job.status = "done"
        set_ai_activity("idle", "พร้อมใช้งาน")
    except JobCancelledError:
        print(f"[job-cancelled] {job.job_id} ({job.provider}/{job.model}) stopped immediately", flush=True)
        job.status = "cancelled"
        job.current_label = "ยกเลิกการสร้างแล้ว"
        job.error = None
        set_ai_activity("idle", "พร้อมใช้งาน")
    except Exception as exc:
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.current_label = "ยกเลิกการสร้างแล้ว"
            job.error = None
            set_ai_activity("idle", "พร้อมใช้งาน")
            return
        print(f"[job-error] {job.job_id} ({job.provider}/{job.model}): {exc}", flush=True)
        job.status = "error"
        job.error = _format_user_friendly_error(exc, job.provider, job.model)
        set_ai_activity("idle", "พร้อมใช้งาน")


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/providers")
def api_providers() -> dict:
    return {"providers": list_provider_status()}


@app.get("/api/quota")
def api_quota(
    provider: str = Query(default="gemini"),
    model: str = Query(default="gemini-3.5-flash-lite"),
) -> dict:
    current = get_quota_status(provider, model)
    alternatives: list[dict] = []
    for p in list_provider_status():
        if not p["configured"]:
            continue
        for m in p["models"]:
            if p["id"] == provider and m == model:
                continue
            st = get_quota_status(p["id"], m)
            if st["status"] in ("ok", "unlimited"):
                alternatives.append(
                    {
                        "provider": p["id"],
                        "model": m,
                        "label": f"{p['label']} — {m}",
                        "value": f"{p['id']}::{m}",
                        "pct": st["pct"],
                        "unlimited": st["unlimited"],
                    }
                )
    alternatives.sort(
        key=lambda a: (
            0 if (current["status"] == "exhausted_rpd" and a["provider"] != provider) else 1,
            0 if a["unlimited"] else 1,
            -a["pct"],
        )
    )
    return {"quota": current, "alternatives": alternatives[:3]}


@app.get("/api/pool")
def api_pool_stats() -> dict:
    _ensure_pool_seeded()
    return GLOBAL_POOL.stats()


@app.post("/api/keys")
def api_set_key(req: ApiKeyRequest) -> dict:
    try:
        set_runtime_api_key(req.provider, req.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "providers": list_provider_status()}


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> dict:
    # Auto-cancel any prior in-flight job so zombie jobs never pile up on Ollama/API queues
    for prev in list(JOBS.values()):
        if prev.status in ("queued", "running", "rendering_pdf"):
            prev.cancel_event.set()
            prev.status = "cancelled"
            prev.current_label = "ยกเลิกงานเก่าอัตโนมัติ"

    job_id = uuid.uuid4().hex[:12]
    total_map = {"question": 1, "mini_test": 15, "mock_test": 30}
    job = JobState(
        job_id=job_id,
        mode=req.mode,
        provider=req.provider,
        model=req.model,
        target_year_be=req.target_year_be,
        total=total_map[req.mode],
    )
    JOBS[job_id] = job
    thread = threading.Thread(target=_run_generation_job, args=(job,), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": job.status, "total": job.total}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_event.set()
    if job.status in ("queued", "running", "rendering_pdf"):
        job.status = "cancelled"
        job.current_label = "ยกเลิกการสร้างแล้ว"
        job.error = None
    set_ai_activity("idle", "พร้อมใช้งาน")
    return {"ok": True, "job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "mode": job.mode,
        "provider": job.provider,
        "model": job.model,
        "status": job.status,
        "completed": job.completed,
        "total": job.total,
        "current_label": job.current_label,
        "error": job.error,
        "exam_pdf_url": f"/api/jobs/{job.job_id}/pdf/exam" if job.exam_pdf else None,
        "solutions_pdf_url": f"/api/jobs/{job.job_id}/pdf/solutions" if job.solutions_pdf else None,
        "exam_preview_url": f"/api/jobs/{job.job_id}/html/exam" if job.exam_html else None,
        "solutions_preview_url": f"/api/jobs/{job.job_id}/html/solutions" if job.solutions_html else None,
        "single_question": job.items[0] if (job.mode == "question" and job.items) else None,
        "questions_summary": [
            {
                "slot_number": q["slot_number"],
                "chapter_th": q.get("chapter_th", q["chapter_id"]),
                "sub_skill_id": q["sub_skill_id"],
                "difficulty_stars": q.get("difficulty_stars", 3),
                "question_format": q["question_format"],
            }
            for q in job.items
        ],
    }


@app.get("/api/jobs/{job_id}/pdf/{doc_type}")
def api_job_pdf(
    job_id: str,
    doc_type: str,
    download: int = Query(default=0),
) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="PDF not ready")
    path = job.exam_pdf if doc_type == "exam" else job.solutions_pdf if doc_type == "solutions" else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    stamp = job.finished_stamp or datetime.now().strftime("%H.%M-%d-%m")
    filename = f"ALevel61_{job.mode}_{stamp}_{doc_type}.pdf"
    headers = (
        {"Content-Disposition": f'attachment; filename="{filename}"'}
        if download
        else {"Content-Disposition": f'inline; filename="{filename}"'}
    )
    return FileResponse(path, media_type="application/pdf", headers=headers)


@app.get("/api/jobs/{job_id}/html/{doc_type}", response_class=HTMLResponse)
def api_job_html(job_id: str, doc_type: str) -> HTMLResponse:
    job = JOBS.get(job_id)
    if not job or job.status != "done":
        raise HTTPException(status_code=404, detail="Preview not ready")
    path = job.exam_html if doc_type == "exam" else job.solutions_html if doc_type == "solutions" else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


def run_dev_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
    import uvicorn

    if reload:
        uvicorn.run(
            "src.server:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "web")],
        )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dev_server()
