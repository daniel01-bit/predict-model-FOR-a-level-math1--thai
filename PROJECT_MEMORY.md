# PROJECT_MEMORY.md — Predict A-Level Math 1 (Neuro-Symbolic Exam Studio)

**Project Name:** `Predict A-Level Math 1` (`P6`)  
**Current Version / Phase:** `v3.0 Release-Ready (Trained, Backtested & GitHub-Staged)`  
**Last Updated:** `2026-11-27`

---

## 1. Core Objective
- Build a **Neuro-Symbolic AI system** combining a **3-Tier Weighted Multi-Source Forecasting Model** (validated via strict zero-leakage Walk-Forward Ablation Backtest) with **SymPy-First Parameter Seeders**, a **Sandboxed Persistent SymPy Verification Gate**, an **Independent LLM Logic Auditor**, and an **Official A-Level 61 A4 PDF Renderer** + Gemini-style Dark Web Dashboard.

---

## 2. Non-Goals
- ❌ No LLM fine-tuning (Small-N exam banks risk verbatim memorization instead of synthesis).
- ❌ No naive equal-weight pooling across curriculum regimes (IPST 2560 vs legacy Saman/Ent).
- ❌ No unverified LLM math answers shipped to users (every item must pass SymPy + Quality + Logic gates).

---

## 3. Tech Stack & Hard Constraints
- **Core Stack:** Python 3.15, FastAPI, Uvicorn, Pydantic v2, SymPy, `google-genai`, Multi-Provider HTTP Router (Gemini/OpenAI/Anthropic/OpenRouter/Ollama), Headless Chromium/Edge PDF Rendering, `pytest`
- **Blueprint Constraint (MyTCAS A-Level 61):** 30 questions (25 MCQ × 3 pts + 5 Numeric × 5 pts = 100 pts) across 4 strands (`15-17`, `3-5`, `6-8`, `2-4`).
- **Zero-Leakage Constraint:** Both model fitting (`exam_year_be < target_year_be`) and `expected_hits` probability scaling in `backtest.py` use strictly historical training folds.

---

## 4. Phase Status (All 4 Phases Complete)
- `[x]` **Phase 1 (v1.0):** Closed Syllabus 2560 Taxonomy (`config/syllabus_2560.json`), IPST 2560 Curriculum DB (`data/curriculum/ipst_2560_curriculum.json`), Pydantic Contracts (`src/schema.py`), Multimodal PDF Ingestion (`src/ingest.py`), Tier 1 ingested (**115 real A-Level 2566–2569 questions**).
- `[x]` **Phase 2 (v1.5):** Tier 3 ingested (**109 real legacy Saman/chapter questions**, total **224 verified questions** in `287.4 KB` JSONL).
- `[x]` **Phase 3 (v2.0):** Zero-Leakage Walk-Forward Ablation Backtest (`src/backtest.py`) — **`model_c_full_3tier` beats Naive Baseline** on both `Recall@15` and `Brier Score` -> A-Level 70 Forecast generated (`data/reports/forecast_2570.md`).
- `[x]` **Phase 4 (v3.0):** SymPy-First Seeders (`src/sympy_seeds.py`), Sandboxed Persistent SymPy Worker (`src/verifier.py`), Multi-Provider Circuit-Breaker Router (`src/llm_router.py`), A-Level 61 Parallel PDF Renderer (`src/pdf_renderer.py`), and Gemini-Dark Interactive Dashboard (`src/server.py`, `web/index.html`).

---

## 5. Next Immediate Action
- Review `README.md`, `.gitignore`, `.env.example`, and `requirements.txt`, then initialize git and push to GitHub when ready (`git init && git add . && git commit && git push`).
