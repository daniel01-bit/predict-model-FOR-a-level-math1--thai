"""Authentic Thai A-Level รหัสวิชา 61 (คณิตศาสตร์ประยุกต์ 1) HTML & A4 PDF Renderer."""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal


DocType = Literal["exam", "solutions"]

# Difficulty level accent colors matching the Web Dashboard (1=Olive, 2=Purple, 3=Crimson)
DIFFICULTY_COLORS: dict[int, str] = {
    1: "#556b2f",  # Level 1 — Olive Green (A-Level Standard)
    2: "#7e22ce",  # Level 2 — Royal Purple (+50% Hard)
    3: "#b91c1c",  # Level 3 — Crimson Red (+125% Nightmare)
}


def _difficulty_color(difficulty_level: int = 1) -> str:
    """Return the hex color for the given difficulty tier (1, 2, or 3)."""
    return DIFFICULTY_COLORS.get(int(difficulty_level), DIFFICULTY_COLORS[1])


def _format_generated_timestamp(dt: datetime | None = None) -> str:
    """Format generation timestamp as 'HH : MM / DD-MM-YY' (2-digit Thai BE year)."""
    now = dt or datetime.now()
    be_yy = (now.year + 543) % 100
    return f"{now:%H} : {now:%M} / {now:%d}-{now:%m}-{be_yy:02d}"

BROWSER_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

MODE_METADATA = {
    "question": {
        "title_th": "ชุดฝึกหัดจำลองรายข้อ (Single Question Simulation)",
        "duration_min": 5,
        "badge": "QUESTION • 1 ข้อ",
    },
    "mini_test": {
        "title_th": "แบบทดสอบย่อยหลากหลายบท (Mini-Test 15 ข้อ)",
        "duration_min": 45,
        "badge": "MINI-TEST • 15 ข้อ",
    },
    "mock_test": {
        "title_th": "แบบทดสอบจำลองเสมือนจริงทำนายแนวข้อสอบ A-Level 70 (Full Mock-Test)",
        "duration_min": 90,
        "badge": "MOCK-TEST • 30 ข้อ (Predicted A-Level 70)",
    },
}


def find_headless_browser() -> Path:
    for path in BROWSER_CANDIDATES:
        if path.exists():
            return path
    raise RuntimeError(
        "No Chromium/Edge browser found for PDF rendering. Install Microsoft Edge or Google Chrome."
    )


def _stars_html(stars: int) -> str:
    n = max(1, min(5, int(stars)))
    return f'<span class="stars" title="ความยาก {n}/5 ดาว">{"★" * n}{"☆" * (5 - n)}</span>'


# Brace-argument commands have NO trailing \b (since '}' is non-word); symbol commands keep trailing \b.
_LATEX_BRACE_CMD = r"\\(?:frac|dfrac|tfrac)\{[^{}]+\}\{[^{}]+\}|\\(?:sqrt|vec|hat|bar|overline|mathbf|mathrm|text)\{[^{}]+\}"
_LATEX_WORD_CMD = (
    r"\\(?:infty|pi|theta|alpha|beta|gamma|sigma|mu|lambda|Delta|times|cdot|pm|le|leq|ge|geq|neq|approx|"
    r"to|rightarrow|leftrightarrow|land|lor|neg|sim|forall|exists|in|notin|subset|cup|cap|ln|log|sin|cos|tan|"
    r"arcsin|arccos|arctan|det|sum|int)\b"
)
# Match full bare math equations containing at least one LaTeX command (e.g. "\frac{(x-4)^2}{169} + \frac{(y+1)^2}{144} = 1")
_BARE_MATH_EXPR_RE = re.compile(
    rf"(?:[A-Za-z0-9\(\)\[\]\+\-\*/=\^_,\.\s]*?(?:{_LATEX_BRACE_CMD}|{_LATEX_WORD_CMD})(?:[A-Za-z0-9\(\)\[\]\{{\}}\+\-\*/=\^_,\.\s\\]*?(?:{_LATEX_BRACE_CMD}|{_LATEX_WORD_CMD}))*[A-Za-z0-9\(\)\[\]\+\-\*/=\^_\.]*(?:\s*[=<>≤≥]\s*[-\+]?[A-Za-z0-9\.\(\)\^_]+)?)"
)
_SINGLE_LATEX_CMD_RE = re.compile(rf"(?:{_LATEX_BRACE_CMD}|{_LATEX_WORD_CMD})")
_BARE_MATH_HINT_RE = re.compile(rf"(?:{_LATEX_BRACE_CMD}|{_LATEX_WORD_CMD}|[A-Za-z0-9\)]\s*[\^_]\s*[\{{\-\+A-Za-z0-9]|!=)")


def _wrap_prose_segment(seg: str) -> str:
    # 0) Convert raw ASCII 'Sigma_{...}^{infinity}' into proper LaTeX \sum_{...}^{\infty}
    if "Sigma_" in seg:
        seg = re.sub(r"Sigma_\{([^{}]+)\}\^\{infinity\}", r"\\sum_{\1}^{\\infty}", seg)

    # 1) Unwrap accidental \text{...} written outside $...$ (which otherwise splits braces across Thai text -> red KaTeX!)
    if r"\text{" in seg:
        for _ in range(3):
            nxt = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
            if nxt == seg:
                break
            seg = nxt

    if "\\" not in seg and "^" not in seg and "_" not in seg and "!=" not in seg:
        return seg

    # 2) Split on Thai characters so we only wrap clean non-Thai math spans
    chunks = re.split(r"([\u0E00-\u0E7F]+)", seg)
    for idx in range(0, len(chunks), 2):
        part = chunks[idx]
        if not _BARE_MATH_HINT_RE.search(part):
            continue
        lead_ws = part[: len(part) - len(part.lstrip())]
        trail_ws = part[len(part.rstrip()) :]
        core = part.strip()
        if not core:
            continue
        # Only wrap if braces are balanced (never feed unbalanced { or } into KaTeX)
        if core.count("{") != core.count("}"):
            continue
        core = core.replace("!=", r"\neq ")
        chunks[idx] = f"{lead_ws}${core}${trail_ws}"
    return "".join(chunks)


def _auto_wrap_bare_latex(text: str, is_choice: bool = False) -> str:
    """Wrap bare LaTeX commands outside $...$ so KaTeX always renders math symbols cleanly."""
    if not text:
        return text
    stripped = text.strip()
    if is_choice:
        if "$" not in stripped and ("\\" in stripped or "^" in stripped or "_" in stripped):
            return f"${stripped}$"
        return stripped

    # Split on existing $...$ / $$...$$ segments so we never double-wrap valid math
    parts = re.split(r"(\$\$[\s\S]*?\$\$|\$[^$]+\$)", text)
    for i in range(0, len(parts), 2):
        parts[i] = _wrap_prose_segment(parts[i])
    return "".join(parts)


def _escape_keep_latex(text: str, is_choice: bool = False) -> str:
    cleaned = re.sub(r"(?:\\text\{)?\b(?:conj|conjugate)\b\}?\s*\(([^()]+)\)", r"\\overline{\1}", text)
    wrapped = _auto_wrap_bare_latex(cleaned, is_choice=is_choice)
    return html.escape(wrapped).replace("\n", "<br>")


def _base_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sarabun:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}},{{left:'\\\\(',right:'\\\\)',display:false}},{{left:'\\\\[',right:'\\\\]',display:true}}], throwOnError:false}});"></script>
<style>
  /* A4 print layout with running page numbers at top-right corner */
  @page {{
    size: A4;
    margin: 18mm 18mm 20mm 18mm;
    @top-right {{
      content: "หน้า " counter(page) " / " counter(pages);
      font-family: 'Sarabun', 'TH Sarabun New', 'Tahoma', sans-serif;
      font-size: 10pt;
      color: #444;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Sarabun', 'TH Sarabun New', 'Tahoma', sans-serif;
    font-size: 13.5pt;
    line-height: 1.55;
    color: #111;
    margin: 0;
    background: #fff;
  }}
  .cover-page {{
    page-break-after: always;
    border: 1.8px solid #111;
    padding: 20px 26px;
    min-height: 252mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .cover-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #111;
    padding-bottom: 12px;
  }}
  .subject-code {{
    border: 2px solid #111;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 15pt;
    text-align: center;
  }}
  .cover-title {{
    text-align: center;
    margin: 16px 0 10px;
  }}
  .cover-title h1 {{
    font-size: 20pt;
    margin: 0;
    font-weight: 700;
  }}
  .cover-title p {{
    margin: 4px 0;
    font-size: 13.5pt;
  }}
  .student-box {{
    border: 1.2px solid #222;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 13pt;
  }}
  .student-row {{
    display: flex;
    gap: 16px;
    margin-bottom: 8px;
  }}
  .student-row:last-child {{ margin-bottom: 0; }}
  .field-line {{
    flex: 1;
    border-bottom: 1px dotted #444;
    padding-bottom: 2px;
  }}
  .instructions {{
    border-top: 1.2px solid #222;
    padding-top: 10px;
    font-size: 12.5pt;
  }}
  .instructions h3 {{
    margin: 0 0 6px;
    font-size: 14pt;
    text-decoration: underline;
  }}
  .instructions ol {{
    margin: 0;
    padding-left: 22px;
  }}
  .instructions li {{ margin-bottom: 4px; }}
  .cover-footer {{
    text-align: center;
    border-top: 2px solid #111;
    padding-top: 10px;
    font-weight: 700;
    font-size: 14pt;
    letter-spacing: 0.4px;
  }}
  .running-header {{
    display: flex;
    justify-content: space-between;
    border-bottom: 1.2px solid #111;
    padding-bottom: 4px;
    margin-bottom: 14px;
    font-size: 11.5pt;
    font-weight: 600;
  }}
  .part-banner {{
    background: #f2f2f2;
    border-left: 4px solid #111;
    padding: 6px 12px;
    margin: 14px 0 12px;
    font-weight: 700;
    font-size: 13.5pt;
  }}
  .question-block {{
    margin-bottom: 18px;
    page-break-inside: avoid;
  }}
  .q-meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}
  .q-num {{
    font-weight: 700;
    font-size: 14pt;
  }}
  .q-tags {{
    font-size: 10.5pt;
    color: #444;
  }}
  .stars {{
    font-size: 12pt;
    letter-spacing: 1px;
    color: #111;
    margin-left: 8px;
  }}
  .q-stem {{
    margin-bottom: 8px;
    text-align: justify;
  }}
  .choices {{
    list-style: none;
    padding-left: 18px;
    margin: 0;
    display: grid;
    grid-template-columns: 1fr;
    gap: 3px;
  }}
  .numeric-box {{
    margin-top: 8px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12pt;
    color: #222;
  }}
  .digit-cell {{
    width: 22px;
    height: 24px;
    border: 1.2px solid #222;
    display: inline-block;
  }}
  .sol-box {{
    border: 1px solid #ccc;
    border-left: 3px solid #111;
    padding: 8px 12px;
    margin-top: 6px;
    background: #fafafa;
    font-size: 12.5pt;
  }}
  .trap-note {{
    margin-top: 6px;
    font-size: 11.5pt;
    color: #333;
  }}
  table.summary-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 18px;
    font-size: 12pt;
  }}
  table.summary-table th, table.summary-table td {{
    border: 1px solid #222;
    padding: 5px 8px;
    text-align: center;
  }}
  table.summary-table th {{
    background: #efefef;
    font-weight: 700;
  }}
</style>
</head>
<body>
"""


def render_exam_html(
    items: list[dict],
    mode: str = "mock_test",
    target_year_be: int = 2570,
    difficulty_level: int = 1,
    generated_at: str | None = None,
) -> str:
    """Build print-ready A4 HTML for the Exam Paper with cover page, difficulty-colored MOCK badge, and KaTeX."""
    meta = MODE_METADATA.get(mode, MODE_METADATA["mock_test"])
    mcq_items = [q for q in items if q["question_format"] == "mcq_5"]
    num_items = [q for q in items if q["question_format"] == "numeric"]
    total_pts = len(mcq_items) * 3 + len(num_items) * 5
    diff_color = _difficulty_color(difficulty_level)
    gen_time = generated_at or _format_generated_timestamp()

    parts = [_base_head(f"MOCK • คณิตศาสตร์ประยุกต์ 1 ({meta['badge']})")]

    # Cover Page: generation timestamp, difficulty-colored MOCK box, Name-only student field, streamlined instructions
    parts.append(f"""
<div class="cover-page">
  <div>
    <div class="cover-top">
      <div>
        <div style="font-weight:700; font-size:14pt;">แบบทดสอบจำลองมาตรฐาน A-Level (สสวท. หลักสูตร 2560)</div>
        <div style="font-size:12pt; color:#333;">{html.escape(meta['title_th'])} • {html.escape(gen_time)}</div>
      </div>
      <!-- Cover top-right badge: MOCK colored by difficulty level (1=Olive, 2=Purple, 3=Crimson) -->
      <div class="subject-code" style="color:{diff_color}; border-color:{diff_color}; letter-spacing:1.2px;">MOCK</div>
    </div>

    <div class="cover-title">
      <h1>วิชา คณิตศาสตร์ประยุกต์ 1 (Math 1)</h1>
      <p><strong>จำนวนข้อสอบ {len(items)} ข้อ</strong> &nbsp;|&nbsp; <strong>คะแนนเต็ม {total_pts} คะแนน</strong> &nbsp;|&nbsp; <strong>เวลาสอบ {meta['duration_min']} นาที</strong></p>
    </div>

    <!-- Student info box: keep Name-Surname field only -->
    <div class="student-box">
      <div class="student-row">
        <div class="field-line"><strong>ชื่อ – นามสกุล:</strong></div>
      </div>
    </div>

    <!-- Exam instructions (2B pencil bubble instruction removed) -->
    <div class="instructions">
      <h3>คำชี้แจง</h3>
      <ol>
        <li>แบบทดสอบฉบับนี้เป็นข้อสอบวิชา <strong>คณิตศาสตร์ประยุกต์ 1 (รหัสวิชา 61)</strong> มีจำนวนทั้งหมด <strong>{len(items)} ข้อ</strong> คะแนนเต็ม <strong>{total_pts} คะแนน</strong> แบ่งออกเป็น 2 ตอน ดังนี้:
          <ul>
            <li><strong>ตอนที่ 1:</strong> แบบปรนัย 5 ตัวเลือก (เลือก 1 คำตอบที่ถูกต้องที่สุด) จำนวน {len(mcq_items)} ข้อ (ข้อละ 3 คะแนน รวม {len(mcq_items)*3} คะแนน)</li>
            <li><strong>ตอนที่ 2:</strong> แบบระบายคำตอบที่เป็นตัวเลข จำนวน {len(num_items)} ข้อ (ข้อละ 5 คะแนน รวม {len(num_items)*5} คะแนน)</li>
          </ul>
        </li>
        <li>แต่ละข้อระบุระดับความยาก <strong>1–5 ดาว (★☆☆☆☆ ถึง ★★★★★)</strong> ตามความซับซ้อนของโจทย์และทักษะที่วัด</li>
        <li>พื้นที่ว่างในเล่มแบบทดสอบสามารถใช้เป็นกระดาษทดได้ และรูปประกอบในข้อสอบอาจไม่ได้เขียนตามมาตราส่วนจริง</li>
        <li>ทุกข้อในชุดนี้ผ่านการตรวจสอบความถูกต้องเชิงสัญลักษณ์ด้วยระบบ <strong>SymPy Verification Gate</strong></li>
      </ol>
    </div>
  </div>
  <div class="cover-footer">
    ห้ามเปิดแบบทดสอบจนกว่าจะได้รับอนุญาต &nbsp;•&nbsp; <span style="color:{diff_color};">MOCK</span> คณิตศาสตร์ประยุกต์ 1
  </div>
</div>
""")

    # Running header on question pages: replace Academic Year with difficulty-colored MOCK
    parts.append(
        f'<div class="running-header">'
        f'<span>คณิตศาสตร์ประยุกต์ 1 ({html.escape(meta["badge"])}) • {html.escape(gen_time)}</span>'
        f'<span style="color:{diff_color}; font-weight:700; letter-spacing:0.8px;">MOCK</span>'
        f'</div>'
    )

    if mcq_items:
        parts.append(
            f'<div class="part-banner">ตอนที่ 1 : แบบปรนัย 5 ตัวเลือก (จำนวน {len(mcq_items)} ข้อ ข้อละ 3 คะแนน)</div>'
        )
        for q in mcq_items:
            choices_html = "".join(
                f'<li><strong>{idx}.</strong> &nbsp;{_escape_keep_latex(str(ch), is_choice=True)}</li>'
                for idx, ch in enumerate(q.get("choices_latex", []), start=1)
            )
            parts.append(f"""
<div class="question-block">
  <div class="q-meta">
    <span class="q-num">ข้อที่ {q['slot_number']}. {_stars_html(q.get('difficulty_stars', 3))}</span>
    <span class="q-tags">{html.escape(q.get('chapter_th', q['chapter_id']))}</span>
  </div>
  <div class="q-stem">{_escape_keep_latex(q['stem_latex'])}</div>
  <ol class="choices">{choices_html}</ol>
</div>
""")

    if num_items:
        parts.append(
            f'<div class="part-banner">ตอนที่ 2 : แบบระบายคำตอบที่เป็นตัวเลข (จำนวน {len(num_items)} ข้อ ข้อละ 5 คะแนน)</div>'
        )
        for q in num_items:
            parts.append(f"""
<div class="question-block">
  <div class="q-meta">
    <span class="q-num">ข้อที่ {q['slot_number']}. {_stars_html(q.get('difficulty_stars', 3))}</span>
    <span class="q-tags">{html.escape(q.get('chapter_th', q['chapter_id']))}</span>
  </div>
  <div class="q-stem">{_escape_keep_latex(q['stem_latex'])}</div>
  <div class="numeric-box">
    <span>คำตอบ:</span>
    <span class="digit-cell"></span><span class="digit-cell"></span><span class="digit-cell"></span><span class="digit-cell"></span>
    <strong>.</strong>
    <span class="digit-cell"></span><span class="digit-cell"></span>
  </div>
</div>
""")

    parts.append("</body></html>")
    return "".join(parts)


def render_solutions_html(
    items: list[dict],
    mode: str = "mock_test",
    target_year_be: int = 2570,
    difficulty_level: int = 1,
    generated_at: str | None = None,
) -> str:
    """Build print-ready A4 HTML for Solutions & Answer Key matching the Exam Paper MOCK format."""
    meta = MODE_METADATA.get(mode, MODE_METADATA["mock_test"])
    diff_color = _difficulty_color(difficulty_level)
    gen_time = generated_at or _format_generated_timestamp()
    parts = [_base_head(f"เฉลยละเอียด MOCK ({meta['badge']})")]

    # Solutions header adapted to match Exam Paper: generation timestamp + difficulty-colored MOCK badge
    parts.append(
        f'<div class="running-header">'
        f'<span>เฉลยละเอียดและจุดหลอก • คณิตศาสตร์ประยุกต์ 1 ({html.escape(meta["badge"])}) • {html.escape(gen_time)}</span>'
        f'<span style="color:{diff_color}; font-weight:700; letter-spacing:0.8px;">MOCK</span>'
        f'</div>'
    )

    # Summary Answer Key Table
    rows = []
    for q in items:
        ans_display = (
            f"ข้อ {q['answer']}"
            if q["question_format"] == "mcq_5"
            else f"{q['answer']}"
        )
        rows.append(
            f"<tr>"
            f"<td><strong>{q['slot_number']}</strong></td>"
            f"<td>{html.escape(q.get('chapter_th', q['chapter_id']))}</td>"
            f"<td>{_stars_html(q.get('difficulty_stars', 3))}</td>"
            f"<td><strong>{html.escape(ans_display)}</strong></td>"
            f"<td>{html.escape(str(q.get('sympy_computed_answer', '-')))}</td>"
            f"</tr>"
        )

    parts.append(f"""
<h2 style="margin:6px 0 8px; font-size:16pt;">ตารางสรุปเฉลย (SymPy-Verified)</h2>
<table class="summary-table">
  <thead>
    <tr>
      <th>ข้อที่</th>
      <th>บทเรียน</th>
      <th>ระดับความยาก</th>
      <th>เฉลย</th>
      <th>ค่าตรวจ SymPy</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>
<h2 style="margin:14px 0 10px; font-size:16pt;">เฉลยละเอียดทีละข้อ</h2>
""")

    for q in items:
        ans_label = (
            f"ตัวเลือกที่ {q['answer']}"
            if q["question_format"] == "mcq_5"
            else f"{q['answer']}"
        )
        parts.append(f"""
<div class="question-block">
  <div class="q-meta">
    <span class="q-num">ข้อที่ {q['slot_number']} — เฉลย: {html.escape(ans_label)} {_stars_html(q.get('difficulty_stars', 3))}</span>
    <span class="q-tags">{html.escape(q.get('chapter_th', q['chapter_id']))} ({html.escape(q['sub_skill_id'])})</span>
  </div>
  <div class="sol-box">
    <div><strong>วิธีทำ:</strong> {_escape_keep_latex(q.get('solution_steps_th', ''))}</div>
    <div class="trap-note"><strong>จุดหลอกที่ต้องระวัง:</strong> {_escape_keep_latex(q.get('key_trap', '-'))}</div>
  </div>
</div>
""")

    parts.append("</body></html>")
    return "".join(parts)


def html_to_pdf(html_content: str, output_pdf: Path, timeout_seconds: float = 75.0) -> Path:
    """Render an HTML document with KaTeX + Sarabun into an A4 PDF via headless Edge/Chrome."""
    browser = find_headless_browser()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "doc.html"
            profile_dir = Path(tmpdir) / "browser_profile"
            html_path.write_text(html_content, encoding="utf-8")
            cmd = [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={profile_dir.resolve()}",
                "--no-first-run",
                "--no-default-browser-check",
                "--no-pdf-header-footer",
                "--virtual-time-budget=2500",
                f"--print-to-pdf={output_pdf.resolve()}",
                html_path.resolve().as_uri(),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
                if proc.returncode == 0 and output_pdf.exists() and output_pdf.stat().st_size > 0:
                    return output_pdf
                last_err = RuntimeError(
                    f"Headless PDF generation failed (rc={proc.returncode}): {proc.stderr[-300:]}"
                )
            except subprocess.TimeoutExpired as exc:
                last_err = exc
    raise RuntimeError(f"Headless PDF generation failed after retry: {last_err}")


def render_exam_bundle(
    items: list[dict],
    mode: str,
    target_year_be: int,
    out_dir: Path,
    difficulty_level: int = 1,
) -> dict[str, Path]:
    """Generate both Exam Paper PDF and Answer Key PDF (plus HTML previews) in out_dir using a shared timestamp."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_time = _format_generated_timestamp()
    exam_html = render_exam_html(
        items,
        mode=mode,
        target_year_be=target_year_be,
        difficulty_level=difficulty_level,
        generated_at=gen_time,
    )
    sol_html = render_solutions_html(
        items,
        mode=mode,
        target_year_be=target_year_be,
        difficulty_level=difficulty_level,
        generated_at=gen_time,
    )

    exam_html_path = out_dir / "exam.html"
    sol_html_path = out_dir / "solutions.html"
    exam_pdf_path = out_dir / "exam.pdf"
    sol_pdf_path = out_dir / "solutions.pdf"

    exam_html_path.write_text(exam_html, encoding="utf-8")
    sol_html_path.write_text(sol_html, encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_exam = pool.submit(html_to_pdf, exam_html, exam_pdf_path)
        f_sol = pool.submit(html_to_pdf, sol_html, sol_pdf_path)
        f_exam.result()
        f_sol.result()

    return {
        "exam_html": exam_html_path,
        "solutions_html": sol_html_path,
        "exam_pdf": exam_pdf_path,
        "solutions_pdf": sol_pdf_path,
    }
