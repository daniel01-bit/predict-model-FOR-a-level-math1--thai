"""Dynamic Archetype, Real-World Context, and Modular Few-Shot Diversity Engine.

Eliminates repetitive exam templates across consecutive generations by injecting:
1. Non-repeating (Subtopic_Archetype, RealWorld_Context, Algebraic_Structure) per slot
2. Curated 3-4 star IPST A-Level few-shot exemplars from data/few_shot_examples/<chapter_id>.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEW_SHOT_DIR = PROJECT_ROOT / "data" / "few_shot_examples"

REAL_WORLD_CONTEXTS: tuple[str, ...] = (
    "การวางแผนการผลิตและต้นทุนของโรงงานอัจฉริยะ",
    "การวิเคราะห์ข้อมูลผู้ใช้แพลตฟอร์มสตรีมมิง/อีคอมเมิร์ซ",
    "การออกแบบโครงสร้างสะพานหรือหลังคาโค้งทางวิศวกรรม",
    "การติดตามการเติบโตของประชากรสัตว์/เชื้อจุลินทรีย์ในห้องแล็บ",
    "การออมเงินระยะยาวและผ่อนชำระรายงวดของนักลงทุน",
    "การเคลื่อนที่ของโดรนสำรวจพื้นที่ในพิกัดสามมิติ",
    "การสุ่มตรวจคุณภาพสินค้าและสถิติคะแนนสอบระดับประเทศ",
    "การวัดมุมและระยะทางในงานสำรวจภูมิประเทศ",
)

CHAPTER_ARCHETYPES: dict[str, list[tuple[str, str]]] = {
    "functions": [
        ("func_closed_range", "หาผลต่าง/ผลบวกของค่าสูงสุดและต่ำสุดสัมบูรณ์บนช่วงปิดที่มีจุดยอดอยู่ภายใน"),
        ("func_piecewise_cont", "ฟังก์ชันแบ่งช่วงเชื่อมต่อกันอย่างต่อเนื่องที่จุดรอยต่อ + หาค่าพารามิเตอร์"),
        ("func_composite_inv", "ฟังก์ชันประกอบ f(g(x)) ผสมอินเวอร์ส f^{-1} พร้อมเงื่อนไขโดเมน"),
    ],
    "sequences_series": [
        ("seq_mixed_arith_geom", "สมการผสมลำดับเลขคณิตและเรขาคณิต a_n + b_n = K หาจำนวนเต็มบวก n"),
        ("seq_shifted_inf_series", "อนุกรมเรขาคณิตอนันต์แบบเลื่อนดัชนี (n=1 กับ n=0) หรือซิกมาผสม"),
        ("seq_financial_annuity", "คณิตศาสตร์การเงิน ดอกเบี้ยทบต้น/เงินงวดต้นงวด-ปลายงวด"),
    ],
    "analytic_geometry": [
        ("conic_ellipse_focal_tri", "วงรี (Pythagorean triple) หาความยาวรอบรูปสามเหลี่ยมโฟกัส PF1F2"),
        ("conic_hyperbola_axes", "ไฮเพอร์โบลา (Pythagorean triple) ผสมระยะโฟกัส-แกนตามขวาง (ห้ามจุดบนเส้นกำกับ)"),
        ("conic_parabola_circle", "พาราโบลาเชื่อมต่อจุดศูนย์กลางวงกลมและระยะห่างจากจุดถึงเส้นตรง"),
    ],
    "vectors_3d": [
        ("vec_parallelepiped_vol", "ปริมาตรทรงสี่เหลี่ยมด้านขนาน |u . (v x w)| จากพิกัดจำนวนเต็ม"),
        ("vec_cross_area_angle", "พื้นที่สี่เหลี่ยม/สามเหลี่ยมจาก |u x v| เชื่อมโยงมุมระหว่างเวกเตอร์"),
        ("vec_perp_projection", "เงื่อนไขเวกเตอร์ตั้งฉาก (u . v = 0) หาพารามิเตอร์แล้วคำนวณขนาดเวกเตอร์ผลรวม"),
    ],
    "expo_log": [
        ("log_domain_quad", "สมการลอการิทึมที่แปลงเป็นสมการกำลังสองและต้องตัดรากนอกโดเมนทิ้ง"),
        ("exp_substitution", "สมการเอกซ์โพเนนเชียลแบบแทนตัวแปร u = a^x พร้อมผลบวกราก"),
    ],
    "calculus": [
        ("calc_extrema_opt", "โจทย์ประยุกต์ค่าสูงสุด/ต่ำสุดสัมบูรณ์ (Optimization) บนช่วงปิด"),
        ("calc_tangent_area", "เส้นสัมผัสเส้นโค้งเชื่อมโยงพื้นที่ปิดล้อมด้วยอินทิกรัลจำกัดเขต"),
    ],
}

DEFAULT_ARCHETYPES: list[tuple[str, str]] = [
    ("multi_step_param", "โจทย์ 2 ขั้นตอน: แก้เงื่อนไขแรกหาพารามิเตอร์ แล้วนำไปคำนวณเป้าหมายขั้นที่สอง"),
    ("applied_modeling", "โจทย์สถานการณ์ประยุกต์สไตล์ สสวท. ที่ต้องสร้างแบบจำลองสมการก่อนคำนวณ"),
    ("domain_trap_filter", "โจทย์ที่ต้องตรวจสอบเงื่อนไขโดเมน/ความสมเหตุสมผลเพื่อคัดรากที่ใช้งานได้จริง"),
]


def sample_diversity_directive(
    chapter_id: str,
    used_archetypes: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Return a non-repeating (archetype_id, Thai diversity directive) for this exam slot."""
    r = rng or random.Random()
    pool = CHAPTER_ARCHETYPES.get(chapter_id, DEFAULT_ARCHETYPES)
    used = used_archetypes if used_archetypes is not None else set()
    fresh = [item for item in pool if item[0] not in used]
    arch_id, arch_desc = r.choice(fresh or pool)
    used.add(arch_id)
    context = r.choice(REAL_WORLD_CONTEXTS)
    directive = f"แนวโครงสร้างโจทย์: {arch_desc} | บริบทแนะนำ: {context}"
    return arch_id, directive


def load_few_shot_examples(
    chapter_id: str,
    k: int = 2,
    rng: random.Random | None = None,
) -> list[dict]:
    """Load up to k random curated 3-4 star few-shot examples from data/few_shot_examples/<chapter_id>.json."""
    path = FEW_SHOT_DIR / f"{chapter_id}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            return []
        r = rng or random.Random()
        return r.sample(data, min(k, len(data)))
    except Exception:
        return []


def format_few_shot_block(chapter_id: str, k: int = 1, rng: random.Random | None = None) -> str:
    """Format curated few-shot examples as concise style/difficulty anchors for the LLM prompt."""
    examples = load_few_shot_examples(chapter_id, k=k, rng=rng)
    if not examples:
        return ""
    lines = ["\n- **ตัวอย่างแนวข้อสอบระดับ 3–4 ดาว สสวท. (ใช้เป็นแรงบันดาลใจด้านสำนวนและระดับความยาก ห้ามลอกตัวเลขเดิม):**"]
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"  ตัวอย่าง {idx} ({ex.get('stars', 3)}★ — {ex.get('archetype', 'multi-step')}): {ex.get('stem_summary', '')}")
    return "\n".join(lines) + "\n"
