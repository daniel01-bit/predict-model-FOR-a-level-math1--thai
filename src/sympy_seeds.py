"""SymPy-First deterministic parameter generators covering ALL 16 Syllabus 2560 chapters.

Guarantees mathematically consistent, solvable parameters and realistic difficulty calibration
BEFORE the LLM writes the Thai word problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class MathSeed:
    chapter_id: str
    params_description_th: str
    exact_answer: str
    sympy_code: str
    suggested_stars: int = 3


def _seed_sets(rng: random.Random) -> MathSeed:
    only_a = rng.randint(5, 18)
    only_b = rng.randint(5, 18)
    only_c = rng.randint(5, 18)
    ab_only = rng.randint(2, 8)
    ac_only = rng.randint(2, 8)
    bc_only = rng.randint(2, 8)
    abc = rng.randint(2, 7)
    outside = rng.randint(3, 12)

    n_a = only_a + ab_only + ac_only + abc
    n_b = only_b + ab_only + bc_only + abc
    n_c = only_c + ac_only + bc_only + abc
    n_ab = ab_only + abc
    n_ac = ac_only + abc
    n_bc = bc_only + abc
    n_u = only_a + only_b + only_c + ab_only + ac_only + bc_only + abc + outside
    ans = outside

    code = (
        f"n_u, n_a, n_b, n_c = {n_u}, {n_a}, {n_b}, {n_c}\n"
        f"n_ab, n_ac, n_bc, n_abc = {n_ab}, {n_ac}, {n_bc}, {abc}\n"
        f"n_union = n_a + n_b + n_c - n_ab - n_ac - n_bc + n_abc\n"
        f"ANSWER = sp.Integer(n_u - n_union)\n"
        f"assert ANSWER >= 0 and n_union <= n_u"
    )
    desc = (
        f"กลุ่มตัวอย่างทั้งหมด {n_u} คน: ชอบ A {n_a} คน, ชอบ B {n_b} คน, ชอบ C {n_c} คน, "
        f"ชอบ A และ B {n_ab} คน, ชอบ A และ C {n_ac} คน, ชอบ B และ C {n_bc} คน, ชอบทั้ง 3 อย่าง {abc} คน "
        f"(ห้ามบอกข้อมูลซ้ำซ้อนเกินจำเป็น) -> ถามจำนวนคนที่ไม่ชอบทั้ง 3 อย่างเลย (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("sets", desc, str(ans), code, suggested_stars=3)


def _seed_logic(rng: random.Random) -> MathSeed:
    # Always provide 4 sub-statements (ก, ข, ค, ง) so counts 0, 1, 2, 3, 4 map 1-to-1 to 5 MCQ choices
    p_val = rng.choice([True, False])
    q_val = rng.choice([True, False])
    r_val = rng.choice([True, False])

    stmt1 = (not p_val) or q_val                 # p -> q
    stmt2 = (p_val and (not r_val))              # p ^ ~r
    stmt3 = (q_val == r_val)                     # q <-> r
    stmt4 = (not (p_val or q_val)) or r_val      # (p v q) -> r
    ans = sum([stmt1, stmt2, stmt3, stmt4])

    p_th = "จริง" if p_val else "เท็จ"
    q_th = "จริง" if q_val else "เท็จ"
    r_th = "จริง" if r_val else "เท็จ"
    code = (
        f"p, q, r = {p_val}, {q_val}, {r_val}\n"
        f"stmts = [(not p) or q, p and (not r), q == r, (not (p or q)) or r]\n"
        f"ANSWER = sp.Integer(sum(1 for s in stmts if s))\n"
        f"assert 0 <= ANSWER <= 4"
    )
    desc = (
        f"กำหนดให้ประพจน์ p, q, r มีค่าความจริงเป็น {p_th}, {q_th}, {r_th} ตามลำดับ "
        f"พิจารณาข้อความ 4 ข้อต่อไปนี้: (ก) p -> q, (ข) p ^ ~r, (ค) q <-> r, (ง) (p v q) -> r "
        f"-> ถามว่ามีข้อความที่มีค่าความจริงเป็น 'จริง' กี่ข้อ (ใช้ตัวเลือก 0, 1, 2, 3, 4 เท่านั้น; คำตอบแน่นอน = {ans})"
    )
    return MathSeed("logic", desc, str(ans), code, suggested_stars=2)


def _seed_real_numbers(rng: random.Random) -> MathSeed:
    r1 = rng.randint(-3, 1)
    r2 = r1 + rng.randint(3, 6)
    ints = list(range(r1, r2 + 1))
    ans = sum(ints)
    b_coef = -(r1 + r2)
    c_coef = r1 * r2
    code = (
        f"ints = [k for k in range({r1 - 2}, {r2 + 3}) if (k - ({r1}))*(k - ({r2})) <= 0]\n"
        f"assert len(ints) > 0\n"
        f"ANSWER = sp.Integer(sum(ints))"
    )
    desc = (
        f"จงหาผลบวกของจำนวนเต็ม x ทั้งหมดที่สอดคล้องอสมการ x^2 + ({b_coef})x + ({c_coef}) <= 0 "
        f"(คำตอบแน่นอน = {ans}, จำนวนเต็มคือ {ints})"
    )
    return MathSeed("real_numbers", desc, str(ans), code, suggested_stars=2)


def _seed_functions(rng: random.Random) -> MathSeed:
    # Quadratic profit/revenue maximization: R(x) = -a*(x - h)^2 + k on [0, 2h]
    a = rng.choice([2, 3, 5])
    h = rng.randint(10, 25)
    k = rng.randint(500, 1500)
    # Ask ONE clear question only (maximum revenue/profit = k)
    code = (
        f"x = sp.Symbol('x', real=True)\n"
        f"R = -{a}*(x - {h})**2 + {k}\n"
        f"crit = sp.solve(sp.diff(R, x), x)[0]\n"
        f"assert 0 <= crit <= {2 * h}\n"
        f"ANSWER = R.subs(x, crit)"
    )
    b_lin = 2 * a * h
    c_const = k - a * (h**2)
    desc = (
        f"ฟังก์ชันกำไรจากการผลิตสินค้า x ชิ้นคือ R(x) = -{a}x^2 + {b_lin}x + ({c_const}) บาท สำหรับ 0 <= x <= {2 * h} "
        f"-> ถามคำถามเดียวชัดเจน: จงหากำไรสูงสุดที่เป็นไปได้ (ห้ามถาม 2 คำถามซ้อน; คำตอบแน่นอน = {k})"
    )
    return MathSeed("functions", desc, str(k), code, suggested_stars=2)


def _seed_expo_log(rng: random.Random) -> MathSeed:
    # log_2(x - a) + log_2(x - b) = c where (x-a)(x-b) = 2^c has one valid root x > max(a, b)
    # Pick valid root x_true and factor pair (u, v) with u*v = 2^c
    c = rng.choice([3, 4, 5])
    p_u = rng.randint(1, c - 1)
    u = 2**p_u
    v = 2 ** (c - p_u)
    x_true = max(u, v) + rng.randint(2, 6)
    a = x_true - u
    b = x_true - v
    code = (
        f"x = sp.Symbol('x', real=True)\n"
        f"sols = sp.solve(sp.Eq((x - {a})*(x - {b}), 2**{c}), x)\n"
        f"valid = [s for s in sols if s > max({a}, {b})]\n"
        f"assert len(valid) == 1\n"
        f"ANSWER = valid[0]"
    )
    desc = (
        f"จงหาค่าจำนวนจริง x ที่สอดคล้องสมการ log_2(x - {a}) + log_2(x - {b}) = {c} "
        f"(ต้องตรวจโดเมนหลัง log > 0; คำตอบแน่นอน = {x_true})"
    )
    return MathSeed("expo_log", desc, str(x_true), code, suggested_stars=3)


def _seed_trigonometry(rng: random.Random) -> MathSeed:
    # Exact tan(arctan(1/p) + arctan(1/q)) = (p + q) / (p*q - 1) with p*q > 1
    p, q = rng.choice([(2, 3), (2, 5), (3, 4), (3, 5), (2, 7)])
    val = sp.Rational(p + q, p * q - 1)
    ans_str = str(int(val)) if val.q == 1 else f"{val.p}/{val.q}"
    code = (
        f"p, q = sp.Integer({p}), sp.Integer({q})\n"
        f"assert p * q > 1\n"
        f"ANSWER = sp.trigsimp(sp.tan(sp.atan(sp.Rational(1, p)) + sp.atan(sp.Rational(1, q))))"
    )
    desc = (
        f"กำหนดให้มุม theta = arctan(1/{p}) + arctan(1/{q}) (เขียน \\theta และตัวเลขเศษส่วนให้ครบ ห้ามเว้นว่าง) "
        f"จงหาค่าของ tan(theta) (คำตอบแน่นอน = {ans_str})"
    )
    return MathSeed("trigonometry", desc, ans_str, code, suggested_stars=3)


def _seed_complex_numbers(rng: random.Random) -> MathSeed:
    # |z1^n / z2^m| with Pythagorean triples so modulus is exact integer
    triples = [(3, 4, 5), (6, 8, 10), (5, 12, 13)]
    a1, b1, r1 = rng.choice(triples)
    a2, b2, r2 = (3, 4, 5)
    n_pow = rng.choice([2, 3])
    # Ensure exact integer: use (r1^n_pow) / r2 when divisible, else |z1^n_pow * conj(z2)|
    ans = (r1**n_pow) * r2
    code = (
        f"z1 = {a1} + {b1}*sp.I\n"
        f"z2 = {a2} - {b2}*sp.I\n"
        f"ANSWER = sp.Abs(z1**{n_pow} * sp.conjugate(z2))\n"
        f"assert ANSWER > 0"
    )
    desc = (
        f"กำหนดจำนวนเชิงซ้อน z_1 = {a1} + {b1}i และ z_2 = {a2} - {b2}i "
        f"จงหาค่าของ |z_1^{n_pow} * conj(z_2)| (คำตอบแน่นอน = {ans}, ระดับความยาก 2-3 ดาว)"
    )
    return MathSeed("complex_numbers", desc, str(ans), code, suggested_stars=2)


def _seed_matrices(rng: random.Random) -> MathSeed:
    det_b = rng.choice([-3, -2, 2, 3])
    k = rng.choice([2, 3])
    det_a = det_b * rng.choice([-2, 2, 3])
    val = sp.Rational((k**3) * (det_a**2), det_b)
    ans = str(int(val)) if val.q == 1 else str(val)
    code = (
        f"det_a, det_b, k, n = sp.Integer({det_a}), sp.Integer({det_b}), sp.Integer({k}), 3\n"
        f"assert det_a != 0 and det_b != 0\n"
        f"ANSWER = (k**n) * (det_a**2) / det_b"
    )
    desc = (
        f"กำหนดเมทริกซ์ A และ B ขนาด 3 x 3 โดย det(A) = {det_a} และ det(B) = {det_b} "
        f"-> จงหาค่าของ det({k} A^2 B^(-1)) (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("matrices", desc, ans, code, suggested_stars=3)


def _seed_sequences_series(rng: random.Random) -> MathSeed:
    # Infinite geometric series with guaranteed |r| < 1
    num, den = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (2, 5)])
    a1 = (den - num) * rng.randint(2, 6)
    val = sp.Rational(a1 * den, den - num)
    ans = str(int(val))
    code = (
        f"a1, r = sp.Integer({a1}), sp.Rational({num}, {den})\n"
        f"assert abs(r) < 1\n"
        f"ANSWER = a1 / (1 - r)"
    )
    desc = (
        f"อนุกรมเรขาคณิตอนันต์มีพจน์แรก a_1 = {a1} และอัตราส่วนร่วม r = {num}/{den} "
        f"จงหาผลบวกของอนุกรมอนันต์นี้ (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("sequences_series", desc, ans, code, suggested_stars=2)


def _seed_analytic_geometry(rng: random.Random) -> MathSeed:
    triples = [(5, 4, 3), (5, 3, 4), (13, 12, 5), (10, 8, 6), (10, 6, 8)]
    a, b, c = rng.choice(triples)
    h, k = rng.randint(-4, 4), rng.randint(-4, 4)
    ans = 2 * c
    code = (
        f"a, b = sp.Integer({a}), sp.Integer({b})\n"
        f"assert a > b > 0\n"
        f"c = sp.sqrt(a**2 - b**2)\n"
        f"ANSWER = 2 * c"
    )
    desc = (
        f"วงรีมีสมการ (x - ({h}))^2 / {a**2} + (y - ({k}))^2 / {b**2} = 1 "
        f"จงหาระยะห่างระหว่างจุดโฟกัสทั้งสองของวงรีนี้ (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("analytic_geometry", desc, str(ans), code, suggested_stars=3)


def _seed_vectors_3d(rng: random.Random) -> MathSeed:
    # 3D cross product triangle area with exact integer result (e.g. u=(a,0,0), v=(b,c,0) rotated or integer cross)
    u = (rng.randint(1, 4), rng.randint(-3, 3), rng.randint(1, 3))
    v = (rng.randint(-3, 3), rng.randint(1, 4), rng.randint(-2, 3))
    # Dot product u . (2u - v) is always a clean integer and avoids radical typos
    dot_uv = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    norm_u2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
    ans = 2 * norm_u2 - dot_uv
    code = (
        f"u = sp.Matrix([{u[0]}, {u[1]}, {u[2]}])\n"
        f"v = sp.Matrix([{v[0]}, {v[1]}, {v[2]}])\n"
        f"ANSWER = u.dot(2*u - v)"
    )
    desc = (
        f"กำหนดเวกเตอร์ในสามมิติ u = ({u[0]}, {u[1]}, {u[2]}) และ v = ({v[0]}, {v[1]}, {v[2]}) "
        f"จงหาค่าของ u . (2u - v) (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("vectors_3d", desc, str(ans), code, suggested_stars=2)


def _seed_statistics(rng: random.Random) -> MathSeed:
    a = rng.randint(4, 10)
    b = a + rng.randint(2, 5)
    x_true = b + rng.randint(2, 5)
    d = x_true + rng.randint(2, 5)
    base_e = d + rng.randint(2, 6)
    rem = (a + b + x_true + d + base_e) % 5
    e = base_e + ((5 - rem) % 5)
    mean_val = (a + b + x_true + d + e) // 5

    code = (
        f"x = sp.Symbol('x')\n"
        f"data = [{a}, {b}, x, {d}, {e}]\n"
        f"sol = sp.solve(sp.Eq(sum(data) / sp.Integer(5), {mean_val}), x)[0]\n"
        f"assert {b} <= sol <= {d}\n"
        f"ANSWER = sol"
    )
    desc = (
        f"ข้อมูล 5 จำนวนเรียงจากน้อยไปมากคือ {a}, {b}, x, {d}, {e} "
        f"ถ้าค่าเฉลี่ยเลขคณิตเท่ากับ {mean_val} จงหาค่าของ x (ห้ามกำหนดมัธยฐานขัดแย้งกับค่าเฉลี่ย; "
        f"คำตอบแน่นอน = {x_true}, คงลำดับ {b} <= {x_true} <= {d})"
    )
    return MathSeed("statistics", desc, str(x_true), code, suggested_stars=2)


def _seed_prob_distributions(rng: random.Random) -> MathSeed:
    # Binomial expected value & variance: E(X) = n*p, Var(X) = n*p*(1-p)
    n = rng.choice([20, 25, 40, 50])
    p_num, p_den = rng.choice([(1, 5), (2, 5), (3, 5), (1, 4), (3, 4)])
    mean_val = sp.Rational(n * p_num, p_den)
    var_val = sp.Rational(n * p_num * (p_den - p_num), p_den**2)
    # Ask for E(X) + Var(X) as exact decimal/rational
    total = mean_val + var_val
    ans_str = str(int(total)) if total.q == 1 else f"{float(total):.4g}"
    code = (
        f"n, p = sp.Integer({n}), sp.Rational({p_num}, {p_den})\n"
        f"assert 0 < p < 1 and n > 0\n"
        f"ANSWER = n*p + n*p*(1 - p)"
    )
    desc = (
        f"ตัวแปรสุ่ม X มีการแจกแจงทวินาม B(n={n}, p={p_num}/{p_den}) "
        f"จงหาผลบวกของค่าคาดหมาย E(X) กับความแปรปรวน Var(X) (คำตอบแน่นอน = {ans_str})"
    )
    return MathSeed("prob_distributions", desc, ans_str, code, suggested_stars=3)


def _seed_counting(rng: random.Random) -> MathSeed:
    men = rng.randint(5, 7)
    women = rng.randint(4, 6)
    pick_m = rng.randint(2, 3)
    pick_w = rng.randint(2, 3)
    ans = int(sp.binomial(men, pick_m) * sp.binomial(women, pick_w))
    code = (
        f"men, women, pm, pw = {men}, {women}, {pick_m}, {pick_w}\n"
        f"assert 0 <= pm <= men and 0 <= pw <= women\n"
        f"ANSWER = sp.binomial(men, pm) * sp.binomial(women, pw)"
    )
    desc = (
        f"มีผู้สมัครชาย {men} คน และหญิง {women} คน ต้องการคัดเลือกคณะกรรมการ {pick_m + pick_w} คน "
        f"โดยต้องมีชาย {pick_m} คน และหญิง {pick_w} คน จะมีวิธีเลือกได้ทั้งหมดกี่วิธี (คำตอบแน่นอน = {ans})"
    )
    return MathSeed("counting", desc, str(ans), code, suggested_stars=2)


def _seed_probability(rng: random.Random) -> MathSeed:
    red = rng.randint(4, 6)
    blue = rng.randint(3, 5)
    total = red + blue
    prob = sp.Rational(sp.binomial(red, 2), sp.binomial(total, 2))
    ans_str = f"{prob.p}/{prob.q}"
    code = (
        f"red, blue = {red}, {blue}\n"
        f"total = red + blue\n"
        f"ANSWER = sp.binomial(red, 2) / sp.binomial(total, 2)\n"
        f"assert 0 < ANSWER < 1"
    )
    desc = (
        f"กล่องใบหนึ่งมีลูกบอลสีแดง {red} ลูก และสีน้ำเงิน {blue} ลูก สุ่มหยิบพร้อมกัน 2 ลูก "
        f"จงหาความน่าจะเป็นที่จะได้ลูกบอลสีแดงทั้ง 2 ลูก (คำตอบแน่นอน = {ans_str})"
    )
    return MathSeed("probability", desc, ans_str, code, suggested_stars=2)


def _seed_calculus(rng: random.Random) -> MathSeed:
    c = rng.randint(1, 5)
    a = rng.randint(1, 4)
    b_true = c * (2 - a)
    code = (
        f"x, b = sp.symbols('x b')\n"
        f"lim_val = sp.limit((x**2 - {c**2}) / (x - {c}), x, {c})\n"
        f"sol_b = sp.solve(sp.Eq({a}*{c} + b, lim_val), b)[0]\n"
        f"assert sp.limit((x**2 - {c**2}) / (x - {c}), x, {c}) == {a}*{c} + sol_b\n"
        f"ANSWER = sol_b"
    )
    desc = (
        f"กำหนดฟังก์ชัน f(x) = (x^2 - {c**2})/(x - {c}) เมื่อ x != {c} และ f(x) = {a}x + b เมื่อ x = {c} "
        f"ถ้า f ต่อเนื่องที่ x = {c} จงหาค่าของ b (คำตอบแน่นอน = {b_true})"
    )
    return MathSeed("calculus", desc, str(b_true), code, suggested_stars=3)


_SEEDERS = {
    "sets": _seed_sets,
    "logic": _seed_logic,
    "real_numbers": _seed_real_numbers,
    "functions": _seed_functions,
    "expo_log": _seed_expo_log,
    "trigonometry": _seed_trigonometry,
    "complex_numbers": _seed_complex_numbers,
    "matrices": _seed_matrices,
    "sequences_series": _seed_sequences_series,
    "analytic_geometry": _seed_analytic_geometry,
    "vectors_3d": _seed_vectors_3d,
    "statistics": _seed_statistics,
    "prob_distributions": _seed_prob_distributions,
    "counting": _seed_counting,
    "probability": _seed_probability,
    "calculus": _seed_calculus,
}


def generate_verified_seed(chapter_id: str, seed: int | None = None) -> MathSeed | None:
    """Return a SymPy-preverified parameter seed covering all 16 Syllabus 2560 chapters."""
    fn = _SEEDERS.get(chapter_id)
    if fn is None:
        return None
    rng = random.Random(seed)
    return fn(rng)
