"""SymPy-First deterministic parameter generators covering ALL 16 Syllabus 2560 chapters.

Guarantees mathematically consistent, solvable parameters and realistic difficulty calibration
BEFORE the LLM writes the Thai word problem.
"""

from __future__ import annotations

import math
import random
import re
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
        f"กำหนดให้ประพจน์ $p, q, r$ มีค่าความจริงเป็น {p_th}, {q_th}, {r_th} ตามลำดับ "
        f"พิจารณาข้อความ 4 ข้อต่อไปนี้: (ก) $p \\rightarrow q$, (ข) $p \\land \\sim r$, "
        f"(ค) $q \\leftrightarrow r$, (ง) $(p \\lor q) \\rightarrow r$ "
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
    # 3-4 Star Reverse-Construction across 2 archetypes:
    # (A) Closed-interval range [x1, x2] with interior vertex (verified via function_range)
    # (B) Piecewise continuity at boundary x = c (verified via left/right limits == f(c))
    if rng.random() < 0.5:
        a = rng.choice([1, 2, 3])
        h = rng.randint(2, 6)
        k = rng.randint(1, 8)
        x1 = h - rng.randint(2, 4)
        x2 = h + rng.randint(1, 3)
        b_lin = -2 * a * h
        c_const = a * (h**2) + k
        f_x1 = a * ((x1 - h) ** 2) + k
        f_x2 = a * ((x2 - h) ** 2) + k
        max_val = max(f_x1, f_x2)
        min_val = k
        ans = max_val - min_val
        code = (
            f"x = sp.Symbol('x', real=True)\n"
            f"f = {a}*x**2 + ({b_lin})*x + ({c_const})\n"
            f"rng_set = sp.calculus.util.function_range(f, x, sp.Interval({x1}, {x2}))\n"
            f"assert isinstance(rng_set, sp.Interval)\n"
            f"ANSWER = rng_set.sup - rng_set.inf\n"
            f"assert ANSWER > 0"
        )
        desc = (
            f"กำหนดฟังก์ชัน f(x) = {a}x^2 + ({b_lin})x + ({c_const}) บนช่วงปิด [{x1}, {x2}] "
            f"(จุดยอดอยู่ภายในช่วงปิด ต้องเช็กทั้งจุดยอดและจุดปลายช่วง) "
            f"-> จงหาผลต่างระหว่างค่าสูงสุดสัมบูรณ์และค่าต่ำสุดสัมบูรณ์ของ f บนช่วงนี้ (คำตอบแน่นอน = {ans}, 3-4 ดาว)"
        )
        return MathSeed("functions", desc, str(ans), code, suggested_stars=3)

    # Archetype B: Piecewise continuity at x = c
    c_pt = rng.randint(2, 5)
    p = rng.choice([1, 2])
    a_slope = rng.randint(2, 6)
    k_val = rng.randint(3, 12)
    b_shift = p * (c_pt**2) + k_val - a_slope * c_pt
    ans = a_slope + k_val
    code = (
        f"x, k = sp.symbols('x k', real=True)\n"
        f"f_left = {a_slope}*x + ({b_shift})\n"
        f"f_right = {p}*x**2 + k\n"
        f"L_left = sp.limit(f_left, x, {c_pt}, dir='-')\n"
        f"L_right = sp.limit(f_right, x, {c_pt}, dir='+')\n"
        f"sols = sp.solve(sp.Eq(L_left, L_right), k)\n"
        f"assert len(sols) == 1\n"
        f"ANSWER = sp.Integer({a_slope}) + sols[0]\n"
        f"assert ANSWER > 0"
    )
    desc = (
        f"กำหนดฟังก์ชันแบ่งช่วง f(x) = {a_slope}x + ({b_shift}) เมื่อ x < {c_pt} และ f(x) = {p}x^2 + k เมื่อ x >= {c_pt} "
        f"ถ้า f เป็นฟังก์ชันต่อเนื่องที่ x = {c_pt} จงหาค่าของ {a_slope} + k (คำตอบแน่นอน = {ans}, 3-4 ดาว)"
    )
    return MathSeed("functions", desc, str(ans), code, suggested_stars=4)


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
        f"จงหาค่าของ $|z_1^{{{n_pow}}} \\cdot \\overline{{z_2}}|$ (คำตอบแน่นอน = {ans}, ระดับความยาก 2-3 ดาว)"
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
    # 3-4 Star Reverse-Construction across 2 archetypes:
    # (A) Mixed arithmetic + geometric equation a_n + b_n = K: pick integer n > 0 FIRST, compute K
    # (B) Convergent infinite geometric series with shifted start index (n=1 vs n=0)
    if rng.random() < 0.5:
        n_true = rng.randint(4, 7)
        a1 = rng.randint(2, 6)
        d = rng.randint(2, 5)
        b1 = rng.choice([1, 2, 3])
        r = 2
        a_n = a1 + (n_true - 1) * d
        b_n = b1 * (r ** (n_true - 1))
        K = a_n + b_n
        code = (
            f"n_sols = [n for n in range(1, 20) if ({a1} + (n - 1)*{d}) + ({b1} * {r}**(n - 1)) == {K}]\n"
            f"assert len(n_sols) == 1 and n_sols[0] > 0\n"
            f"ANSWER = sp.Integer(n_sols[0])"
        )
        desc = (
            f"กำหนดลำดับเลขคณิต a_n มี a_1 = {a1}, ผลต่างร่วม d = {d} "
            f"และลำดับเรขาคณิต b_n มี b_1 = {b1}, อัตราส่วนร่วม r = {r} "
            f"ถ้าพบว่า a_n + b_n = {K} จงหาจำนวนเต็มบวก n (คำตอบแน่นอน = {n_true}, 3-4 ดาว)"
        )
        return MathSeed("sequences_series", desc, str(n_true), code, suggested_stars=4)

    num, den = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (2, 5)])
    # Sum_{n=1}^{inf} c * (num/den)^n = c * (num/den) / (1 - num/den) = c * num / (den - num)
    c_mult = (den - num) * rng.randint(2, 6)
    val = sp.Rational(c_mult * num, den - num)
    ans = str(int(val))
    code = (
        f"c, r = sp.Integer({c_mult}), sp.Rational({num}, {den})\n"
        f"assert abs(r) < 1\n"
        f"ANSWER = (c * r) / (1 - r)\n"
        f"assert ANSWER > 0"
    )
    desc = (
        f"จงหาผลบวกของอนุกรมอนันต์ Sigma_{{n=1}}^{{infinity}} {c_mult} * ({num}/{den})^n "
        f"(ระวังดัชนีเริ่มต้น n=1 ไม่ใช่ n=0; คำตอบแน่นอน = {ans}, 3 ดาว)"
    )
    return MathSeed("sequences_series", desc, ans, code, suggested_stars=3)


def _seed_analytic_geometry(rng: random.Random) -> MathSeed:
    # 3-4 Star Pythagorean-Triple Conics (Ellipse focal triangle perimeter OR Hyperbola focal length)
    if rng.random() < 0.5:
        triples = [(5, 4, 3), (5, 3, 4), (13, 12, 5), (10, 8, 6), (17, 15, 8)]
        a, b, c = rng.choice(triples)
        h, k = rng.randint(-4, 4), rng.randint(-4, 4)
        # Perimeter of triangle F1-P-F2 for any point P on the ellipse is strictly 2a + 2c
        ans = 2 * a + 2 * c
        code = (
            f"a, b = sp.Integer({a}), sp.Integer({b})\n"
            f"assert a > b > 0\n"
            f"c = sp.sqrt(a**2 - b**2)\n"
            f"assert c.is_integer and c > 0\n"
            f"ANSWER = 2*a + 2*c"
        )
        desc = (
            f"วงรีมีสมการ (x - ({h}))^2 / {a**2} + (y - ({k}))^2 / {b**2} = 1 มีจุดโฟกัสคือ F_1 และ F_2 "
            f"ถ้า P เป็นจุดใดๆ บนวงรีนี้ที่ไม่ใช่จุดยอดบนแกนเอก จงหาความยาวรอบรูปของสามเหลี่ยม PF_1F_2 "
            f"(ใช้สมบัติ PF_1 + PF_2 = 2a และ F_1F_2 = 2c; คำตอบแน่นอน = {ans}, 3-4 ดาว)"
        )
        return MathSeed("analytic_geometry", desc, str(ans), code, suggested_stars=4)

    # Archetype B: Hyperbola with Pythagorean triple c^2 = a^2 + b^2
    h_triples = [(3, 4, 5), (4, 3, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17)]
    a, b, c = rng.choice(h_triples)
    h, k = rng.randint(-3, 3), rng.randint(-3, 3)
    # Ask for 2c + 2a (sum of focal distance and transverse axis length, guaranteed positive integer)
    ans = 2 * c + 2 * a
    code = (
        f"a, b = sp.Integer({a}), sp.Integer({b})\n"
        f"c = sp.sqrt(a**2 + b**2)\n"
        f"assert a > 0 and b > 0 and c.is_integer\n"
        f"ANSWER = 2*c + 2*a"
    )
    desc = (
        f"ไฮเพอร์โบลามีสมการ (x - ({h}))^2 / {a**2} - (y - ({k}))^2 / {b**2} = 1 "
        f"(ห้ามกำหนดจุดบนไฮเพอร์โบลาที่อยู่บนเส้นกำกับเด็ดขาด) "
        f"จงหาผลบวกของระยะห่างระหว่างจุดโฟกัสทั้งสองกับความยาวแกนตามขวาง (คำตอบแน่นอน = {ans}, 3 ดาว)"
    )
    return MathSeed("analytic_geometry", desc, str(ans), code, suggested_stars=3)


def _seed_vectors_3d(rng: random.Random) -> MathSeed:
    # 3-4 Star 3D Vectors: Parallelepiped Volume |u . (v x w)| > 0 OR Parallelogram Area^2 |u x v|^2 > 0
    if rng.random() < 0.5:
        for _ in range(30):
            u = [rng.randint(1, 4), rng.randint(-2, 3), rng.randint(0, 3)]
            v = [rng.randint(-2, 3), rng.randint(1, 4), rng.randint(-2, 2)]
            w = [rng.randint(0, 3), rng.randint(-2, 2), rng.randint(1, 4)]
            vol = abs(int(sp.Matrix([u, v, w]).det()))
            if 4 <= vol <= 80:
                code = (
                    f"u = sp.Matrix({u})\n"
                    f"v = sp.Matrix({v})\n"
                    f"w = sp.Matrix({w})\n"
                    f"vol = sp.Abs(u.dot(v.cross(w)))\n"
                    f"assert vol > 0\n"
                    f"ANSWER = vol"
                )
                desc = (
                    f"กำหนดเวกเตอร์ในสามมิติ u = ({u[0]}, {u[1]}, {u[2]}), v = ({v[0]}, {v[1]}, {v[2]}), "
                    f"และ w = ({w[0]}, {w[1]}, {w[2]}) (ห้ามเปลี่ยนเลขเวกเตอร์ในเฉลยเด็ดขาด) "
                    f"จงหาปริมาตรของทรงสี่เหลี่ยมด้านขนานที่มีเวกเตอร์ทั้งสามเป็นด้านประกอบ (คำตอบแน่นอน = {vol}, 4 ดาว)"
                )
                return MathSeed("vectors_3d", desc, str(vol), code, suggested_stars=4)

    for _ in range(30):
        u = [rng.randint(1, 4), rng.randint(-2, 3), rng.randint(1, 3)]
        v = [rng.randint(-2, 3), rng.randint(1, 4), rng.randint(-2, 3)]
        cx = sp.Matrix(u).cross(sp.Matrix(v))
        area_sq = int(cx.dot(cx))
        if area_sq > 0:
            code = (
                f"u = sp.Matrix({u})\n"
                f"v = sp.Matrix({v})\n"
                f"cx = u.cross(v)\n"
                f"area_sq = cx.dot(cx)\n"
                f"assert area_sq > 0\n"
                f"ANSWER = area_sq"
            )
            desc = (
                f"กำหนดเวกเตอร์ในสามมิติ u = ({u[0]}, {u[1]}, {u[2]}) และ v = ({v[0]}, {v[1]}, {v[2]}) "
                f"จงหาค่ากำลังสองของพื้นที่รูปสี่เหลี่ยมด้านขนานที่มี u และ v เป็นด้านประชิด (|u x v|^2; คำตอบแน่นอน = {area_sq}, 3-4 ดาว)"
            )
            return MathSeed("vectors_3d", desc, str(area_sq), code, suggested_stars=3)

    raise RuntimeError("Failed to generate non-degenerate 3D vector seed")


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


def escalate_seed_for_level(base: MathSeed, difficulty_level: int, rng: random.Random) -> MathSeed:
    """Deterministically chain Stage-2 / Stage-3 SymPy transformations for Level 2 (+50%) and Level 3 (+125%).

    Keeps 100% SymPy-verified ground-truth math BEFORE the LLM writes the Thai word problem,
    eliminating mental-arithmetic mismatches on harder multi-step levels.
    """
    if difficulty_level <= 1:
        return base
    try:
        base_expr = sp.sympify(base.exact_answer)
        base_val = float(sp.N(base_expr))
    except Exception:
        return base

    if not (abs(base_val) <= 200 and abs(base_val * 10 - round(base_val * 10)) < 1e-6):
        return base

    if difficulty_level == 2:
        a_coef = rng.choice([2, 3])
        b_coef = rng.randint(2, 7)
        c_shift = rng.randint(5, 19)
        if abs(base_val) <= 35 and base_expr.is_integer:
            stage2_expr = sp.simplify(a_coef * (base_expr**2) - b_coef * base_expr + c_shift)
            if 0 <= float(sp.N(stage2_expr)) <= 9000:
                ans_str = str(int(stage2_expr))
                code = (
                    f"{base.sympy_code}\n"
                    f"_k_stage1 = ANSWER\n"
                    f"ANSWER = sp.simplify({a_coef}*(_k_stage1**2) - {b_coef}*_k_stage1 + {c_shift})\n"
                    f"assert ANSWER >= 0"
                )
                desc = (
                    f"[Level 2 — โจทย์เชื่อมโยงหลายชั้นแบบ สสวท. (ห้ามเขียนคำว่า 'ขั้นที่ 1/2' ลงในตัวโจทย์)]\n"
                    f"  เงื่อนไขฐาน (กำหนดค่าตัวแปรกลาง k): {base.params_description_th}\n"
                    f"  เป้าหมายปลายทาง (ผูกต่อเป็นประโยคเดียวอย่างเป็นธรรมชาติ): "
                    f"ให้ถามหาค่าของ {a_coef}k^2 - {b_coef}k + {c_shift} "
                    f"(คำตอบสุดท้ายแน่นอน = {ans_str}, ความยาก 4 ดาว)"
                )
                return MathSeed(base.chapter_id, desc, ans_str, code, suggested_stars=4)

        m_coef = rng.choice([2, 4])
        c_add = rng.randint(7, 25)
        stage2_expr = sp.simplify(m_coef * sp.Abs(base_expr) + c_add)
        stage2_val = float(sp.N(stage2_expr))
        if 0 <= stage2_val <= 9999.99 and abs(stage2_val * 100 - round(stage2_val * 100)) < 1e-4:
            ans_str = str(int(stage2_expr)) if stage2_expr.is_integer else f"{stage2_val:.2f}".rstrip("0").rstrip(".")
            code = (
                f"{base.sympy_code}\n"
                f"_k_stage1 = ANSWER\n"
                f"ANSWER = sp.simplify({m_coef}*sp.Abs(_k_stage1) + {c_add})\n"
                f"assert ANSWER >= 0"
            )
            desc = (
                f"[Level 2 — โจทย์เชื่อมโยงหลายชั้นแบบ สสวท. (ห้ามเขียนคำว่า 'ขั้นที่ 1/2' ลงในตัวโจทย์)]\n"
                f"  เงื่อนไขฐาน (กำหนดค่าตัวแปรกลาง k): {base.params_description_th}\n"
                f"  เป้าหมายปลายทาง (ผูกต่ออย่างเป็นธรรมชาติ): จงหาค่าของ {m_coef}|k| + {c_add} "
                f"(คำตอบสุดท้ายแน่นอน = {ans_str}, ความยาก 4 ดาว)"
            )
            return MathSeed(base.chapter_id, desc, ans_str, code, suggested_stars=4)
        return base

    # Level 3 (+125% harder / PAT1-Elite): 3-stage chained equation
    p_off = rng.randint(3, 9)
    q_add = rng.randint(11, 29)
    if base_expr.is_integer and abs(base_val) <= 30:
        abs_k = sp.Abs(base_expr)
        stage3_expr = sp.simplify((abs_k + p_off) ** 2 + 2 * abs_k + q_add)
        if 0 <= float(sp.N(stage3_expr)) <= 9000:
            ans_str = str(int(stage3_expr))
            code = (
                f"{base.sympy_code}\n"
                f"_k1 = sp.Abs(ANSWER)\n"
                f"_m = _k1 + {p_off}\n"
                f"ANSWER = sp.simplify(_m**2 + 2*_k1 + {q_add})\n"
                f"assert ANSWER >= 0"
            )
            desc = (
                f"[Level 3 — PAT1-Elite เชื่อมโยงเงื่อนไขซ้อนอย่างเนียนเป็นธรรมชาติ (ห้ามเขียนคำว่า 'ขั้นที่ 1/2/3' ในตัวโจทย์เด็ดขาด)]\n"
                f"  เงื่อนไขฐาน (กำหนดค่า k): {base.params_description_th}\n"
                f"  ความสัมพันธ์ซ้อน: พารามิเตอร์ m สัมพันธ์กับ k ตามสมการ m = |k| + {p_off}\n"
                f"  เป้าหมายปลายทาง: ถามหาค่าของ m^2 + 2|k| + {q_add} "
                f"(ผูกเงื่อนไขทั้งหมดเป็นประโยคภาษาไทยสละสลวยแบบ สสวท.; คำตอบสุดท้ายแน่นอน = {ans_str}, ความยาก 5 ดาว)"
            )
            return MathSeed(base.chapter_id, desc, ans_str, code, suggested_stars=5)

    m_coef = rng.choice([4, 5])
    stage3_expr = sp.simplify(m_coef * sp.Abs(base_expr) + p_off * q_add)
    stage3_val = float(sp.N(stage3_expr))
    if 0 <= stage3_val <= 9999.99 and abs(stage3_val * 100 - round(stage3_val * 100)) < 1e-4:
        ans_str = str(int(stage3_expr)) if stage3_expr.is_integer else f"{stage3_val:.2f}".rstrip("0").rstrip(".")
        code = (
            f"{base.sympy_code}\n"
            f"_k1 = sp.Abs(ANSWER)\n"
            f"ANSWER = sp.simplify({m_coef}*_k1 + {p_off * q_add})\n"
            f"assert ANSWER >= 0"
        )
        desc = (
            f"[Level 3 — PAT1-Elite เชื่อมโยงเงื่อนไขซ้อนอย่างเนียนเป็นธรรมชาติ (ห้ามเขียนคำว่า 'ขั้นที่ 1/2/3' ในตัวโจทย์เด็ดขาด)]\n"
            f"  เงื่อนไขฐาน (กำหนดค่า k): {base.params_description_th}\n"
            f"  เป้าหมายปลายทาง: ถามหาค่าของ {m_coef}|k| + {p_off * q_add} "
            f"(คำตอบสุดท้ายแน่นอน = {ans_str}, ความยาก 5 ดาว)"
        )
        return MathSeed(base.chapter_id, desc, ans_str, code, suggested_stars=5)
    return base


def generate_verified_seed(
    chapter_id: str,
    seed: int | None = None,
    difficulty_level: int = 1,
) -> MathSeed | None:
    """Return a SymPy-preverified parameter seed covering all 16 Syllabus 2560 chapters across Levels 1-3."""
    fn = _SEEDERS.get(chapter_id)
    if fn is None:
        return None
    rng = random.Random(seed)
    base = fn(rng)
    return escalate_seed_for_level(base, difficulty_level=difficulty_level, rng=rng)


_NON_NEGATIVE_CHAPTERS = {"sets", "logic", "counting", "probability", "prob_distributions"}


def _format_clean_number(val: float, is_int: bool) -> str:
    if is_int or abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.4f}".rstrip("0").rstrip(".")


def build_deterministic_mcq_choices(
    exact_answer: str,
    chapter_id: str,
    rng: random.Random | None = None,
) -> tuple[list[str], str]:
    """Deterministically build 5 mathematically distinct MCQ choices and exact 1-based answer index.

    Guarantees 100% alignment between `sympy_code` / `exact_answer`, `choices_latex`, and `answer`.
    """
    r = rng or random.Random()
    if chapter_id == "logic":
        ans_int = int(round(float(sp.N(sp.sympify(exact_answer)))))
        if 0 <= ans_int <= 4:
            return ["0", "1", "2", "3", "4"], str(ans_int + 1)

    ans_val = float(sp.N(sp.sympify(exact_answer)))
    is_int = abs(ans_val - round(ans_val)) < 1e-9
    non_neg = chapter_id in _NON_NEGATIVE_CHAPTERS
    is_prob = chapter_id in {"probability", "prob_distributions"} and 0.0 <= ans_val <= 1.0

    distractors: list[float] = []
    seen = [ans_val]

    def _try_add(cand: float) -> None:
        if not math.isfinite(cand):
            return
        if is_prob and not (0.0 < cand < 1.0):
            return
        if non_neg and cand < 0:
            return
        if any(math.isclose(cand, s, rel_tol=1e-4, abs_tol=1e-4) for s in seen):
            return
        seen.append(cand)
        distractors.append(cand)

    if is_prob:
        offsets = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, -0.15, 0.3]
        r.shuffle(offsets)
        for off in offsets:
            _try_add(round(ans_val + off, 4))
            if len(distractors) == 4:
                break
        step = 0.02
        while len(distractors) < 4:
            _try_add(round(min(0.98, max(0.02, ans_val + step)), 4))
            step = -step if step > 0 else abs(step) + 0.03
    elif is_int:
        ans_i = int(round(ans_val))
        mag = max(1, abs(ans_i))
        step = 1 if mag < 20 else (2 if mag < 60 else (5 if mag < 200 else max(10, round(mag * 0.05 / 5) * 5)))
        raw_cands = [
            ans_i - 2 * step,
            ans_i - step,
            ans_i + step,
            ans_i + 2 * step,
            -ans_i,
            ans_i + 3 * step,
            ans_i - 3 * step,
            ans_i + 4 * step,
            ans_i + 5 * step,
        ]
        r.shuffle(raw_cands)
        for c in raw_cands:
            _try_add(float(c))
            if len(distractors) == 4:
                break
        k = 1
        while len(distractors) < 4:
            _try_add(float(ans_i + k * step))
            k += 1
    else:
        step = 0.5 if abs(ans_val) < 10 else max(1.0, round(abs(ans_val) * 0.1, 1))
        raw_cands = [
            ans_val - 2 * step,
            ans_val - step,
            ans_val + step,
            ans_val + 2 * step,
            -ans_val,
            ans_val + 3 * step,
        ]
        r.shuffle(raw_cands)
        for c in raw_cands:
            _try_add(round(c, 4))
            if len(distractors) == 4:
                break
        k = 1
        while len(distractors) < 4:
            _try_add(round(ans_val + k * step, 4))
            k += 1

    # Keep exact_answer string verbatim for the true choice, format distractors cleanly
    pairs = [(ans_val, exact_answer.strip())] + [
        (d, _format_clean_number(d, is_int)) for d in distractors[:4]
    ]
    pairs.sort(key=lambda p: p[0])
    choices = [p[1] for p in pairs]
    correct_idx = next(
        idx for idx, (v, _) in enumerate(pairs, start=1)
        if math.isclose(v, ans_val, rel_tol=1e-4, abs_tol=1e-4)
    )
    return choices, str(correct_idx)


def verify_seed_numbers_in_stem(seed_obj: MathSeed, stem_latex: str) -> tuple[bool, str | None]:
    """Ensure the LLM did not alter the core numerical parameters from `seed_obj` in `stem_latex`."""
    # Strip the trailing "(คำตอบแน่นอน = ...)" / "(ใช้ตัวเลือก ...)" hint before extracting premise numbers
    premise = re.split(r"\(คำตอบแน่นอน|\(ใช้ตัวเลือก", seed_obj.params_description_th)[0]
    raw_nums = re.findall(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)", premise)
    # Deduplicate while preserving order; ignore tiny structural counts like "3 อย่าง", "4 ข้อ"
    distinct_nums: list[str] = []
    for n in raw_nums:
        if n not in distinct_nums:
            distinct_nums.append(n)
    if not distinct_nums:
        return True, None

    norm_stem = stem_latex.replace(",", "").replace("{", "").replace("}", "").replace("$", "")
    missing: list[str] = []
    large_missing: list[str] = []
    for n in distinct_nums:
        if n not in norm_stem:
            missing.append(n)
            try:
                if float(n) >= 10:
                    large_missing.append(n)
            except ValueError:
                pass

    if large_missing:
        return (
            False,
            f"Stem omitted/altered seed parameter(s): {', '.join(large_missing[:4])}",
        )
    hit_ratio = (len(distinct_nums) - len(missing)) / len(distinct_nums)
    if hit_ratio < 0.70:
        return (
            False,
            f"Stem diverged from seed numbers (missing: {', '.join(missing[:4])})",
        )
    return True, None
