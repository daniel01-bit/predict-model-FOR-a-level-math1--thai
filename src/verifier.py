"""Deterministic SymPy symbolic verifier and originality gate for v3.0 Mock Exam questions."""

from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass

from src.schema import QuestionFormat


_WORKER_SCRIPT = r"""
import ast, decimal, fractions, itertools, json, math, statistics, sys
import sympy as sp

if not hasattr(sp, "comb"):
    sp.comb = sp.binomial
if not hasattr(sp, "perm"):
    sp.perm = lambda n, r=None: sp.factorial(n) if r is None else sp.factorial(n) // sp.factorial(n - r)

ALLOWED_MODULES = {"sympy", "math", "itertools", "fractions", "statistics", "decimal"}
FORBIDDEN_NAMES = {
    "__builtins__", "__import__", "__loader__", "__spec__", "__file__",
    "eval", "exec", "compile", "open", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "breakpoint", "input", "type", "object",
}

def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Forbidden dunder attribute access: {node.attr}")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError(f"Forbidden identifier in SymPy sandbox: {node.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise ImportError(f"Module '{alias.name}' is not allowed in SymPy sandbox")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_MODULES:
                raise ImportError(f"Module '{node.module}' is not allowed in SymPy sandbox")

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in SymPy sandbox")
    return __import__(name, globals, locals, fromlist, level)


class ALevelSymPyGate:
    @staticmethod
    def verify_sequence_n(a1, d, b1, r, target_sum):
        n = sp.Symbol("n", integer=True, positive=True)
        eq = sp.Eq((a1 + (n - 1) * d) + (b1 * r ** (n - 1)), target_sum)
        sol = [s for s in sp.solve(eq, n) if s.is_integer and s > 0]
        assert len(sol) == 1, f"Invalid n for sequence equation: {sol}"
        return int(sol[0])

    @staticmethod
    def verify_infinite_series(expr_str, start_idx=1):
        n = sp.Symbol("n", integer=True, positive=True)
        expr = sp.sympify(expr_str, locals={"n": n})
        total = sp.summation(expr, (n, start_idx, sp.oo))
        assert total.is_finite and total.is_real, f"Series diverges or invalid: {total}"
        return total

    @staticmethod
    def verify_function_range(expr_str, x_min, x_max):
        x = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={"x": x})
        rng = sp.calculus.util.function_range(expr, x, sp.Interval(x_min, x_max))
        assert isinstance(rng, sp.Interval), f"Discontinuous or invalid range: {rng}"
        return rng.start, rng.end

    @staticmethod
    def verify_piecewise_continuity(pieces_with_bounds):
        x = sp.Symbol("x", real=True)
        for i in range(len(pieces_with_bounds) - 1):
            expr_left, c_point = pieces_with_bounds[i]
            expr_right, _ = pieces_with_bounds[i + 1]
            lim_l = sp.limit(sp.sympify(expr_left, locals={"x": x}), x, c_point, dir="-")
            lim_r = sp.limit(sp.sympify(expr_right, locals={"x": x}), x, c_point, dir="+")
            assert sp.simplify(lim_l - lim_r) == 0, f"Discontinuous at x={c_point}: {lim_l} != {lim_r}"
        return True

    @staticmethod
    def verify_conic_params(conic_type, a, b, c):
        assert a > 0 and b > 0 and c > 0, "Conic parameters must be positive"
        if conic_type == "ellipse":
            assert sp.simplify(a**2 - (b**2 + c**2)) == 0, f"Invalid ellipse: {a}^2 != {b}^2 + {c}^2"
        elif conic_type == "hyperbola":
            assert sp.simplify(c**2 - (a**2 + b**2)) == 0, f"Invalid hyperbola: {c}^2 != {a}^2 + {b}^2"
        else:
            raise ValueError(f"Unknown conic_type: {conic_type}")
        return True

    @staticmethod
    def verify_curves_intersect(eq1_str, eq2_str):
        x, y = sp.symbols("x y", real=True)
        sols = sp.solve(
            [sp.sympify(eq1_str, locals={"x": x, "y": y}), sp.sympify(eq2_str, locals={"x": x, "y": y})],
            (x, y),
        )
        assert len(sols) > 0, "Curves do not intersect in real plane!"
        return sols

    @staticmethod
    def verify_vector_area(u_list, v_list):
        u = sp.Matrix(u_list)
        v = sp.Matrix(v_list)
        area = sp.simplify(u.cross(v).norm())
        assert area > 0, "Vectors are parallel (Area = 0)!"
        return area

    @staticmethod
    def verify_vector_volume(u_list, v_list, w_list):
        mat = sp.Matrix([u_list, v_list, w_list])
        vol = sp.Abs(mat.det())
        assert vol > 0, "Vectors are coplanar (Volume = 0)!"
        return vol

SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs, "min": min, "max": max, "sum": sum, "range": range,
    "len": len, "int": int, "float": float, "round": round, "pow": pow,
    "enumerate": enumerate, "zip": zip, "list": list, "tuple": tuple,
    "set": set, "dict": dict, "sorted": sorted, "any": any, "all": all,
    "bool": bool, "str": str, "map": map, "filter": filter, "isinstance": isinstance,
    "AssertionError": AssertionError, "ValueError": ValueError,
    "RuntimeError": RuntimeError, "Exception": Exception,
}

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        req = json.loads(raw)
        code = req["code"]
        _validate_ast(code)
        ns = {
            "__builtins__": SAFE_BUILTINS,
            "sp": sp,
            "math": math,
            "itertools": itertools,
            "fractions": fractions,
            "statistics": statistics,
            "decimal": decimal,
            "ALevelSymPyGate": ALevelSymPyGate,
        }
        exec(code, ns)
        if "ANSWER" not in ns:
            for alt_key in ("ans", "answer", "result"):
                if alt_key in ns:
                    ns["ANSWER"] = ns[alt_key]
                    break
            else:
                raise RuntimeError("sympy_code did not define ANSWER")
        val = ns["ANSWER"]
        # Global Gate 1: Non-empty / non-None
        if val is None or val == [] or val == {} or val == set() or val == sp.EmptySet:
            raise ValueError("Empty or None ANSWER (contradictory/unsolvable conditions)")
        # Global Gate 2: Reject ambiguous multi-solution collections
        if isinstance(val, dict):
            if len(val) != 1:
                raise ValueError(f"Ambiguous solution dict: {val}")
            val = next(iter(val.values()))
        if isinstance(val, (list, tuple, set)):
            if len(val) != 1:
                raise ValueError(f"Ambiguous or empty solution collection: {val}")
            val = next(iter(val))
        if isinstance(val, sp.FiniteSet):
            if len(val) != 1:
                raise ValueError(f"Ambiguous or empty FiniteSet: {val}")
            val = next(iter(val))
        # Global Gate 3: Zero free symbols (catches under-determined / missing-info problems)
        if isinstance(val, sp.Basic):
            if len(val.free_symbols) > 0:
                raise ValueError(f"Under-determined: ANSWER still has free symbols {sorted(str(s) for s in val.free_symbols)} -> {val}")
            if val in (sp.oo, -sp.oo, sp.zoo, sp.nan) or val.is_finite is False:
                raise ValueError(f"Non-finite ANSWER: {val}")
        # Global Gate 4: Must evaluate to a finite real number (no stray imaginary part)
        cval = complex(sp.N(val))
        if abs(cval.imag) >= 1e-7 or not math.isfinite(cval.real):
            raise ValueError(f"Non-real or non-finite ANSWER: {val}")
        sys.stdout.write(json.dumps({"ok": True, "answer": str(float(cval.real))}) + "\n")
        sys.stdout.flush()
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        sys.stdout.flush()
"""


class _PersistentSymPyWorker:
    """Keeps one isolated Python process with SymPy pre-imported (~50x faster than cold spawn)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _ensure_started(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, "-I", "-u", "-c", _WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        return self._proc

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1)
            except Exception:
                pass
            self._proc = None

    def run(self, sympy_code: str, timeout_seconds: float = 5.0) -> tuple[bool, str]:
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(json.dumps({"code": sympy_code}) + "\n")
                proc.stdin.flush()
            except Exception:
                self._kill()
                proc = self._ensure_started()
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(json.dumps({"code": sympy_code}) + "\n")
                proc.stdin.flush()

            out_q: queue.Queue[str] = queue.Queue(maxsize=1)

            def _reader() -> None:
                try:
                    line = proc.stdout.readline()  # type: ignore[union-attr]
                    out_q.put(line)
                except Exception:
                    out_q.put("")

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            try:
                line = out_q.get(timeout=timeout_seconds)
            except queue.Empty:
                self._kill()
                return False, f"SymPy execution timed out after {timeout_seconds}s"

            if not line:
                self._kill()
                return False, "SymPy worker exited unexpectedly"

            try:
                payload = json.loads(line)
                if payload.get("ok"):
                    return True, str(payload["answer"])
                return False, f"SymPy error: {payload.get('error', 'unknown')}"
            except Exception as exc:
                self._kill()
                return False, f"Invalid JSON from SymPy worker: {exc}"


_SYMPY_WORKER = _PersistentSymPyWorker()


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    computed_answer: str | None
    expected_answer: str
    max_stem_similarity: float
    error: str | None = None


def _tokenize(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\u0E00-\u0E7F]+", " ", text.lower())
    return {tok for tok in cleaned.split() if len(tok) > 1}


def stem_jaccard_similarity(stem_a: str, stem_b: str) -> float:
    a, b = _tokenize(stem_a), _tokenize(stem_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_sympy_code(sympy_code: str, timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Execute SymPy verification snippet in the persistent isolated worker."""
    return _SYMPY_WORKER.run(sympy_code, timeout_seconds=timeout_seconds)


def _parse_numeric(value: str) -> float | None:
    cleaned = (
        value.replace(",", "")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace(r"\%", "")
        .replace("%", "")
        .replace(r"^\circ", "")
        .replace("°", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Support full A-Level LaTeX expressions: \dfrac, \tfrac, \sqrt[n]{x}, \log_{b}{x}, trig, e^x, \pi
    expr = value.replace("$", "").replace(",", "").strip()
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = expr.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    expr = expr.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    expr = expr.replace(r"\%", "").replace("%", "").replace(r"^\circ", "").replace("°", "")
    expr = expr.replace("^", "**").replace(r"\pi", "pi")
    for fn_latex, fn_sympy in (
        (r"\arccos", "acos"),
        (r"\arcsin", "asin"),
        (r"\arctan", "atan"),
        (r"\cos", "cos"),
        (r"\sin", "sin"),
        (r"\tan", "tan"),
        (r"\ln", "log"),
    ):
        expr = expr.replace(fn_latex, fn_sympy)
    # Iteratively expand nested \frac{a}{b}, \sqrt[n]{a}, \sqrt{a}, and \log_{b}{a}
    for _ in range(5):
        prev = expr
        expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", expr)
        expr = re.sub(r"\\sqrt\s*\[([^\[\]]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", expr)
        expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
        expr = re.sub(r"\\log_\{([^{}]+)\}\s*\{([^{}]+)\}", r"(log(\2)/log(\1))", expr)
        expr = re.sub(r"\\log_\{([^{}]+)\}\s*\(([^()]+)\)", r"(log(\2)/log(\1))", expr)
        expr = re.sub(r"\\log_(\d+)\s*\{([^{}]+)\}", r"(log(\2)/log(\1))", expr)
        expr = re.sub(r"\\log_(\d+)\s*\(([^()]+)\)", r"(log(\2)/log(\1))", expr)
        if expr == prev:
            break
    expr = re.sub(r"\\log\s*\{([^{}]+)\}", r"(log(\1)/log(10))", expr)
    expr = re.sub(r"\\log\s*\(([^()]+)\)", r"(log(\1)/log(10))", expr)
    expr = expr.replace("{", "(").replace("}", ")")
    # Insert implicit multiplication: 2sqrt(3) -> 2*sqrt(3), 2pi -> 2*pi, 2(3) -> 2*(3), )( -> )*(
    expr = re.sub(r"(\d)\s*(?=(?:sqrt|pi|acos|asin|atan|cos|sin|tan|log|e\*\*|\())", r"\1*", expr)
    expr = re.sub(r"\)\s*(?=(?:\d|sqrt|pi|\())", r")*", expr)
    # Strip leading variable assignment like "x = 4" or "t = 1" if purely numeric RHS
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    if re.fullmatch(r"[0-9a-zA-Z\s\+\-\*/\(\)\._]+", expr):
        try:
            import sympy as sp

            val = complex(sp.N(sp.sympify(expr, locals={"e": sp.E})))
            if abs(val.imag) < 1e-9 and math.isfinite(val.real):
                return float(val.real)
        except Exception:
            return None
    return None


def _numbers_match(comp_num: float, exp_num: float, exp_raw: str) -> bool:
    """Match SymPy ground truth against expected numeric/choice string with A-Level rounding support.

    Accepts exact matches (rel_tol=1e-4) AND exact decimal rounding to `d >= 2` decimal places
    (e.g. SymPy=23.4375 vs A-Level 2-decimal sheet expected='23.44'), plus percentage scaling.
    """
    if math.isclose(comp_num, exp_num, rel_tol=1e-4, abs_tol=1e-4):
        return True

    stripped = exp_raw.replace(",", "").replace("$", "").strip()
    # Percentage equivalence (e.g., SymPy=0.25 vs expected='25%' or vice versa)
    if stripped.endswith("%") or stripped.endswith(r"\%"):
        if math.isclose(comp_num * 100.0, exp_num, rel_tol=1e-3, abs_tol=1e-2):
            return True
        if math.isclose(comp_num, exp_num / 100.0, rel_tol=1e-3, abs_tol=1e-4):
            return True

    # Decimal rounding equivalence: e.g. '23.44' (d=2) matches 23.4375 since |23.4375 - 23.44| <= 0.005
    m_dec = re.fullmatch(r"[\-\+]?\d+\.(\d+)", stripped)
    if m_dec:
        dec_places = len(m_dec.group(1))
        if dec_places >= 2:
            half_ulp = 0.5 * (10.0 ** (-dec_places))
            if abs(comp_num - exp_num) <= half_ulp + 1e-9:
                return True
    return False


RED_FLAG_PHRASES = (
    "ขออภัย",
    "ปรับโจทย์",
    "แก้โจทย์",
    "แก้ไขโจทย์",
    "เปลี่ยนโจทย์",
    "สมมติโจทย์",
    "ปรับตัวเลข",
    "ปรับเวกเตอร์",
    "ปรับสมการ",
    "ปรับค่า",
    "เพื่อให้ได้ค่า",
    "เพื่อให้ได้คำตอบ",
    "เพื่อให้ลงตัว",
    "แก้โดยใช้ Python",
    "โจทย์ผิด",
    "ไม่มีคำตอบในตัวเลือก",
    "ไม่มีจำนวนเต็มที่ตรง",
    "ขัดแย้งกัน",
    "ไม่สอดคล้อง",
    "หาคำตอบไม่ได้",
    "adjust the problem",
)


def sanitize_latex_text(text: str) -> str:
    """Repair common JSON-eaten LaTeX control characters without corrupting \\neq, \\neg, \\notin, etc."""
    if not text:
        return text
    text = text.replace("\t", r"\t").replace("\x0c", r"\f").replace("\x08", r"\b").replace("\r", r"\r")
    text = re.sub(r"(\d|\$|\s)imes(\d|\$|\s)", r"\1\\times\2", text)
    text = re.sub(r"(?<![a-zA-Z\\])rac\{", r"\\frac{", text)
    text = re.sub(r"(?<![a-zA-Z\\])ext\{", r"\\text{", text)
    text = re.sub(r"(?<!\\)\bightarrow\b", r"\\rightarrow", text)
    text = re.sub(r"(?<!\\)\bheta\b", r"\\theta", text)
    text = re.sub(r"(?<!\\)\borall\b", r"\\forall", text)
    # Only convert literal \n when NOT followed by an ASCII letter (preserves \neq, \neg, \notin, \nu, \nabla)
    text = re.sub(r"\\n(?![a-zA-Z])", "\n", text)
    return text.strip()


class ALevelSymPyGate:
    """Reusable deterministic SymPy validators for the 4 highest-error A-Level Math 1 chapters."""

    @staticmethod
    def verify_sequence_n(a1, d, b1, r, target_sum) -> int:
        import sympy as sp

        n = sp.Symbol("n", integer=True, positive=True)
        eq = sp.Eq((a1 + (n - 1) * d) + (b1 * r ** (n - 1)), target_sum)
        sol = [s for s in sp.solve(eq, n) if s.is_integer and s > 0]
        assert len(sol) == 1, f"Invalid n for sequence equation: {sol}"
        return int(sol[0])

    @staticmethod
    def verify_infinite_series(expr_str: str, start_idx: int = 1):
        import sympy as sp

        n = sp.Symbol("n", integer=True, positive=True)
        expr = sp.sympify(expr_str, locals={"n": n})
        total = sp.summation(expr, (n, start_idx, sp.oo))
        assert total.is_finite and total.is_real, f"Series diverges or invalid: {total}"
        return total

    @staticmethod
    def verify_function_range(expr_str: str, x_min, x_max):
        import sympy as sp

        x = sp.Symbol("x", real=True)
        expr = sp.sympify(expr_str, locals={"x": x})
        rng = sp.calculus.util.function_range(expr, x, sp.Interval(x_min, x_max))
        assert isinstance(rng, sp.Interval), f"Discontinuous or invalid range: {rng}"
        return rng.start, rng.end

    @staticmethod
    def verify_piecewise_continuity(pieces_with_bounds: list[tuple[str, float | int | None]]) -> bool:
        import sympy as sp

        x = sp.Symbol("x", real=True)
        for i in range(len(pieces_with_bounds) - 1):
            expr_left, c_point = pieces_with_bounds[i]
            expr_right, _ = pieces_with_bounds[i + 1]
            lim_l = sp.limit(sp.sympify(expr_left, locals={"x": x}), x, c_point, dir="-")
            lim_r = sp.limit(sp.sympify(expr_right, locals={"x": x}), x, c_point, dir="+")
            assert sp.simplify(lim_l - lim_r) == 0, f"Discontinuous at x={c_point}: {lim_l} != {lim_r}"
        return True

    @staticmethod
    def verify_conic_params(conic_type: str, a, b, c) -> bool:
        import sympy as sp

        assert a > 0 and b > 0 and c > 0, "Conic parameters must be positive"
        if conic_type == "ellipse":
            assert sp.simplify(a**2 - (b**2 + c**2)) == 0, f"Invalid ellipse: {a}^2 != {b}^2 + {c}^2"
        elif conic_type == "hyperbola":
            assert sp.simplify(c**2 - (a**2 + b**2)) == 0, f"Invalid hyperbola: {c}^2 != {a}^2 + {b}^2"
        else:
            raise ValueError(f"Unknown conic_type: {conic_type}")
        return True

    @staticmethod
    def verify_curves_intersect(eq1_str: str, eq2_str: str):
        import sympy as sp

        x, y = sp.symbols("x y", real=True)
        sols = sp.solve(
            [sp.sympify(eq1_str, locals={"x": x, "y": y}), sp.sympify(eq2_str, locals={"x": x, "y": y})],
            (x, y),
        )
        assert len(sols) > 0, "Curves do not intersect in real plane!"
        return sols

    @staticmethod
    def verify_vector_area(u_list: list, v_list: list):
        import sympy as sp

        u = sp.Matrix(u_list)
        v = sp.Matrix(v_list)
        area = sp.simplify(u.cross(v).norm())
        assert area > 0, "Vectors are parallel (Area = 0)!"
        return area

    @staticmethod
    def verify_vector_volume(u_list: list, v_list: list, w_list: list):
        import sympy as sp

        mat = sp.Matrix([u_list, v_list, w_list])
        vol = sp.Abs(mat.det())
        assert vol > 0, "Vectors are coplanar (Volume = 0)!"
        return vol


def sanitize_choice_text(choice: str) -> str:
    """Clean accidental numbering prefixes like '1.) ', '1) ', '1. ', or '\\mathbf{1}\\text{.) }' from choices."""
    cleaned = sanitize_latex_text(choice)
    cleaned = re.sub(r"^\\mathbf\{\d+\}\\text\{\.?\)\s*\}\s*", "", cleaned)
    cleaned = re.sub(r"^\(?\d+\.?\)\s+", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
    return cleaned.strip()


def check_latex_and_choice_quality(
    stem_latex: str,
    question_format: QuestionFormat,
    choices_latex: list[str],
    solution_steps_th: str = "",
) -> str | None:
    """Deterministic gate catching AI hedging red-flags, Lao script leaks, broken LaTeX, and choice mismatches."""
    combined_prose = f"{stem_latex}\n{solution_steps_th}"
    for flag in RED_FLAG_PHRASES:
        if flag in combined_prose:
            return f"AI hedging/apology red-flag phrase detected: {flag!r}"

    # Block accidental Lao Unicode characters (U+0E80-U+0EFF) mixed into Thai text (e.g. 'ຄ້າ')
    if re.search(r"[\u0E80-\u0EFF]", combined_prose):
        return "Lao script character detected in Thai math text"

    all_texts = [stem_latex, solution_steps_th] + list(choices_latex)
    for txt in all_texts:
        if not txt:
            continue
        no_escaped = txt.replace(r"\$", "").replace("$$", "")
        if no_escaped.count("$") % 2 != 0:
            return f"Unbalanced '$' LaTeX delimiter in: {txt[:80]!r}"
        for bad_pat in (
            r"(?<![a-zA-Z\\])imes(?![a-zA-Z])",
            r"(?<![a-zA-Z\\])rac\{",
            r"(?<!\\)\bightarrow\b",
            r"(?<!\\)\bext\{",
            r"(?<!\\)\bheta\b",
            r"\\(arctan|arcsin|arccos|sin|cos|tan|ln|log)\(\s*\)",
            r"\\[\u0E00-\u0E7F]",
            r"\\n(?![a-zA-Z])",
            r"\bVIn\b",
            r"\bifa\b",
            r"\bifla\b",
            r"\\mathbf\{1\}",
        ):
            if re.search(bad_pat, txt):
                return f"Garbled LaTeX/text token matched {bad_pat!r} in: {txt[:80]!r}"

    if question_format == QuestionFormat.MCQ_5:
        stripped = [c.replace("$", "").strip() for c in choices_latex]
        if len(stripped) != 5 or any(not c for c in stripped):
            return "MCQ_5 must have exactly 5 non-empty choices"
        if any(re.search(r"\\infty|[\(\[][\-\d\s]+,\s*[\-\d\\\w]+[\)\]]", c) for c in choices_latex):
            return "Choices contain interval notation like (a, b) or \\infty; reframe the question to ask for a single scalar (e.g., a+b or sum/count of integer solutions)"
        if any(re.search(r"\\?\{[^\}]+,[^\}]+\\?\}", c) for c in choices_latex):
            return "Choices contain set notation like {-2, 1}; reframe to ask for a single scalar (e.g., sum or count of elements)"
        if any(re.search(r"\b(true|false)\b|จริง|เท็จ|\\text\{", c, re.I) for c in choices_latex):
            return "Choices contain boolean/Thai words instead of numeric values; reframe to ask for a numeric count"
        asks_which_prop = any(
            kw in stem_latex for kw in ("ประพจน์ใด", "ข้อใดสมมูล", "ข้อใดต่อไปนี้สมมูล", "ข้อใดเป็นสัจนิรันดร์")
        )
        asks_count = any(kw in stem_latex for kw in ("กี่ข้อ", "กี่กรณี", "จำนวน", "ผลบวก"))
        if asks_which_prop and not asks_count and stripped == ["1", "2", "3", "4", "5"]:
            return "Stem asks for an equivalent proposition/statement, but choices are bare numbers 1-5"

    return None


def verify_generated_question(
    stem_latex: str,
    question_format: QuestionFormat,
    choices_latex: list[str],
    expected_answer: str,
    sympy_code: str,
    historical_stems: list[str] | None = None,
    max_allowed_similarity: float = 0.65,
    solution_steps_th: str = "",
    chapter_id: str | None = None,
) -> VerificationResult:
    """Verify mathematical correctness via SymPy + LaTeX quality + uniqueness + non-duplication."""
    quality_err = check_latex_and_choice_quality(
        stem_latex, question_format, choices_latex, solution_steps_th=solution_steps_th
    )
    if quality_err:
        return VerificationResult(
            passed=False,
            computed_answer=None,
            expected_answer=expected_answer,
            max_stem_similarity=0.0,
            error=f"Quality gate failed: {quality_err}",
        )

    # Anti-hardcode check: sympy_code must actually perform computation, not just 'ANSWER = 5'
    non_comment_lines = [
        ln.strip() for ln in sympy_code.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    if len(non_comment_lines) <= 1 and re.fullmatch(r"ANSWER\s*=\s*[\-\+]?\d+(\.\d+)?", non_comment_lines[0] if non_comment_lines else ""):
        return VerificationResult(
            passed=False,
            computed_answer=None,
            expected_answer=expected_answer,
            max_stem_similarity=0.0,
            error="sympy_code hardcoded ANSWER without computing or asserting problem conditions",
        )

    historical_stems = historical_stems or []
    max_sim = max(
        (stem_jaccard_similarity(stem_latex, old) for old in historical_stems),
        default=0.0,
    )
    if max_sim >= max_allowed_similarity:
        return VerificationResult(
            passed=False,
            computed_answer=None,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=f"Stem too similar to historical exam (Jaccard={max_sim:.2f} >= {max_allowed_similarity})",
        )

    ok, computed = run_sympy_code(sympy_code)
    if not ok:
        return VerificationResult(
            passed=False,
            computed_answer=None,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=computed,
        )

    comp_num = _parse_numeric(computed)

    # Global Gate 5: Chapter & physical-quantity domain assertions
    if comp_num is not None:
        if chapter_id in ("counting", "sets") and (
            comp_num < -1e-6 or abs(comp_num - round(comp_num)) > 1e-4
        ):
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Chapter '{chapter_id}' requires non-negative integer answer, got {comp_num}",
            )
        if chapter_id == "probability" and comp_num < -1e-6:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Probability cannot be negative: {comp_num}",
            )
        if comp_num < -1e-6 and any(
            kw in stem_latex for kw in ("พื้นที่", "ปริมาตร", "ระยะห่าง", "ความยาว", "ส่วนเบี่ยงเบนมาตรฐาน", "ความแปรปรวน")
        ):
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Physical quantity (area/volume/distance/SD/variance) cannot be negative: {comp_num}",
            )

    if question_format == QuestionFormat.NUMERIC:
        if "/" in expected_answer or r"\frac" in expected_answer:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Part 2 (grid-in) answer cannot be a raw fraction ({expected_answer!r}); use a terminating decimal or scale the target expression",
            )
        exp_num = _parse_numeric(expected_answer)
        if comp_num is None or exp_num is None:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error="Numeric answer or SymPy output is not a valid number",
            )
        if comp_num < -1e-9 or exp_num < -1e-9:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Part 2 (grid-in) answer cannot be negative: {comp_num}",
            )
        if comp_num > 9999.99 + 1e-6 or exp_num > 9999.99 + 1e-6:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Part 2 (grid-in) answer exceeds A-Level grid limit (9999.99): {comp_num}",
            )
        if abs(comp_num * 100.0 - round(comp_num * 100.0)) > 1e-4:
            return VerificationResult(
                passed=False,
                computed_answer=computed,
                expected_answer=expected_answer,
                max_stem_similarity=round(max_sim, 4),
                error=f"Part 2 (grid-in) answer must have at most 2 decimal places: {comp_num}",
            )
        matched = _numbers_match(comp_num, exp_num, expected_answer)
        return VerificationResult(
            passed=matched,
            computed_answer=computed,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=None if matched else f"Mismatch: SymPy={comp_num}, expected={exp_num}",
        )

    # MCQ_5 validation
    if len(choices_latex) != 5 or len(set(c.strip() for c in choices_latex)) != 5:
        return VerificationResult(
            passed=False,
            computed_answer=computed,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error="MCQ must have 5 distinct choices",
        )

    if expected_answer not in {"1", "2", "3", "4", "5"}:
        return VerificationResult(
            passed=False,
            computed_answer=computed,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=f"Invalid MCQ answer index: {expected_answer!r}",
        )

    if comp_num is None:
        return VerificationResult(
            passed=False,
            computed_answer=computed,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=f"SymPy ANSWER={computed!r} is not a finite real number",
        )

    parsed_choices = [_parse_numeric(c) for c in choices_latex]
    if any(v is None for v in parsed_choices):
        bad_idx = next(i + 1 for i, v in enumerate(parsed_choices) if v is None)
        return VerificationResult(
            passed=False,
            computed_answer=computed,
            expected_answer=expected_answer,
            max_stem_similarity=round(max_sim, 4),
            error=f"Choice #{bad_idx} ({choices_latex[bad_idx - 1]!r}) is not a valid numeric/LaTeX expression",
        )

    # Ensure all 5 choices are mathematically distinct (blocks duplicate values like 0.5 and 1/2)
    for i in range(5):
        for j in range(i + 1, 5):
            if math.isclose(parsed_choices[i], parsed_choices[j], rel_tol=1e-4, abs_tol=1e-4):  # type: ignore[arg-type]
                return VerificationResult(
                    passed=False,
                    computed_answer=computed,
                    expected_answer=expected_answer,
                    max_stem_similarity=round(max_sim, 4),
                    error=f"Choices #{i+1} and #{j+1} are mathematically equal ({parsed_choices[i]})",
                )

    chosen_idx = int(expected_answer) - 1
    chosen_num = parsed_choices[chosen_idx]
    matched = _numbers_match(comp_num, chosen_num, choices_latex[chosen_idx])  # type: ignore[arg-type]

    return VerificationResult(
        passed=matched,
        computed_answer=computed,
        expected_answer=expected_answer,
        max_stem_similarity=round(max_sim, 4),
        error=None if matched else f"SymPy={computed!r} does not match choice #{expected_answer} ({choices_latex[chosen_idx]!r})",
    )
