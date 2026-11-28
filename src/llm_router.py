"""Hardened Multi-provider LLM router (Gemini, OpenAI, Anthropic, OpenRouter, Ollama).

Includes:
- One-time .env loading and thread-safe genai.Client connection caching
- Per-model Circuit Breaker (cooldown on 429/503) + gentle pacing to prevent RPM storms
- OpenAI reasoning model (o1/o3/o4) temperature handling
- Anthropic 8192 max_tokens for 5-question batch JSON
- Provider-aware concurrency recommendations (Ollama = 1 worker, Cloud = 3 workers)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
T = TypeVar("T", bound=BaseModel)


class JobCancelledError(RuntimeError):
    """Raised immediately when the user cancels an active exam generation job."""


def check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("Job cancelled by user")

# Runtime overrides set from the dashboard UI (kept in-process; never logged)
RUNTIME_API_KEYS: dict[str, str] = {}

_ENV_LOADED = False
_ENV_LOCK = threading.Lock()
_CLIENT_CACHE: dict[tuple[str, int], genai.Client] = {}
_CLIENT_LOCK = threading.Lock()
_COOLDOWN_LOCK = threading.Lock()
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}
_LAST_DISPATCH_TS = 0.0
_PACE_LOCK = threading.Lock()
_GEMINI_PACE_SECONDS = 1.0
_NO_THINKING_CONFIG_MODELS: set[str] = set()

_QUOTA_LOCK = threading.Lock()
_QUOTA_FILE = PROJECT_ROOT / "data" / "reports" / "quota_usage.json"
_RPM_TIMESTAMPS: dict[str, list[float]] = {}
_QUOTA_STATE: dict[str, dict] = {}
_QUOTA_LOADED_DATE: str = ""
_ACTIVITY_LOCK = threading.Lock()
_CURRENT_ACTIVITY: dict[str, object] = {
    "state": "idle",
    "text": "พร้อมใช้งาน",
    "detail": "เลือกโหมดด้านล่างเพื่อเริ่มสร้างข้อสอบ",
    "recent": [],
    "updated_at": 0.0,
}


def set_ai_activity(state: str, text: str, detail: str = "") -> None:
    """Update real-time AI activity + rolling 3-step event log shown in the top-right HUD."""
    now = time.time()
    stamp = time.strftime("%H:%M:%S")
    with _ACTIVITY_LOCK:
        _CURRENT_ACTIVITY["state"] = state
        _CURRENT_ACTIVITY["text"] = text
        _CURRENT_ACTIVITY["detail"] = detail
        _CURRENT_ACTIVITY["updated_at"] = now
        recent = list(_CURRENT_ACTIVITY.get("recent", []))
        if state == "idle":
            if detail:
                recent.append(f"{stamp} • {detail}")
                _CURRENT_ACTIVITY["recent"] = recent[-3:]
        else:
            line = f"{stamp} • {text}"
            if not recent or not recent[-1].endswith(text):
                recent.append(line)
                _CURRENT_ACTIVITY["recent"] = recent[-3:]


def get_ai_activity() -> dict:
    with _ACTIVITY_LOCK:
        st = dict(_CURRENT_ACTIVITY)
    # Auto-return to idle if stale (> 120s without updates)
    if st["state"] != "idle" and (time.time() - float(st.get("updated_at", 0.0))) > 120.0:
        return {
            "state": "idle",
            "text": "พร้อมใช้งาน",
            "detail": "เลือกโหมดด้านล่างเพื่อเริ่มสร้างข้อสอบ",
            "recent": list(st.get("recent", []))[-3:],
        }
    return {
        "state": st["state"],
        "text": st["text"],
        "detail": st.get("detail", ""),
        "recent": list(st.get("recent", []))[-3:],
    }

GEMINI_FREE_QUOTA_DEFAULTS: dict[str, dict[str, int]] = {
    "gemini-3.5-flash-lite": {"rpd": 500, "rpm": 15},
    "gemini-3.1-flash-lite": {"rpd": 500, "rpm": 15},
    "gemini-4-hydrogen": {"rpd": 20, "rpm": 5},
    "gemini-4-neon": {"rpd": 20, "rpm": 5},
    "gemini-3.8-flash": {"rpd": 20, "rpm": 10},
    "gemini-3.6-flash": {"rpd": 20, "rpm": 10},
}


def _today_key() -> str:
    return time.strftime("%Y-%m-%d")


def _load_quota_state_unlocked() -> None:
    global _QUOTA_STATE, _QUOTA_LOADED_DATE
    today = _today_key()
    if _QUOTA_LOADED_DATE == today:
        return
    _QUOTA_LOADED_DATE = today
    _QUOTA_STATE = {}
    try:
        if _QUOTA_FILE.exists():
            raw = json.loads(_QUOTA_FILE.read_text(encoding="utf-8"))
            if raw.get("date") == today and isinstance(raw.get("models"), dict):
                _QUOTA_STATE = raw["models"]
    except Exception:
        _QUOTA_STATE = {}


def _save_quota_state_unlocked() -> None:
    try:
        _QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"date": _QUOTA_LOADED_DATE or _today_key(), "models": _QUOTA_STATE}
        _QUOTA_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _parse_duration_seconds(msg: str) -> float | None:
    """Parse durations like '2h40m58.6s', '1m30s', or '22.5s' from Gemini 429 text."""
    m = re.search(
        r"retry\s+in\s+(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s",
        msg,
        re.IGNORECASE,
    )
    if m:
        h = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        sec = float(m.group(3) or 0.0)
        total = h * 3600 + mins * 60 + sec
        if total > 0:
            return total
    m2 = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None


def _parse_server_limit(msg: str) -> int | None:
    m = re.search(r"limit:\s*(\d+)", msg, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return val if val > 0 else None
    return None


def record_request_usage(provider: str, model: str) -> None:
    """Record a dispatched LLM call for real-time RPM/RPD quota tracking."""
    key = f"{provider}:{model}"
    now = time.time()
    with _QUOTA_LOCK:
        _load_quota_state_unlocked()
        ts_list = [t for t in _RPM_TIMESTAMPS.get(key, []) if now - t < 60.0]
        ts_list.append(now)
        _RPM_TIMESTAMPS[key] = ts_list

        entry = _QUOTA_STATE.setdefault(key, {"rpd_used": 0})
        defaults = GEMINI_FREE_QUOTA_DEFAULTS.get(model, {"rpd": 20, "rpm": 10})
        rpd_limit = int(entry.get("rpd_limit") or defaults["rpd"])
        # If a call succeeded after a previous exhaustion, the provider has reset quota.
        if entry.get("exhausted_kind") == "rpd" or int(entry.get("rpd_used", 0)) >= rpd_limit:
            entry["rpd_used"] = 1
        else:
            entry["rpd_used"] = int(entry.get("rpd_used", 0)) + 1
        entry["exhausted_kind"] = None
        entry["reset_at"] = 0.0
        entry["last_used_at"] = now
        _save_quota_state_unlocked()


def record_quota_exhausted(provider: str, model: str, msg: str, is_daily: bool) -> None:
    """Auto-calibrate quota gauge immediately when a 429 response arrives."""
    key = f"{provider}:{model}"
    now = time.time()
    server_limit = _parse_server_limit(msg)
    reset_s = _parse_duration_seconds(msg)
    with _QUOTA_LOCK:
        _load_quota_state_unlocked()
        defaults = GEMINI_FREE_QUOTA_DEFAULTS.get(model, {"rpd": 20, "rpm": 10})
        entry = _QUOTA_STATE.setdefault(key, {"rpd_used": 0})
        if is_daily:
            eff_limit = server_limit or int(entry.get("rpd_limit") or defaults["rpd"])
            entry["rpd_limit"] = eff_limit
            entry["rpd_used"] = max(int(entry.get("rpd_used", 0)), eff_limit)
            entry["exhausted_kind"] = "rpd"
            entry["reset_at"] = now + (reset_s if reset_s and reset_s > 60 else 3600.0)
        else:
            entry["exhausted_kind"] = "rpm"
            entry["reset_at"] = now + (reset_s if reset_s else 60.0)
        entry["last_429_at"] = now
        _save_quota_state_unlocked()


def get_quota_status(provider: str, model: str) -> dict:
    """Return real-time quota gauge data for the selected (provider, model)."""
    key = f"{provider}:{model}"
    now = time.time()
    if provider == "ollama":
        return {
            "provider": provider,
            "model": model,
            "unlimited": True,
            "status": "unlimited",
            "rpd_used": 0,
            "rpd_limit": None,
            "rpd_remaining": None,
            "rpm_used": 0,
            "rpm_limit": None,
            "pct": 100,
            "reset_in_seconds": 0,
            "message_th": "Local Ollama • ใช้งานฟรีไม่จำกัด",
            "activity": get_ai_activity(),
        }

    with _QUOTA_LOCK:
        _load_quota_state_unlocked()
        raw_entry = _QUOTA_STATE.get(key)
        if raw_entry:
            raw_reset_at = float(raw_entry.get("reset_at", 0.0) or 0.0)
            raw_kind = raw_entry.get("exhausted_kind")
            if raw_kind and raw_reset_at > 0 and now >= raw_reset_at:
                # Reset window has elapsed -> clear exhaustion and restore daily counter if RPD reset
                if raw_kind == "rpd":
                    raw_entry["rpd_used"] = 0
                raw_entry["exhausted_kind"] = None
                raw_entry["reset_at"] = 0.0
                _save_quota_state_unlocked()
        entry = dict(raw_entry or {})
        rpm_ts = [t for t in _RPM_TIMESTAMPS.get(key, []) if now - t < 60.0]
        _RPM_TIMESTAMPS[key] = rpm_ts
        rpm_used = len(rpm_ts)

    if provider != "gemini":
        return {
            "provider": provider,
            "model": model,
            "unlimited": True,
            "status": "ok",
            "rpd_used": int(entry.get("rpd_used", 0)),
            "rpd_limit": None,
            "rpd_remaining": None,
            "rpm_used": rpm_used,
            "rpm_limit": None,
            "pct": 100,
            "reset_in_seconds": 0,
            "message_th": f"ใช้งานแล้ววันนี้ {int(entry.get('rpd_used', 0))} ครั้ง (API แบบจ่ายตามจริง)",
            "activity": get_ai_activity(),
        }

    defaults = GEMINI_FREE_QUOTA_DEFAULTS.get(model, {"rpd": 20, "rpm": 10})
    rpd_limit = int(entry.get("rpd_limit") or defaults["rpd"])
    rpm_limit = int(defaults["rpm"])
    rpd_used = int(entry.get("rpd_used", 0))
    reset_at = float(entry.get("reset_at", 0.0) or 0.0)
    exhausted_kind = entry.get("exhausted_kind")
    reset_in = max(0, int(round(reset_at - now)))

    if exhausted_kind == "rpd" and reset_in > 0:
        rpd_remaining = 0
        pct = 0
        status = "exhausted_rpd"
        msg_th = f"โควตารายวันของ {model} หมดแล้ว"
    elif exhausted_kind == "rpm" and reset_in > 0:
        rpd_remaining = max(0, rpd_limit - rpd_used)
        pct = max(0, min(100, int(round((rpd_remaining / max(1, rpd_limit)) * 100))))
        status = "limited_rpm"
        msg_th = f"ติดลิมิตรายนาทีของ {model} ชั่วคราว"
    else:
        rpd_remaining = max(0, rpd_limit - rpd_used)
        pct = max(0, min(100, int(round((rpd_remaining / max(1, rpd_limit)) * 100))))
        if rpd_remaining == 0:
            status = "exhausted_rpd"
            msg_th = f"โควตารายวันของ {model} หมดแล้ว"
        elif pct <= 25:
            status = "low"
            msg_th = f"โควตาเหลือ {rpd_remaining}/{rpd_limit} ครั้ง (ใกล้หมด)"
        else:
            status = "ok"
            msg_th = f"โควตาเหลือ {rpd_remaining}/{rpd_limit} ครั้ง"

    return {
        "provider": provider,
        "model": model,
        "unlimited": False,
        "status": status,
        "rpd_used": rpd_used,
        "rpd_limit": rpd_limit,
        "rpd_remaining": rpd_remaining,
        "rpm_used": rpm_used,
        "rpm_limit": rpm_limit,
        "pct": pct,
        "reset_in_seconds": reset_in,
        "message_th": msg_th,
        "activity": get_ai_activity(),
    }


def _parse_retry_after_seconds(msg: str) -> float | None:
    """Extract server-suggested retry delay (in seconds) from Gemini 429 error text."""
    # If server asks to wait hours (e.g. "retry in 2h40m58s"), it's a daily quota reset — don't sleep!
    if re.search(r"retry\s+in\s+\d+h", msg, re.IGNORECASE):
        return None
    m = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if not m:
        m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m:
        return max(4.0, min(35.0, float(m.group(1)) + 1.0))
    return None


def _is_daily_quota_exhausted(msg: str) -> bool:
    """Detect non-recoverable Free Tier daily quota (RPD) or zero-limit exhaustion."""
    low = msg.lower()
    if re.search(r"retry\s+in\s+\d+h", low):
        return True
    return (
        "per day" in low
        or "perday" in low
        or "requests per day" in low
        or "requestsperday" in low
        or ("quota exceeded" in low and "limit: 0" in low)
    )

PROVIDER_CATALOG: dict[str, dict] = {
    "gemini": {
        "label": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "models": [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-4-hydrogen",
            "gemini-4-neon",
            "gemini-3.8-flash",
            "gemini-3.6-flash",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4.1", "gpt-4.1-mini", "o4-mini"],
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "models": [
            "google/gemini-2.5-flash",
            "openai/gpt-4.1-mini",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-r1",
        ],
    },
    "ollama": {
        "label": "Local Ollama",
        "env_var": "OLLAMA_BASE_URL",
        "models": ["qwen3:8b", "qwen2.5:7b", "llama3.1:8b"],
    },
}


def recommend_workers(provider: str) -> int:
    """Balance concurrency against provider rate limits (Ollama=1, Gemini Free-friendly=2, Paid Cloud=4)."""
    if provider == "ollama":
        return 1
    if provider == "gemini":
        return 2
    return 4


def recommend_batch_size(provider: str) -> int:
    """Local 4B-8B models handle 1 question at a time far more reliably than 5-question JSON batches."""
    return 1 if provider == "ollama" else 5


def _load_env_files() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED:
            return
        load_dotenv(PROJECT_ROOT / ".env")
        _ENV_LOADED = True


def get_api_key(provider: str) -> str | None:
    _load_env_files()
    if provider in RUNTIME_API_KEYS and RUNTIME_API_KEYS[provider].strip():
        return RUNTIME_API_KEYS[provider].strip()
    spec = PROVIDER_CATALOG.get(provider)
    if not spec:
        return None
    val = os.getenv(spec["env_var"])
    return val.strip() if val else None


def set_runtime_api_key(provider: str, key: str) -> None:
    if provider not in PROVIDER_CATALOG:
        raise ValueError(f"Unsupported provider: {provider}")
    if key.strip():
        RUNTIME_API_KEYS[provider] = key.strip()
    else:
        RUNTIME_API_KEYS.pop(provider, None)


def _get_gemini_client(api_key: str, timeout_ms: int) -> genai.Client:
    cache_key = (api_key, timeout_ms)
    with _CLIENT_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )
            _CLIENT_CACHE[cache_key] = client
        return client


def mark_model_cooldown(model_key: str, seconds: float) -> None:
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWN_UNTIL[model_key] = max(
            _MODEL_COOLDOWN_UNTIL.get(model_key, 0.0),
            time.monotonic() + seconds,
        )


def is_model_in_cooldown(model_key: str) -> bool:
    with _COOLDOWN_LOCK:
        return time.monotonic() < _MODEL_COOLDOWN_UNTIL.get(model_key, 0.0)


def clear_cooldowns() -> None:
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWN_UNTIL.clear()


def _pace_dispatch(min_interval_seconds: float = 0.25) -> None:
    global _LAST_DISPATCH_TS
    with _PACE_LOCK:
        now = time.monotonic()
        wait = min_interval_seconds - (now - _LAST_DISPATCH_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_DISPATCH_TS = time.monotonic()


def _discover_ollama_models(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        if resp.ok:
            models = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
            if models:
                return models
    except Exception:
        pass
    return []


def list_provider_status() -> list[dict]:
    """Return all providers, whether their API key is configured, and selectable models."""
    _load_env_files()
    result: list[dict] = []
    for pid, spec in PROVIDER_CATALOG.items():
        if pid == "ollama":
            base_url = get_api_key("ollama") or "http://localhost:11434"
            live_models = _discover_ollama_models(base_url)
            configured = bool(live_models) or bool(get_api_key("ollama"))
            models = live_models or spec["models"]
        else:
            configured = bool(get_api_key(pid))
            models = list(spec["models"])

        result.append(
            {
                "id": pid,
                "label": spec["label"],
                "env_var": spec["env_var"],
                "configured": configured,
                "models": models,
            }
        )
    return result


def _extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def build_openai_payload(model: str, json_prompt: str, temperature: float) -> dict:
    """Build OpenAI/OpenRouter payload, omitting temperature for reasoning models (o1/o3/o4)."""
    payload: dict = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": json_prompt}],
    }
    base_name = model.split("/")[-1].lower()
    if not base_name.startswith(("o1", "o3", "o4")):
        payload["temperature"] = temperature
    return payload


def _post_with_retry(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: float,
    max_retries: int = 2,
    cancel_event: threading.Event | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        check_cancelled(cancel_event)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait_s = 1.5 * (2**attempt)
                if cancel_event is not None:
                    if cancel_event.wait(wait_s):
                        raise JobCancelledError("Job cancelled by user")
                else:
                    time.sleep(wait_s)
                continue
            resp.raise_for_status()
            return resp
        except JobCancelledError:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait_s = 1.5 * (2**attempt)
                if cancel_event is not None:
                    if cancel_event.wait(wait_s):
                        raise JobCancelledError("Job cancelled by user")
                else:
                    time.sleep(wait_s)
                continue
            raise
    raise RuntimeError(f"HTTP request failed: {last_exc}")


def generate_structured_json(
    provider: str,
    model: str,
    prompt: str,
    schema_cls: Type[T],
    temperature: float = 0.35,
    timeout_seconds: float = 65.0,
    thinking_budget: int = 128,
    cancel_event: threading.Event | None = None,
) -> T:
    """Generate and validate a Pydantic-structured response from any configured provider."""
    check_cancelled(cancel_event)
    if provider == "gemini":
        api_key = get_api_key("gemini")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = _get_gemini_client(api_key, int(timeout_seconds * 1000))
        transient_tokens = ("503", "504", "500", "429", "DEADLINE", "UNAVAILABLE", "timed out", "timeout", "ReadTimeout")
        last_err = ""

        # Strictly call ONLY the single model selected by the user (no cross-model hopping).
        # Free-Tier friendly: apply thinking_budget to ALL Gemini models (including hydrogen/neon),
        # adaptive pacing + Retry-After backoff for RPM/TPM 429s, and instant fast-fail on daily RPD exhaustion.
        global _GEMINI_PACE_SECONDS
        for attempt in range(3):
            check_cancelled(cancel_event)
            _pace_dispatch(_GEMINI_PACE_SECONDS)
            check_cancelled(cancel_event)
            set_ai_activity(
                "working",
                f"ส่งคำขอไปที่ Gemini ({model})",
                f"รอโมเดลสร้างโจทย์ + โค้ด SymPy (รอบส่งที่ {attempt + 1}/3)",
            )
            try:
                cfg_kwargs: dict = {
                    "response_mime_type": "application/json",
                    "response_schema": schema_cls,
                    "temperature": temperature,
                }
                if model not in _NO_THINKING_CONFIG_MODELS:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                except Exception as first_exc:
                    if "INVALID_ARGUMENT" in str(first_exc) and "thinking_config" in cfg_kwargs:
                        _NO_THINKING_CONFIG_MODELS.add(model)
                        cfg_kwargs.pop("thinking_config", None)
                        resp = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=types.GenerateContentConfig(**cfg_kwargs),
                        )
                    else:
                        raise
                check_cancelled(cancel_event)
                record_request_usage("gemini", model)
                return schema_cls.model_validate_json(resp.text)
            except JobCancelledError:
                raise
            except Exception as exc:
                msg = str(exc)
                if any(tok in msg for tok in transient_tokens):
                    last_err = msg.splitlines()[0][:180]
                    if "429" in msg and _is_daily_quota_exhausted(msg):
                        record_quota_exhausted("gemini", model, msg, is_daily=True)
                        mark_model_cooldown(f"gemini:{model}", 300.0)
                        raise RuntimeError(
                            f"โควตารายวัน (RPD) ของรุ่น '{model}' หมดแล้ว — "
                            f"สลับไปใช้รุ่น Gemini อื่นในเมนูด้านบน (เช่น gemini-3.5-flash-lite) "
                            f"หรือใช้ Local Ollama ฟรีในเครื่องได้เลย"
                        ) from exc
                    if attempt < 2:
                        if "429" in msg:
                            record_quota_exhausted("gemini", model, msg, is_daily=False)
                            _GEMINI_PACE_SECONDS = max(_GEMINI_PACE_SECONDS, 4.5)
                            wait_s = _parse_retry_after_seconds(msg) or (10.0 * (attempt + 1))
                        else:
                            wait_s = 2.5 * (attempt + 1)
                        set_ai_activity("waiting", f"ติดลิมิตชั่วคราว • รอ {wait_s:.0f} วิ. แล้วลองใหม่...")
                        print(
                            f"[llm-router] Gemini '{model}' hit transient limit (attempt {attempt + 1}/3), "
                            f"waiting {wait_s:.1f}s before retry...",
                            flush=True,
                        )
                        if cancel_event is not None:
                            if cancel_event.wait(wait_s):
                                raise JobCancelledError("Job cancelled by user")
                        else:
                            time.sleep(wait_s)
                        continue
                    if "429" in msg:
                        record_quota_exhausted("gemini", model, msg, is_daily=False)
                    mark_model_cooldown(f"gemini:{model}", 60.0 if "429" in msg else 15.0)
                    raise RuntimeError(f"Model '{model}' unavailable: {last_err}") from exc
                raise
        raise RuntimeError(f"Model '{model}' unavailable: {last_err}")

    schema_json = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False)
    json_prompt = (
        f"{prompt}\n\n"
        f"IMPORTANT: Respond ONLY with a valid JSON object strictly matching this JSON Schema:\n"
        f"{schema_json}"
    )

    if provider in ("openai", "openrouter"):
        api_key = get_api_key(provider)
        if not api_key:
            raise RuntimeError(f"{PROVIDER_CATALOG[provider]['env_var']} is not configured.")
        url = (
            "https://api.openai.com/v1/chat/completions"
            if provider == "openai"
            else "https://openrouter.ai/api/v1/chat/completions"
        )
        resp = _post_with_retry(
            url=url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=build_openai_payload(model, json_prompt, temperature),
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        return schema_cls.model_validate_json(_extract_json_block(content))

    if provider == "anthropic":
        api_key = get_api_key("anthropic")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        resp = _post_with_retry(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": model,
                "max_tokens": 8192,
                "temperature": temperature,
                "messages": [{"role": "user", "content": json_prompt}],
            },
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
        content = resp.json()["content"][0]["text"]
        return schema_cls.model_validate_json(_extract_json_block(content))

    if provider == "ollama":
        base_url = (get_api_key("ollama") or "http://localhost:11434").rstrip("/")
        payload: dict = {
            "model": model,
            "stream": True,
            "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": temperature, "num_predict": 1400},
            "messages": [{"role": "user", "content": json_prompt}],
        }
        if any(k in model.lower() for k in ("qwen3", "deepseek", "glm", "r1")):
            payload["think"] = False

        set_ai_activity(
            "working",
            f"ส่งคำขอไปที่ Local Ollama ({model})",
            "กำลังประมวลผลบนเครื่อง (สตรีม JSON + โค้ด SymPy)...",
        )

        def _stream_ollama(req_payload: dict) -> tuple[int, str]:
            check_cancelled(cancel_event)
            with requests.post(
                f"{base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json=req_payload,
                timeout=(10.0, 60.0),
                stream=True,
            ) as resp:
                if not resp.ok:
                    return resp.status_code, resp.text[:300]
                chunks: list[str] = []
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if cancel_event is not None and cancel_event.is_set():
                        resp.close()
                        raise JobCancelledError("Job cancelled by user")
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except Exception:
                        continue
                    msg_part = (obj.get("message") or {}).get("content", "")
                    if msg_part:
                        chunks.append(msg_part)
                    if obj.get("done"):
                        break
                return 200, "".join(chunks)

        status_code, content = _stream_ollama(payload)
        if status_code >= 400 and "think" in payload:
            payload.pop("think", None)
            status_code, content = _stream_ollama(payload)
        if status_code >= 400:
            raise RuntimeError(f"Ollama error ({status_code}): {content}")
        check_cancelled(cancel_event)
        return schema_cls.model_validate_json(_extract_json_block(content))

    raise ValueError(f"Unsupported provider: {provider}")
