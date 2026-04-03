from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.generation.llm_client import LLMClientError, chat_completion
from app.utils.config_loader import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "app/core/config/yaml/guardian_config.yaml"


def _get_by_path(mapping: dict[str, Any], path: str, default: Any | None = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


# =========================
# LOAD CONFIG
# =========================

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


# =========================
# SIGNAL ENGINE
# =========================

def match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def build_signals(config: dict, text: str) -> dict:
    keyword_filter = config.get("keyword_filter")
    feature_extraction = config.get("feature_extraction", {})
    scope = config.get("scope_rules", {})

    regex_patterns = {}
    if isinstance(keyword_filter, dict):
        regex_patterns = {
            "banned": keyword_filter.get("banned_regex_patterns", []),
            "injection": keyword_filter.get("injection_regex_patterns", []),
            "harmful": keyword_filter.get("harmful_intent_regex_patterns", []),
            "malicious_legal": keyword_filter.get("malicious_legal_intent_regex_patterns", []),
            "out_of_scope": keyword_filter.get("out_of_scope_regex_patterns", []),
            "anchor": keyword_filter.get("legal_anchor_regex_patterns", []),
        }
    elif isinstance(feature_extraction, dict):
        regex_patterns = feature_extraction.get("regex_patterns", {}) if isinstance(feature_extraction.get("regex_patterns", {}), dict) else {}

    if not regex_patterns:
        raise KeyError("keyword_filter")

    def find(patterns):
        hits = []
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                hits.append(p)
        return hits

    def pick_patterns(*keys: str) -> list[str]:
        for key in keys:
            patterns = regex_patterns.get(key)
            if isinstance(patterns, list):
                return patterns
        return []

    banned_hits = find(pick_patterns("banned", "banned_regex_patterns"))
    injection_hits = find(pick_patterns("injection", "injection_regex_patterns"))
    harmful_hits = find(pick_patterns("harmful", "harmful_intent", "harmful_intent_regex_patterns"))
    malicious_hits = find(pick_patterns("malicious", "malicious_legal", "malicious_legal_intent", "malicious_legal_intent_regex_patterns"))
    out_scope_hits = find(pick_patterns("out_of_scope", "out_of_scope_regex_patterns"))
    anchor_hits = find(pick_patterns("legal_anchor", "anchor", "legal_anchor_regex_patterns"))

    short_no_anchor = (
        len(text.split()) < int(scope.get("min_query_tokens_without_anchor", 6))
        and not anchor_hits
    )

    return {
        # raw signals
        "has_safety": bool(banned_hits or injection_hits or harmful_hits or malicious_hits),
        "has_out_scope": bool(out_scope_hits),
        "has_anchor": bool(anchor_hits),
        "short_no_anchor": short_no_anchor,

        # HIT DEBUG (QUAN TRỌNG)
        "hits": {
            "banned": banned_hits,
            "injection": injection_hits,
            "harmful": harmful_hits,
            "malicious_legal": malicious_hits,
            "out_of_scope": out_scope_hits,
            "anchor": anchor_hits,
        },

        # SCORE SYSTEM (0–1 SIMPLE NORMALIZED)
        "scores": {
            "banned_score": 1.0 if banned_hits else 0.0,
            "injection_score": 1.0 if injection_hits else 0.0,
            "harmful_score": 1.0 if harmful_hits else 0.0,
            "malicious_score": 1.0 if malicious_hits else 0.0,
            "out_scope_score": 1.0 if out_scope_hits else 0.0,
            "anchor_score": 1.0 if anchor_hits else 0.0,
        }
    }


# =========================
# FAST / DEEP GUARD
# =========================

def fast_guard(signals: dict) -> str:
    return "FLAG_FAST" if signals["has_safety"] else "PASS_FAST"


def deep_guard(signals: dict) -> str:
    if signals["has_safety"]:
        return "UNSAFE"

    if signals["has_anchor"] and not signals["has_out_scope"]:
        return "SAFE"

    return "UNCERTAIN"


# =========================
# HARD GATE (PRE-MODEL)
# =========================

def hard_gate(config: dict, signals: dict) -> dict:
    hg = config.get("hard_gate", {})
    scope = config.get("scope_rules", {})
    policy = config.get("decision_policy", {})

    if signals["has_safety"] and hg.get("stop_on_safety_signal", True):
        return {
            "stop": True,
            "label": "unsafe",
            "reason": "safety_signal",
        }

    if signals["has_out_scope"] and hg.get("stop_on_out_of_scope_signal", True):
        return {
            "stop": True,
            "label": "out_of_scope",
            "reason": "out_of_scope_signal",
        }

    if (
        signals["short_no_anchor"]
        and bool(scope.get("require_legal_anchor_for_short_queries", True))
    ):
        return {
            "stop": True,
            "label": policy.get("on_missing_legal_anchor", "out_of_scope"),
            "reason": "Hard gate: missing legal anchor",
        }

    return {"stop": False, "label": "pass", "reason": "ok"}


# =========================
# FINAL DECISION ENGINE
# =========================

def final_decision(main_guard: str, intent: str) -> str:
    if main_guard == "SAFE":
        return "safe"

    if main_guard == "UNSAFE":
        return "unsafe"

    if intent in {"unsafe_query"}:
        return "unsafe"

    if intent in {"good_intent"}:
        return "safe"

    return "out_of_scope"


# =========================
# MODEL PARSER
# =========================

def parse_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removesuffix("```")
    return json.loads(text)


# =========================
# MODEL PIPELINE
# =========================

def run_model_guard(config: dict, prompt: str) -> dict:
    settings = Settings()
    model = settings.models_config.get("runtime", {}).get("guard", {})

    res = chat_completion(
        provider=model.get("provider", "groq"),
        model_id=model.get("model_id", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=128,
        response_format={"type": "json_object"},
    )

    return parse_json(res["text"])


def run_intent_fallback(config: dict, text: str, signals: dict) -> dict:
    settings = Settings()
    intent_fallback_cfg = _get_by_path(settings.guardian_config, "intent_fallback", {})
    model_ref = intent_fallback_cfg.get("model_ref", "models_config.runtime.intent_fallback")
    prompt_key = intent_fallback_cfg.get("prompt_key", "intent_fallback_classify_vn")

    intent_config = settings.resolve_ref(str(model_ref), default={})
    prompt_text = settings.get_prompt(str(prompt_key), "{query}")

    if not isinstance(intent_config, dict):
        raise LLMClientError(f"Missing intent fallback model config for ref: {model_ref}")

    prompt = prompt_text.replace("{query}", text)

    res = chat_completion(
        provider=str(intent_config.get("provider", "groq")),
        model_id=str(intent_config.get("model_id", "")),
        messages=[
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=float(intent_config.get("temperature", 0.0)),
        max_tokens=int(intent_config.get("max_tokens", 256)),
        timeout_s=float(intent_config.get("timeout_s", 8)),
        response_format={"type": "json_object"},
    )

    payload = parse_json(res["text"])
    safety_label = str(payload.get("safety_label", "unknown")).lower()
    intent_label = str(payload.get("intent_label", "unknown")).lower()
    confidence = float(payload.get("confidence", 0.0) or 0.0)

    if safety_label == "unsafe" or intent_label == "bad_intent":
        final = "unsafe"
    elif safety_label == "out_of_scope":
        final = "out_of_scope"
    elif safety_label == "safe" and intent_label == "good_intent":
        final = "safe"
    elif confidence < 0.50:
        final = "clarify"
    else:
        final = "clarify" if signals.get("short_no_anchor") or signals.get("has_out_scope") else "safe"

    return {
        "safety_label": safety_label,
        "intent_label": intent_label,
        "confidence": confidence,
        "final": final,
        "raw": payload,
    }


def should_run_intent_fallback(config: dict, signals: dict, hg: dict) -> bool:
    if hg.get("stop"):
        return False

    intent_cfg = config.get("intent_fallback", {})
    if not isinstance(intent_cfg, dict) or not intent_cfg.get("enabled", False):
        return False

    if signals.get("scores", {}).get("malicious_score") == 1.0:
        return True

    if signals.get("short_no_anchor"):
        return True

    if signals.get("has_out_scope"):
        return True

    if signals.get("has_anchor") and signals.get("has_safety") is False:
        return True

    return False


# =========================
# HEURISTIC PIPELINE
# =========================

def heuristic_pipeline(config: dict, text: str) -> dict:
    signals = build_signals(config, text)

    trace = []

    trace.append(build_trace("input", {"text": text}))
    trace.append(build_trace("signals", signals))

    hg = hard_gate(config, signals)
    trace.append(build_trace("hard_gate", hg))

    # STOP EARLY
    if hg["stop"]:
        result = {
            "route": "hard_gate",
            "model_used": False,
            "fast_guard": "FLAG_FAST" if hg["label"] == "unsafe" else "PASS_FAST",
            "deep_guard": "UNSAFE" if hg["label"] == "unsafe" else "UNCERTAIN",
            "main_guard": "unsafe" if hg["label"] == "unsafe" else "uncertain",
            "final": hg["label"],
            "reason": hg["reason"],
            "trace": trace,
        }

        return result

    # FAST
    fast = fast_guard(signals)
    trace.append(build_trace("fast_guard", {"value": fast}))

    # DEEP
    deep = deep_guard(signals)
    trace.append(build_trace("deep_guard", {"value": deep}))

    # MAIN DECISION
    main = (
        "unsafe" if fast == "FLAG_FAST" or deep == "UNSAFE"
        else "safe" if deep == "SAFE"
        else "uncertain"
    )

    trace.append(build_trace("main_guard", {"value": main}))

    intent = "good_intent" if signals["has_anchor"] else "safe_query"

    result = {
        "route": "heuristic",
        "model_used": False,

        "fast_guard": fast,
        "deep_guard": deep,
        "main_guard": main,
        "intent": intent,
        "final": final_decision(main.upper(), intent),

        # DEBUG OUTPUT
        "trace": trace,
    }

    return result


# =========================
# MAIN ENTRY
# =========================

def run_guardian_pipeline(text: str, mode: str = "heuristic") -> dict:
    config = load_config()
    signals = build_signals(config, text)
    trace = []
    trace.append({"stage": "input", "text": text})
    trace.append({"stage": "signals", "signals": signals})
    hg = hard_gate(config, signals)
    trace.append({"stage": "hard_gate", "result": hg})

    # HARD STOP
    if hg["stop"]:
        return {
            "route": "hard_gate",
            "model_used": False,
            "gate_decision": "STOP",
            "gate_label": hg["label"],
            "deep_guard": "UNSAFE" if hg["label"] == "unsafe" else "UNCERTAIN",
            "main_guard": "unsafe" if hg["label"] == "unsafe" else "uncertain",
            "final": hg["label"],
            "final_decision": hg["label"],
            "reason": hg["reason"],
            "trace": trace,
        }

    intent_result: dict[str, Any] | None = None

    # MODEL MODE
    if mode == "real":
        prompt = f"""
Return JSON:
{{"safety":"SAFE|UNSAFE","intent":"good_intent|bad_intent|unknown","confidence":0.0}}
TEXT:
{text}
"""
        model_data = run_model_guard(config, prompt)
        trace.append({
            "stage": "model",
            "model_output": model_data
        })

        if should_run_intent_fallback(config, signals, hg):
            try:
                intent_result = run_intent_fallback(config, text, signals)
                trace.append({
                    "stage": "intent_fallback",
                    "intent_output": intent_result,
                })
            except LLMClientError as exc:
                trace.append({
                    "stage": "intent_fallback_error",
                    "error": str(exc),
                })

        final_label = model_data.get("safety", "SAFE").lower()
        intent_label = model_data.get("intent", "unknown")

        if intent_result is not None:
            final_label = intent_result["final"]
            intent_label = intent_result["intent_label"]

        return {
            "route": "model",
            "model_used": True,
            "gate_decision": "PASS",
            "gate_label": "pass",
            "fast_guard": "MODEL_DRIVEN",
            "deep_guard": model_data.get("safety"),
            "intent": intent_label,
            "confidence": intent_result["confidence"] if intent_result is not None else model_data.get("confidence"),
            "intent_fallback": intent_result or {"final": "not_used"},
            "final": final_label,
            "final_decision": final_label,
            # FULL DEBUG
            "trace": trace,
        }

    # FALLBACK HEURISTIC
    result = heuristic_pipeline(config, text)

    return result


def simulate_guardian_pipeline(text: str) -> dict:
    """Compatibility wrapper used by tests to expose the legacy simulation schema."""
    config = load_config()
    signals = build_signals(config, text)

    fast = fast_guard(signals)
    if signals["has_safety"]:
        deep = "UNSAFE_CONTEXT"
        main = "unsafe"
        final = "unsafe"
    elif any(token in text.lower() for token in ["hợp đồng", "doanh nghiệp", "luật", "điều ", "khoản "]):
        deep = "SAFE_CONTEXT"
        main = "safe"
        final = "safe"
    elif signals["has_anchor"] and not signals["has_out_scope"]:
        deep = "SAFE_CONTEXT"
        main = "safe"
        final = "safe"
    else:
        deep = "UNCERTAIN"
        main = "uncertain"
        final = "out_of_scope"

    intent_fallback = "not_used"
    if signals["has_safety"]:
        intent_fallback = "unsafe_query"
    elif signals["has_anchor"]:
        intent_fallback = "good_intent"
    elif signals["has_out_scope"]:
        intent_fallback = "bad_intent"
    elif signals["short_no_anchor"]:
        intent_fallback = "safe_query"

    return {
        "fast_guard": fast,
        "deep_guard": deep,
        "main_guard": main,
        "intent_fallback": intent_fallback,
        "final_decision": final,
        "reason": "ok",
        "route": "heuristic",
        "model_used": False,
        "trace": [],
        "intent": intent_fallback,
    }


def build_trace(stage: str, data: dict) -> dict:
    return {
        "stage": stage,
        "data": data
    }