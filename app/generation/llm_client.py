"""Minimal LLM client helpers for OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import httpx

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
	load_dotenv = None


if load_dotenv is not None:
	load_dotenv()

# Singleton HTTP Client to prevent connection leaks
_http_client = None

def get_http_client() -> httpx.Client:
	global _http_client
	if _http_client is None:
		_http_client = httpx.Client(timeout=30.0)
	return _http_client


from app.utils.config_loader import Settings

class LLMClientError(RuntimeError):
	"""Raised when a provider call fails or returns invalid data."""


def _groq_api_key() -> str:
	api_key = os.getenv("GROQ_API_KEY", "").strip()
	if not api_key:
		raise LLMClientError("Missing GROQ_API_KEY environment variable.")
	return api_key

def _openrouter_api_key() -> str:
	api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
	if not api_key:
		raise LLMClientError("Missing OPENROUTER_API_KEY environment variable.")
	return api_key


def _extract_text(payload: dict[str, Any]) -> str:
	choices = payload.get("choices") or []
	if not choices:
		raise LLMClientError(f"API response did not include any choices. Raw payload: {payload}")

	message = choices[0].get("message") or {}
	content = message.get("content")
	
	# OpenRouter reasoning models có thể trả content rỗng nhưng lại nằm trong reasoning_details
	reasoning = message.get("reasoning_details")

	# Nếu có content dạng string
	if isinstance(content, str) and content.strip():
		return content

	# Nếu có content dạng list
	if isinstance(content, list):
		parts: list[str] = []
		for item in content:
			if isinstance(item, dict) and item.get("type") == "text":
				parts.append(str(item.get("text", "")))
		text = "".join(parts).strip()
		if text:
			return text

	# Fallback lấy nội dung logic (reasoning) nếu content thật sự rỗng
	if reasoning and isinstance(reasoning, str):
		return reasoning

	raise LLMClientError(f"API response did not include textual content. Raw payload: {payload}")


def _extract_usage(payload: dict[str, Any]) -> dict[str, Any]:
	usage = payload.get("usage") or {}
	if not isinstance(usage, dict):
		return {}
	return {
		"prompt_tokens": usage.get("prompt_tokens"),
		"completion_tokens": usage.get("completion_tokens"),
		"total_tokens": usage.get("total_tokens"),
	}


def chat_completion(
	*,
	provider: str,
	model_id: str,
	messages: list[dict[str, Any]],
	temperature: float = 0.0,
	max_tokens: int | None = None,
	timeout_s: float = 30.0,
	response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
	"""Send a chat completion request to an OpenAI-compatible provider."""
	provider_normalized = provider.strip().lower()
	if provider_normalized not in ("groq", "openrouter"):
		raise LLMClientError(f"Unsupported provider: {provider}")

	request_body: dict[str, Any] = {
		"model": model_id,
		"messages": messages,
		"temperature": temperature,
	}
	if max_tokens is not None:
		request_body["max_tokens"] = max_tokens
	if response_format is not None:
		request_body["response_format"] = response_format

	settings = Settings()
	models_cfg = settings.models_config

	if provider_normalized == "groq":
		api_url = models_cfg.get("groq", {}).get("api_url", "https://api.groq.com/openai/v1/chat/completions")
		headers = {
			"Authorization": f"Bearer {_groq_api_key()}",
			"Content-Type": "application/json",
		}
	elif provider_normalized == "openrouter":
		api_url = models_cfg.get("openrouter", {}).get("api_url", "https://openrouter.ai/api/v1/chat/completions")
		headers = {
			"Authorization": f"Bearer {_openrouter_api_key()}",
			"Content-Type": "application/json",
			"HTTP-Referer": "https://localhost", 
			"X-Title": "AILawBot",
		}
		request_body.pop("response_format", None)
		# Bật cờ reasoning theo Document của OpenRouter
		request_body["reasoning"] = {"enabled": True}

	start_time = time.time()
	try:
		# Use the singleton client instead of httpx.post directly to avoid socket leaks
		client = get_http_client()
		response = client.post(
			api_url,
			headers=headers,
			json=request_body,
			timeout=timeout_s,
		)
		response.raise_for_status()
	except httpx.HTTPError as exc:  # pragma: no cover - network/client dependent
		error_detail = exc.response.text if hasattr(exc, 'response') and exc.response else ""
		raise LLMClientError(f"{provider_normalized.capitalize()} request failed: {exc}\nDetail: {error_detail}") from exc

	try:
		payload = response.json()
	except json.JSONDecodeError as exc:  # pragma: no cover - network/client dependent
		raise LLMClientError(f"{provider_normalized.capitalize()} response was not valid JSON.") from exc

	latency_s = time.time() - start_time

	return {
		"text": _extract_text(payload),
		"usage": _extract_usage(payload),
		"raw": payload,
		"latency_s": latency_s,
	}
