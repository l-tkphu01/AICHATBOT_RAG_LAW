"""Utilities for loading and validating module-based project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "app" / "core" / "config" / "yaml"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"

MODULE_SECTION_KEYS = {
    "system": {"app", "database", "api", "logging"},
    "ingestion": {"pdf_processing", "chunking", "embedding", "vector_store"},
    "query": {"retrieval", "reranker", "query_processing"},
    "validation": {"validation"},
}
SHARED_FRAGMENT_KEYS = {"guardian", "models", "prompts"}
REQUIRED_TOP_LEVEL_SECTIONS = {
    "app",
    "pdf_processing",
    "chunking",
    "embedding",
    "vector_store",
    "retrieval",
    "query_processing",
    "reranker",
}


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or inconsistent."""


def _resolve_config_path(reference: str | Path, base_dir: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_yaml_mapping(file_path: Path, *, label: str) -> dict[str, Any]:
    if not file_path.exists():
        raise ConfigError(f"Missing {label}: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must contain a YAML mapping: {file_path}")

    return payload


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    config_sources = manifest.get("config_sources") or {}
    if not isinstance(config_sources, dict):
        raise ConfigError("config_sources must be a mapping")
    if not config_sources:
        raise ConfigError("config_sources cannot be empty")
    return config_sources


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the manifest and merge all module YAML files into one runtime config."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    manifest_path = Path(config_path).resolve()
    manifest = _load_yaml_mapping(manifest_path, label="config manifest")
    config_sources = _validate_manifest(manifest)

    config: dict[str, Any] = {
        "metadata": manifest.get("metadata", {}),
        "config_sources": dict(config_sources),
    }
    shared_configs: dict[str, dict[str, Any]] = {}
    resolved_config_sources: dict[str, str] = {}

    for source_name, source_ref in config_sources.items():
        source_path = _resolve_config_path(source_ref, manifest_path.parent)
        resolved_config_sources[source_name] = str(source_path)
        payload = _load_yaml_mapping(source_path, label=f"{source_name} config")

        if source_name in SHARED_FRAGMENT_KEYS:
            shared_configs[source_name] = payload
            continue

        expected_sections = MODULE_SECTION_KEYS.get(source_name)
        if expected_sections is None:
            raise ConfigError(f"Unknown config source '{source_name}'")

        payload_keys = set(payload.keys())
        unknown_keys = payload_keys - expected_sections
        missing_keys = expected_sections - payload_keys
        if unknown_keys:
            raise ConfigError(
                f"Unexpected section(s) in '{source_name}' config: {', '.join(sorted(unknown_keys))}"
            )
        if missing_keys:
            raise ConfigError(
                f"Missing section(s) in '{source_name}' config: {', '.join(sorted(missing_keys))}"
            )

        for section_name, section_value in payload.items():
            if section_name in config:
                raise ConfigError(f"Duplicate top-level section '{section_name}'")
            config[section_name] = section_value

    config["shared_configs"] = shared_configs
    config["resolved_config_sources"] = resolved_config_sources

    if "query_processing" in config and isinstance(config["query_processing"], dict):
        config["query_processing"]["config_refs"] = {
            key: resolved_config_sources[key]
            for key in SHARED_FRAGMENT_KEYS
            if key in resolved_config_sources
        }

    missing_sections = [section for section in REQUIRED_TOP_LEVEL_SECTIONS if section not in config]
    if missing_sections:
        raise ConfigError(f"Main config is missing required sections: {', '.join(sorted(missing_sections))}")

    return config


class Settings:
    """Singleton that exposes the merged runtime config and convenience views."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self.config = load_config()
        self.metadata = self.config.get("metadata", {})
        self.config_sources = self.config["config_sources"]
        self.resolved_config_sources = self.config["resolved_config_sources"]
        self.shared_configs = self.config["shared_configs"]
        self.guardian_config = self.shared_configs["guardian"]
        self.models_config = self.shared_configs["models"]
        self.prompts_config = self.shared_configs["prompts"]

        self.app = self.config["app"]
        self.pdf_processing = self.config["pdf_processing"]
        self.chunking = self.config["chunking"]
        self.embedding = self.config["embedding"]
        self.vector_store = self.config["vector_store"]
        self.retrieval = self.config["retrieval"]
        self.query_processing = self.config["query_processing"]
        self.reranker = self.config["reranker"]
        self.validation = self.config["validation"]
        self.database = self.config["database"]
        self.api = self.config["api"]
        self.logging = self.config["logging"]

        self.raw_pdf_dir = BASE_DIR / self.pdf_processing["raw_pdf_dir"]
        self.processed_dir = BASE_DIR / self.pdf_processing["processed_dir"]
        self.vector_store_persist_dir = BASE_DIR / self.vector_store["persist_directory"]
        self.vector_store_collection_name = self.vector_store["collection_name"]
        self.chunk_size = self.chunking["chunk_size"]
        self.chunk_overlap = self.chunking["chunk_overlap"]
        self.min_chunk_size = self.chunking["min_chunk_size"]
        self.include_parent_context = self.chunking["include_parent_context"]
        self.embedding_dimension = self.embedding["dimension"]
        self.embedding_model = self.embedding["model"]
        self.dense_top_k = self.retrieval["dense_top_k"]
        self.sparse_top_k = self.retrieval["sparse_top_k"]
        self.fusion_top_k = self.retrieval["fusion_top_k"]
        self.rerank_top_k = self.retrieval["rerank_top_k"]
        self.relevance_threshold = self.retrieval["relevance_threshold"]
        self.reranker_model = self.reranker["model"]
        
        # Generation config đã chuyển vào models_config hoàn toàn
        # nên Settings không cần gán trực tiếp llm_top_p nữa
        
        self.api_host = self.api["host"]
        self.api_port = self.api["port"]
        self.log_dir = BASE_DIR / self.logging["log_dir"]

        self._validate_declared_refs()

    def resolve_ref(self, reference: str, default: Any | None = None, *, required: bool = False) -> Any:
        """Resolve a dotted reference from config files into runtime data.

        Supported roots:
        - models_config
        - prompts_config
        - guardian_config
        - query_config
        - top-level merged sections (retrieval, reranker, ...)
        """
        if not isinstance(reference, str) or not reference.strip():
            if required:
                raise ConfigError("Config reference must be a non-empty string")
            return default

        parts = [part for part in reference.strip().split(".") if part]
        root = parts[0]

        if root == "models_config":
            current: Any = self.models_config
        elif root == "guardian_config":
            current = self.guardian_config
        elif root == "prompts_config":
            current = self.prompts_config
            # Allow both formats:
            # - prompts_config.some_prompt_key
            # - prompts_config.prompts.some_prompt_key
            if len(parts) >= 2 and parts[1] != "prompts":
                parts = [parts[0], "prompts", *parts[1:]]
        elif root == "query_config":
            current = self.query_processing
        else:
            current = self.config.get(root)

        if current is None:
            if required:
                raise ConfigError(f"Unknown config reference root '{root}' in '{reference}'")
            return default

        for part in parts[1:]:
            if not isinstance(current, dict) or part not in current:
                if required:
                    raise ConfigError(f"Missing config reference '{reference}'")
                return default
            current = current[part]

        return current

    def get_prompt(self, prompt_key: str, default: str = "") -> str:
        """Return a prompt text by key from prompts_config.prompts."""
        prompt_value = self.resolve_ref(f"prompts_config.{prompt_key}", default=default)
        return prompt_value if isinstance(prompt_value, str) else default

    def _validate_model_ref(self, label: str, model_ref: Any) -> None:
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise ConfigError(f"{label} must be a non-empty string")
        resolved = self.resolve_ref(model_ref, required=True)
        if not isinstance(resolved, dict):
            raise ConfigError(f"{label} must point to a model config mapping: '{model_ref}'")

    def _validate_prompt_ref(self, label: str, prompt_ref: Any) -> None:
        if not isinstance(prompt_ref, str) or not prompt_ref.strip():
            raise ConfigError(f"{label} must be a non-empty string")

        if prompt_ref.startswith("prompts_config"):
            resolved = self.resolve_ref(prompt_ref, required=True)
            if not isinstance(resolved, str):
                raise ConfigError(f"{label} must point to prompt text: '{prompt_ref}'")
            return

        prompt_text = self.get_prompt(prompt_ref)
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ConfigError(f"{label} references missing prompt key '{prompt_ref}'")

    def _validate_declared_refs(self) -> None:
        """Fail fast when config-declared model/prompt references are broken."""
        rewrite_cfg = self.query_processing.get("rewrite", {}) if isinstance(self.query_processing, dict) else {}
        flow_cfg = self.query_processing.get("flow", {}) if isinstance(self.query_processing, dict) else {}
        routing_cfg = self.query_processing.get("routing", {}) if isinstance(self.query_processing, dict) else {}
        intent_fallback_cfg = self.guardian_config.get("intent_fallback", {}) if isinstance(self.guardian_config, dict) else {}

        multi_model_ref = rewrite_cfg.get("multi_query_model_ref")
        if isinstance(multi_model_ref, str) and multi_model_ref.strip():
            self._validate_model_ref("query_processing.rewrite.multi_query_model_ref", multi_model_ref)

        legacy_model_ref = rewrite_cfg.get("model_ref")
        if isinstance(legacy_model_ref, str) and legacy_model_ref.strip():
            self._validate_model_ref("query_processing.rewrite.model_ref", legacy_model_ref)

        flow_intent_ref = flow_cfg.get("intent_stage")
        if isinstance(flow_intent_ref, str) and ("." in flow_intent_ref or flow_intent_ref in {"models_config", "guardian_config", "prompts_config", "query_config"}):
            self._validate_model_ref("query_processing.flow.intent_stage", flow_intent_ref)

        rewrite_stage_ref = flow_cfg.get("rewrite_stage")
        if isinstance(rewrite_stage_ref, str) and rewrite_stage_ref.strip():
            self._validate_prompt_ref("query_processing.flow.rewrite_stage", rewrite_stage_ref)

        short_rewrite_stage_ref = flow_cfg.get("short_rewrite_stage")
        if isinstance(short_rewrite_stage_ref, str) and short_rewrite_stage_ref.strip():
            self._validate_prompt_ref("query_processing.flow.short_rewrite_stage", short_rewrite_stage_ref)

        prompt_keys = rewrite_cfg.get("prompt_keys", {}) if isinstance(rewrite_cfg, dict) else {}
        for variant_name, prompt_key in prompt_keys.items():
            self._validate_prompt_ref(f"query_processing.rewrite.prompt_keys.{variant_name}", prompt_key)

        guardian_model_ref = intent_fallback_cfg.get("model_ref")
        if isinstance(guardian_model_ref, str) and guardian_model_ref.strip():
            self._validate_model_ref("guardian_config.intent_fallback.model_ref", guardian_model_ref)

        guardian_prompt_key = intent_fallback_cfg.get("prompt_key")
        if isinstance(guardian_prompt_key, str) and guardian_prompt_key.strip():
            self._validate_prompt_ref("guardian_config.intent_fallback.prompt_key", guardian_prompt_key)

        if isinstance(routing_cfg, dict):
            for route_name, route_cfg in routing_cfg.items():
                if not isinstance(route_cfg, dict):
                    continue
                response_key = route_cfg.get("response_key")
                if isinstance(response_key, str) and response_key.strip():
                    self._validate_prompt_ref(f"query_processing.routing.{route_name}.response_key", response_key)

    def section(self, name: str, default: Any | None = None) -> Any:
        """Return a top-level config section with a safe fallback."""
        return self.config.get(name, default)
