"""Compatibility facade for project configuration.

Import from here when you need the traditional ``BASE_DIR`` constant or the
singleton ``settings`` object. The actual merge/validation logic lives in
``app.utils.config_loader``.
"""

from __future__ import annotations

from app.utils.config_loader import BASE_DIR, CONFIG_DIR, ConfigError, DEFAULT_CONFIG_PATH, Settings, load_config


settings = Settings()
