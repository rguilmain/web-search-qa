"""Application configuration.

Non-secret defaults live here and can be overridden with environment variables
or a local ``.env`` file. Secrets are wrapped in ``SecretStr`` so they are not
printed accidentally.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when the application cannot run with the current configuration."""


class Settings(BaseSettings):
    """Runtime settings loaded from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenRouter
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "deepseek/deepseek-v4.1-flash"
    request_timeout: float = Field(default=60.0, ge=5.0, le=300.0)

    # Agent behaviour
    default_max_steps: int = Field(default=15, ge=1, le=50)
    max_history_turns: int = Field(default=8, ge=0, le=50)
    max_observation_chars: int = Field(default=20_000, ge=1_000, le=100_000)

    # Wikipedia / Wikimedia
    wikipedia_api_key: SecretStr | None = None
    wikipedia_language: str = "en"
    wikipedia_user_agent: str = (
        "web-search-qa/0.1 (local research agent; contact: richguilmain@gmail.com) bot"
    )
    wikipedia_page_chars: int = Field(default=12_000, ge=1_000, le=50_000)
    wikipedia_max_links: int = Field(default=40, ge=0, le=200)

    # Web search
    search_provider: Literal["ddgs", "tavily", "brave"] = "ddgs"
    search_results: int = Field(default=5, ge=1, le=10)
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    # Page fetching
    fetch_user_agent: str = (
        "web-search-qa/0.1 (local research agent; contact: richguilmain@gmail.com) bot"
    )
    fetch_max_bytes: int = Field(default=5_000_000, ge=10_000, le=25_000_000)
    fetch_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    fetch_page_chars: int = Field(default=12_000, ge=1_000, le=50_000)
    fetch_max_redirects: int = Field(default=5, ge=0, le=10)
    fetch_max_links: int = Field(default=30, ge=0, le=100)

    @field_validator("search_provider", mode="before")
    @classmethod
    def _lowercase_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator(
        "wikipedia_language",
        "wikipedia_user_agent",
        "fetch_user_agent",
        "default_model",
        "openrouter_base_url",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @property
    def has_openrouter_key(self) -> bool:
        return bool(self._secret(self.openrouter_api_key))

    def require_openrouter_key(self) -> str:
        key = self._secret(self.openrouter_api_key)
        if not key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
                "your OpenRouter API key."
            )
        return key

    def wikipedia_token(self) -> str | None:
        return self._secret(self.wikipedia_api_key)

    def tavily_key(self) -> str | None:
        return self._secret(self.tavily_api_key)

    def brave_key(self) -> str | None:
        return self._secret(self.brave_api_key)

    @property
    def wikipedia_api_url(self) -> str:
        return f"https://{self.wikipedia_language}.wikipedia.org/w/api.php"

    @property
    def openrouter_models_url(self) -> str:
        return f"{self.openrouter_base_url.rstrip('/')}/models"

    @staticmethod
    def _secret(value: SecretStr | None) -> str | None:
        if value is None:
            return None
        secret = value.get_secret_value().strip()
        return secret or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object."""

    return Settings()
