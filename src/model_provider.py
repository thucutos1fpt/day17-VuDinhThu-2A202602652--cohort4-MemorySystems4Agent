from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a supported, canonical provider name."""
    aliases = {
        "openai": "openai", "custom": "custom", "openai-compatible": "custom",
        "gemini": "gemini", "google": "gemini", "anthropic": "anthropic",
        "anthorpic": "anthropic", "claude": "anthropic", "ollama": "ollama",
        "openrouter": "openrouter", "open-router": "openrouter",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported provider {value!r}. Supported providers: {supported}.")
    return normalized


def build_chat_model(config: ProviderConfig) -> Any:
    """Build a LangChain chat model lazily.

    Imports live here so offline benchmarks do not require optional SDK packages.
    """
    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}
    if provider in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI
        kwargs = dict(common)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom" and config.base_url:
            kwargs["base_url"] = config.base_url
        if provider == "openrouter":
            kwargs["base_url"] = config.base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**common, google_api_key=config.api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**common, api_key=config.api_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(**common, base_url=config.base_url) if config.base_url else ChatOllama(**common)
    raise AssertionError("normalize_provider returned an unknown provider")
