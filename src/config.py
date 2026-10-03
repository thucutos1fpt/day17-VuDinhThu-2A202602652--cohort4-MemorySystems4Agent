from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    provider = os.getenv("LLM_PROVIDER", "openai")
    api_key_by_provider = {
        "openai": os.getenv("OPENAI_API_KEY"), "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"), "ollama": os.getenv("OLLAMA_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"), "custom": os.getenv("CUSTOM_API_KEY"),
    }
    from model_provider import normalize_provider
    provider = normalize_provider(provider)
    model = ProviderConfig(
        provider=provider, model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")), api_key=api_key_by_provider[provider],
        base_url=os.getenv("CUSTOM_BASE_URL") if provider == "custom" else (
            os.getenv("OLLAMA_BASE_URL") if provider == "ollama" else os.getenv("OPENROUTER_BASE_URL")),
    )
    judge = ProviderConfig(
        provider=normalize_provider(os.getenv("JUDGE_PROVIDER", provider)),
        model_name=os.getenv("JUDGE_MODEL", model.model_name), temperature=0,
        api_key=api_key_by_provider.get(normalize_provider(os.getenv("JUDGE_PROVIDER", provider))),
        base_url=model.base_url,
    )
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir,
                     int(os.getenv("COMPACT_THRESHOLD_TOKENS", "700")),
                     int(os.getenv("COMPACT_KEEP_MESSAGES", "6")), model, judge)
