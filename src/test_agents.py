from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 40
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.write_text("dungct", "# User profile\n- **Name:** DũngCT")
    assert store.edit_text("dungct", "DũngCT", "Dũng CT")
    assert "Dũng CT" in store.read_text("dungct")
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for _ in range(5):
        agent.reply("u", "thread", "Một câu rất dài " * 10)
    assert agent.compaction_count("thread") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    message = "Mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    advanced.reply("u", "first", message)
    baseline.reply("u", "first", message)
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    assert "DũngCT" in advanced.reply("u", "fresh", question)["response"]
    assert "DũngCT" not in baseline.reply("u", "fresh", question)["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for _ in range(12):
        text = "Nội dung kiểm thử dài để tạo áp lực prompt context. " * 8
        advanced.reply("u", "long", text)
        baseline.reply("u", "long", text)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
