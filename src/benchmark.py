from __future__ import annotations

from dataclasses import dataclass, replace
import json
import tempfile
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    found = sum(item.casefold() in answer.casefold() for item in expected)
    return found / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    score = recall_points(answer, expected)
    # A non-empty, concise answer is a usable deterministic proxy in offline mode.
    return min(1.0, score + (0.1 if answer.strip() else 0.0))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    total_agent_tokens = total_prompt_tokens = total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_before = memory_after = 0
    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = str(conversation["id"])
        if hasattr(agent, "memory_file_size"):
            memory_before += agent.memory_file_size(user_id)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, str(turn))
        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)
        for index, query in enumerate(conversation.get("recall_questions", [])):
            answer = agent.reply(user_id, f"{thread_id}-recall-{index}", str(query["question"]))["response"]
            expected = list(query.get("expected_contains", []))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
        if hasattr(agent, "memory_file_size"):
            memory_after += agent.memory_file_size(user_id)
    count = len(recall_scores) or 1
    return BenchmarkRow(agent_name, total_agent_tokens, total_prompt_tokens,
                        sum(recall_scores) / count, sum(quality_scores) / count,
                        max(0, memory_after - memory_before), total_compactions)


def format_rows(rows: list[BenchmarkRow]) -> str:
    columns = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    values = [[r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed,
               f"{r.recall_score:.2f}", f"{r.response_quality:.2f}", r.memory_growth_bytes, r.compactions]
              for r in rows]
    try:
        from tabulate import tabulate
        return tabulate(values, headers=columns, tablefmt="github")
    except ImportError:
        return "\n".join([" | ".join(columns), " | ".join("---" for _ in columns)] +
                         [" | ".join(map(str, row)) for row in values])


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    suites = [("Standard Benchmark", config.data_dir / "conversations.json"),
              ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json")]
    for title, path in suites:
        conversations = load_conversations(path)
        # A temporary state directory makes each printed run reproducible while the
        # advanced agent still persists facts across conversations inside the suite.
        with tempfile.TemporaryDirectory() as state_dir:
            suite_config = replace(config, state_dir=Path(state_dir))
            rows = [
                run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=True), conversations, suite_config),
                run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=True), conversations, suite_config),
            ]
        print(f"\n{title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
