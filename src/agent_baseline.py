from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is not None:
            state = self.sessions.setdefault(thread_id, SessionState())
            prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages) + estimate_tokens(message)
            result = self.langchain_agent.invoke({"messages": [("user", message)]}, config={"configurable": {"thread_id": thread_id}})
            content = result["messages"][-1].content
            state.messages.extend([{"role": "user", "content": message}, {"role": "assistant", "content": str(content)}])
            state.prompt_tokens_processed += prompt_tokens
            state.token_usage += estimate_tokens(str(content))
            return {"response": str(content), "tokens": estimate_tokens(str(content)), "prompt_tokens": prompt_tokens}
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        # TODO: return cumulative agent token count for one thread.
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        # TODO: estimate how much prompt context this baseline kept processing.
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        state = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in state.messages) + estimate_tokens(message)
        state.messages.append({"role": "user", "content": message})
        # This deliberately uses only the current session: no profile is consulted.
        response = "Mình đã nhận được nội dung trong phiên hiện tại."
        state.messages.append({"role": "assistant", "content": response})
        generated = estimate_tokens(response)
        state.token_usage += generated
        state.prompt_tokens_processed += prompt_tokens
        return {"response": response, "tokens": generated, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langchain.agents import create_agent
            return create_agent(build_chat_model(self.config.model), checkpointer=InMemorySaver())
        except (ImportError, ModuleNotFoundError):
            return None
