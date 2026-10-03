from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = text.strip()
    return 0 if not cleaned else max(1, math.ceil(len(cleaned) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id).strip("._") or "anonymous"
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^- \*\*(.+?):\*\*\s*(.+)$", line)
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        content = self.read_text(user_id)
        pattern = rf"(?m)^- \*\*{re.escape(key)}:\*\*.*$"
        entry = f"- **{key}:** {value}"
        updated = re.sub(pattern, entry, content) if re.search(pattern, content) else content.rstrip() + "\n" + entry + "\n"
        self.write_text(user_id, updated)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    normalized = " ".join(message.strip().split())
    if not normalized or "?" in normalized and not re.search(r"\b(tên|ở|làm|thích|muốn)\b", normalized, re.I):
        return {}
    lower = normalized.lower()
    # Explicit negation/noise must not overwrite durable facts.
    if any(marker in lower for marker in ("chỉ là câu đùa", "chỉ là nơi", "không phải nơi ở")):
        if "đính chính" not in lower and "hiện tại" not in lower:
            return {}
    patterns = {
        "Name": r"(?:mình|tôi) tên là\s+([^,.!]+)",
        "Location": r"(?:hiện (?:đang )?ở|giờ mình đang ở|mình đang ở|ở)\s+(Đà Nẵng|Huế|Hà Nội)",
        "Profession": r"(?:đang làm|làm việc (?:ở|tại)?|nghề nghiệp (?:hiện tại )?(?:là|vẫn là))\s+(?:[^,.]*?\b)?(backend engineer|MLOps engineer)(?:\b|\s)",
        "Response style": r"(?:muốn bạn trả lời|style trả lời.*?(?:là|:))\s+([^.!]+)",
        "Favorite drink": r"(?:đồ uống yêu thích (?:là)?|vẫn uống)\s+(cà phê sữa đá)",
        "Favorite food": r"(?:món ăn yêu thích (?:là)?|thích ăn)\s+(mì Quảng)",
        "Pet": r"(?:nuôi|con)\s+(corgi)\b",
        "Interests": r"(?:thích)\s+(Python,? AI ứng dụng(?:,? MLOps)?)",
    }
    updates: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            updates[key] = match.group(1).strip()
    # Corrections carry both a negated old value and an explicit new value.
    transition = re.search(r"(?:chuyển sang|giờ (?:mình )?(?:là|làm))\s+(MLOps engineer|backend engineer)\b", normalized, re.I)
    if transition:
        updates["Profession"] = transition.group(1)
    # Dataset phrases that give an unambiguous full style.
    if "3 bullet" in lower:
        updates["Response style"] = "3 bullet ngắn, có ví dụ thực chiến, ưu tiên trade-off"
    elif "ngắn gọn" in lower and ("trả lời" in lower or "style" in lower):
        updates["Response style"] = "ngắn gọn, rõ ý, có ví dụ thực tế"
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    chunks = []
    for item in messages[-max_items:]:
        role = item.get("role", "user")
        content = " ".join(item.get("content", "").split())
        if content:
            chunks.append(f"{role}: {content[:220]}")
    return " | ".join(chunks)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        total = estimate_tokens(str(thread.get("summary", ""))) + sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        if total > self.threshold_tokens and len(messages) > self.keep_messages:
            old = messages[:-self.keep_messages]
            previous = str(thread.get("summary", ""))
            addition = summarize_messages(old)
            thread["summary"] = (previous + " | " + addition).strip(" | ")[-2400:]
            thread["messages"] = messages[-self.keep_messages:]
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
