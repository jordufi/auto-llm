from __future__ import annotations

from typing import Iterable


def format_chat(messages: Iterable[dict], system_prompt: str = "You are a helpful assistant.") -> str:
    parts = [f"<|system|>\n{system_prompt.strip()}\n"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)
