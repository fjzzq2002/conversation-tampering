"""Provider-agnostic stateful chat client.

Wraps OpenRouter / OpenAI / Anthropic chat completions behind a uniform interface.
Owns the full conversation history (`turns`), supports prefill (Anthropic-style),
extended thinking, and round-trip JSON serialization for save/replay.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import anthropic
from openai import OpenAI


# Anthropic extended-thinking budget tokens by reasoning-effort level.
REASONING_TO_BUDGET: dict[str, int] = {
    "minimal": 4096,
    "low": 8000,
    "medium": 16000,
    "high": 24000,
}
# Extra max_tokens above the thinking budget so the model still has room for the actual reply.
ANTHROPIC_OUTPUT_HEADROOM: int = 4096

# OpenRouter / OpenAI reasoning-effort levels we accept. None or "off" disables reasoning.
REASONING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high")


def call_with_backoff(fn, *, max_attempts: int = 6, base: float = 2.0):
    """Run `fn()` with exponential backoff on transient API errors."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError):
            transient = True
        except Exception as e:
            name = type(e).__name__
            transient = name in (
                "APIConnectionError", "RateLimitError", "APITimeoutError",
                "InternalServerError", "APIStatusError",
            ) or "rate" in str(e).lower() or "timeout" in str(e).lower()
            if not transient:
                raise
        if attempt == max_attempts - 1:
            raise
        wait = base ** attempt + random.uniform(0, 0.5)
        print(f"[backoff] attempt {attempt + 1} failed; sleeping {wait:.1f}s",
              file=sys.stderr, flush=True)
        time.sleep(wait)


@dataclass
class TurnRecord:
    """One conversation turn."""
    role: str  # "user" or "assistant"
    content: str  # canonical visible text
    thinking: str | None = None  # human-readable thinking trace, if any
    # Anthropic raw content blocks (with thinking signatures), needed to carry thinking
    # forward across turns. Populated only on assistant turns from the Anthropic provider.
    raw_blocks: list[dict[str, Any]] | None = field(default=None, repr=False)


@dataclass
class _Reply:
    """Internal: one API call's parsed result."""
    text: str
    thinking: str | None = None
    raw_blocks: list[dict[str, Any]] | None = None


class Chatter:
    """Stateful chat client that owns conversation history.

    Features
      * Three providers: 'anthropic' (native API), 'openai' (direct), 'openrouter' (compatible).
      * Anthropic-only: extended thinking (`reasoning_effort`), thinking trace exposure,
        prefill via trailing assistant message, and `carry_thinking` to feed prior thinking
        blocks back into subsequent turns.
      * Tampering: `tamper_last(new_text)` rewrites the last assistant message's visible
        text, preserving any associated thinking blocks (useful for gaslight experiments).
      * Serialization: `to_dict()` / `save(path)` / `Chatter.from_dict(d)` / `Chatter.load(p)`
        give round-trip JSON state.
    """

    def __init__(self, provider: str, model: str, *, carry_thinking: bool = False):
        self.provider = provider
        self.model_arg = model  # keep original (e.g. "anthropic/claude-opus-4.5") for portability
        self.carry_thinking = carry_thinking
        self.turns: list[TurnRecord] = []
        self._setup_client()

    # -- client setup ---------------------------------------------------------

    def _setup_client(self) -> None:
        if self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self.client: Any = anthropic.Anthropic(api_key=api_key)
            self.model = self._normalize_anthropic(self.model_arg)
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
            self.model = self.model_arg.removeprefix("openai/")
        elif self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            self.model = self.model_arg
        elif self.provider == "vllm":
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
            api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = self.model_arg
        else:
            raise ValueError(f"unknown provider: {self.provider!r}")

    @staticmethod
    def _normalize_anthropic(model: str) -> str:
        # "anthropic/claude-opus-4.5" -> "claude-opus-4-5"
        m = model.removeprefix("anthropic/")
        return re.sub(r"(\d)\.(\d)", r"\1-\2", m)

    # -- conversation interface ----------------------------------------------

    def add_user(self, content: str) -> TurnRecord:
        """Append a user message to history. No API call."""
        turn = TurnRecord("user", content)
        self.turns.append(turn)
        return turn

    def add_assistant(self, content: str, *, thinking: str | None = None,
                      raw_blocks: list[dict] | None = None) -> TurnRecord:
        """Append a synthetic assistant message (e.g., to seed history). No API call."""
        turn = TurnRecord("assistant", content, thinking=thinking, raw_blocks=raw_blocks)
        self.turns.append(turn)
        return turn

    def respond(self, *, reasoning_effort: str | None = None,
                prefill: str | None = None) -> TurnRecord:
        """Generate an assistant response from the current history; append it; return it.

        `reasoning_effort` ∈ {None, "off", "minimal", "low", "medium", "high"}.
        For Anthropic, non-None enables extended thinking.
        For OpenAI, sets `reasoning_effort`. For OpenRouter, sets nested `reasoning.effort`.

        `prefill` (Anthropic-recommended): force the assistant response to start with this
        string. The model continues from the prefill; the returned turn's `content` is
        prefill + continuation. Not compatible with `reasoning_effort` on Anthropic.
        """
        if reasoning_effort and reasoning_effort not in REASONING_LEVELS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_LEVELS}, got {reasoning_effort!r}")
        reply = self._call(reasoning_effort=reasoning_effort, prefill=prefill)
        content = (prefill + reply.text) if prefill is not None else reply.text
        turn = TurnRecord(
            role="assistant",
            content=content,
            thinking=reply.thinking,
            raw_blocks=reply.raw_blocks,
        )
        self.turns.append(turn)
        return turn

    def send(self, user_message: str, *, reasoning_effort: str | None = None,
             prefill: str | None = None) -> TurnRecord:
        """Convenience: `add_user(user_message)` then `respond(...)`."""
        self.add_user(user_message)
        return self.respond(reasoning_effort=reasoning_effort, prefill=prefill)

    def tamper_last(self, new_content: str) -> None:
        """Rewrite the visible text of the last assistant turn (keeps thinking + blocks intact)."""
        if not self.turns or self.turns[-1].role != "assistant":
            raise RuntimeError("no assistant turn at end of history")
        self.turns[-1].content = new_content

    # -- provider-specific raw API calls -------------------------------------

    def _call(self, *, reasoning_effort: str | None, prefill: str | None) -> _Reply:
        if self.provider == "anthropic":
            return self._call_anthropic(reasoning_effort=reasoning_effort, prefill=prefill)
        if self.provider == "openai":
            return self._call_openai_responses(reasoning_effort=reasoning_effort, prefill=prefill)
        # openrouter + vllm both speak Chat Completions
        return self._call_openrouter(reasoning_effort=reasoning_effort, prefill=prefill)

    # ---- Anthropic ---------------------------------------------------------

    def _build_anthropic_messages(self, prefill: str | None) -> list[dict]:
        out: list[dict] = []
        for t in self.turns:
            if t.role == "user":
                out.append({"role": "user", "content": t.content})
                continue
            if self.carry_thinking and t.raw_blocks:
                blocks: list[dict] = [b for b in t.raw_blocks if b.get("type") == "thinking"]
                if t.content:
                    blocks.append({"type": "text", "text": t.content})
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "assistant", "content": t.content})
        if prefill is not None:
            out.append({"role": "assistant", "content": prefill})
        return out

    def _call_anthropic(self, *, reasoning_effort: str | None, prefill: str | None) -> _Reply:
        messages = self._build_anthropic_messages(prefill)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 1.0,
        }
        if reasoning_effort and reasoning_effort != "off":
            budget = REASONING_TO_BUDGET.get(reasoning_effort, 16000)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = budget + ANTHROPIC_OUTPUT_HEADROOM
        resp = call_with_backoff(lambda: self.client.messages.create(**kwargs))
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        raw_blocks: list[dict] = []
        for b in resp.content:
            bt = getattr(b, "type", None)
            if bt == "thinking":
                thinking_parts.append(b.thinking)
                raw_blocks.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
            elif bt == "text":
                text_parts.append(b.text)
                raw_blocks.append({"type": "text", "text": b.text})
        return _Reply(
            text="".join(text_parts),
            thinking="\n".join(thinking_parts) if thinking_parts else None,
            raw_blocks=raw_blocks or None,
        )

    # ---- OpenAI direct (Responses API) -------------------------------------

    def _build_openai_input(self, prefill: str | None) -> list[dict]:
        """Build the `input` array for the Responses API.
        Reasoning items (when carry_thinking=True) are emitted as separate items before
        the assistant message they belong to, so the model has its prior reasoning context.
        """
        out: list[dict] = []
        for t in self.turns:
            if t.role == "user":
                out.append({"role": "user", "content": t.content})
                continue
            # assistant turn
            if self.carry_thinking and t.raw_blocks:
                for b in t.raw_blocks:
                    if b.get("type") == "reasoning":
                        item = {
                            "type": "reasoning",
                            "id": b["id"],
                            "summary": b.get("summary", []),
                        }
                        if b.get("encrypted_content"):
                            item["encrypted_content"] = b["encrypted_content"]
                        out.append(item)
            if t.content:
                out.append({"role": "assistant", "content": t.content})
        if prefill is not None:
            out.append({"role": "assistant", "content": prefill})
        return out

    def _call_openai_responses(self, *, reasoning_effort: str | None, prefill: str | None) -> _Reply:
        input_items = self._build_openai_input(prefill)
        kwargs: dict = {
            "model": self.model,
            "input": input_items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if reasoning_effort and reasoning_effort != "off":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = call_with_backoff(lambda: self.client.responses.create(**kwargs))
        text_parts: list[str] = []
        summary_parts: list[str] = []
        raw_blocks: list[dict] = []
        for item in resp.output:
            it = getattr(item, "type", None)
            if it == "reasoning":
                summary_text = ""
                if getattr(item, "summary", None):
                    summary_text = "\n".join(
                        s.text if hasattr(s, "text") else str(s) for s in item.summary
                    )
                if summary_text:
                    summary_parts.append(summary_text)
                raw_blocks.append({
                    "type": "reasoning",
                    "id": item.id,
                    "summary": [s.model_dump() if hasattr(s, "model_dump") else s for s in (item.summary or [])],
                    "encrypted_content": getattr(item, "encrypted_content", None),
                    "status": getattr(item, "status", None),
                })
            elif it == "message":
                for c in item.content:
                    if getattr(c, "type", None) == "output_text":
                        text_parts.append(c.text)
        return _Reply(
            text="".join(text_parts),
            thinking="\n".join(summary_parts) if summary_parts else None,
            raw_blocks=raw_blocks or None,
        )

    # ---- OpenRouter (Chat Completions) -------------------------------------

    def _build_openrouter_messages(self, prefill: str | None) -> list[dict]:
        out: list[dict] = [{"role": t.role, "content": t.content} for t in self.turns]
        if prefill is not None:
            out.append({"role": "assistant", "content": prefill})
        return out

    def _call_openrouter(self, *, reasoning_effort: str | None, prefill: str | None) -> _Reply:
        messages = self._build_openrouter_messages(prefill)
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": 1.0}
        extra_body: dict = {}
        if self.provider == "vllm":
            # Cap output to avoid degenerate loops blowing context.
            kwargs["max_tokens"] = int(os.environ.get("VLLM_MAX_TOKENS", "256"))
            if prefill is not None:
                # vLLM extension: continue from the trailing assistant message
                # rather than start a fresh assistant turn.
                extra_body["continue_final_message"] = True
                extra_body["add_generation_prompt"] = False
        if reasoning_effort and reasoning_effort != "off":
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = call_with_backoff(lambda: self.client.chat.completions.create(**kwargs))
        return _Reply(text=resp.choices[0].message.content or "")

    # -- serialization --------------------------------------------------------

    def to_dict(self, *, include_raw_blocks: bool = True) -> dict:
        """Serialize to a JSON-friendly dict. Round-trippable via `Chatter.from_dict`."""
        return {
            "provider": self.provider,
            "model": self.model_arg,
            "carry_thinking": self.carry_thinking,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    **({"thinking": t.thinking} if t.thinking else {}),
                    **({"raw_blocks": t.raw_blocks} if (include_raw_blocks and t.raw_blocks) else {}),
                }
                for t in self.turns
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> Chatter:
        c = cls(data["provider"], data["model"], carry_thinking=data.get("carry_thinking", False))
        for tr in data.get("turns", []):
            c.turns.append(TurnRecord(
                role=tr["role"],
                content=tr["content"],
                thinking=tr.get("thinking"),
                raw_blocks=tr.get("raw_blocks"),
            ))
        return c

    @classmethod
    def load(cls, path: str | Path) -> Chatter:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- container-y conveniences --------------------------------------------

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Iterable[TurnRecord]:
        return iter(self.turns)

    def __repr__(self) -> str:
        return f"Chatter(provider={self.provider!r}, model={self.model!r}, turns={len(self.turns)})"
