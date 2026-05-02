from __future__ import annotations

import asyncio
import sys
from typing import Any

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, TextBlock, query
from claude_code_sdk._errors import ClaudeSDKError, MessageParseError
from claude_code_sdk._internal import client as _sdk_client
from claude_code_sdk._internal.message_parser import parse_message as _original_parse
from claude_code_sdk.types import SystemMessage

_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-7",
}


def _safe_parse(data: dict[str, Any]) -> Any:
    """Wrap parse_message to return a no-op SystemMessage for unknown event types.

    SDK v0.0.25 raises MessageParseError on unknown types like rate_limit_event,
    which are informational CLI events that don't affect the response.
    """
    try:
        return _original_parse(data)
    except MessageParseError as e:
        if "Unknown message type" in str(e):
            return SystemMessage(subtype="unknown", data=data)
        raise


_sdk_client.parse_message = _safe_parse  # type: ignore[attr-defined]


async def call_claude_async(prompt: str, *, model: str = "sonnet") -> str | None:
    """Call Claude via the SDK. Use this in async contexts (e.g. parallel batch calls)."""
    model_id = _MODEL_MAP.get(model, model)
    options = ClaudeCodeOptions(
        model=model_id,
        max_turns=1,
        allowed_tools=[],
    )
    parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "".join(parts) if parts else None


def call_claude(prompt: str, *, model: str = "sonnet", timeout: int = 120) -> str | None:
    """Call Claude via the Claude Code SDK, return text or None on failure."""
    try:
        coro = asyncio.wait_for(call_claude_async(prompt, model=model), timeout=timeout)
        return asyncio.run(coro)
    except asyncio.TimeoutError:
        print(f"Warning: Claude call timed out after {timeout}s", file=sys.stderr)
        return None
    except ClaudeSDKError as e:
        print(f"Warning: Claude SDK error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Claude call failed: {e}", file=sys.stderr)
        return None
