"""LLM client adapters for the L3 NL-to-SQL agent.

The OSS core has no LLM dependency (``docs/contracts/architecture.md``
Invariant #6). This module talks to provider HTTP endpoints directly via
``httpx`` (already a top-level dep), so ``import pretensor`` never pulls
an LLM SDK. Instantiation is lazy: callers build an :class:`AnthropicHttpClient`
or :class:`OpenAIHttpClient` inside the runner — module load does not call
``os.environ`` or open any sockets.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from pretensor.benchmark.l3.agent import (
    AgentMessage,
    AgentStep,
    AgentTool,
    AgentToolCall,
)

__all__ = [
    "AnthropicHttpClient",
    "LlmCallError",
    "LlmClient",
    "LlmResponse",
    "OpenAIHttpClient",
]


_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_MAX_TOKENS = 4096
_ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_OPENAI_API = "https://api.openai.com/v1/chat/completions"


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """One LLM completion plus the runner's per-call telemetry.

    ``text`` is the raw model output (no fence stripping — that's the
    runner's job). ``prompt_tokens`` / ``completion_tokens`` are
    optional; some providers omit them. ``latency_ms`` is wall-clock
    time around the HTTP request, captured by the client itself so
    the runner does not need to reach into provider-specific timing.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class LlmCallError(RuntimeError):
    """Raised on transport failure, non-2xx response, or unparseable body."""


@runtime_checkable
class LlmClient(Protocol):
    """Sync NL-to-SQL completion surface.

    A single ``complete`` call is one prompt → one completion. Sync because
    L3 evaluates one question at a time and the LLM call is the dominant
    latency — async would only buy concurrency the runner does not use.
    """

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
    ) -> LlmResponse: ...


class AnthropicHttpClient:
    """Default Anthropic Messages API client.

    Reads ``ANTHROPIC_API_KEY`` from env at construction; raises
    :class:`LlmCallError` if absent. Uses an internally-owned
    :class:`httpx.Client` so the runner does not need to manage the
    transport.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LlmCallError(
                "ANTHROPIC_API_KEY is not set; cannot run the Anthropic client."
            )
        self._api_key = resolved_key
        self._client = client or httpx.Client(timeout=_DEFAULT_TIMEOUT_S)
        self._max_tokens = max_tokens

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
    ) -> LlmResponse:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        payload, latency_ms = self._post(body)
        text = _extract_anthropic_text(payload)
        usage = payload.get("usage") or {}
        return LlmResponse(
            text=text,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
        )

    def agent_complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        model: str,
        temperature: float,
    ) -> AgentStep:
        """One tool-use step against the Anthropic Messages API.

        Translates :class:`AgentMessage` history to Anthropic's
        content-block wire format, sends the request with the supplied
        ``tools`` array, and decodes the response into a provider-agnostic
        :class:`AgentStep`. ``stop_reason="tool_use"`` becomes
        ``tool_calls`` populated; ``stop_reason="end_turn"`` (or
        ``"stop_sequence"``) becomes a final ``text``.
        """
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [_anthropic_message(m) for m in messages],
            "tools": [_anthropic_tool(t) for t in tools],
        }
        payload, latency_ms = self._post(body)
        text, tool_calls = _decode_anthropic_step(payload)
        usage = payload.get("usage") or {}
        return AgentStep(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
        )

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST ``body`` to Anthropic, returning ``(payload, latency_ms)``.

        Centralises the transport / error / timing path shared between
        ``complete`` and ``agent_complete`` so the two stay byte-for-byte
        consistent on retry semantics, header set, and timeout.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        try:
            response = self._client.post(_ANTHROPIC_API, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LlmCallError(f"Anthropic transport error: {exc}") from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if response.status_code >= 400:
            raise LlmCallError(
                f"Anthropic returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlmCallError(f"Anthropic response was not JSON: {exc}") from exc
        return payload, latency_ms


class OpenAIHttpClient:
    """OpenAI chat-completions client.

    Reads ``OPENAI_API_KEY`` from env at construction. Same shape as the
    Anthropic client so the runner can swap providers behind the
    :class:`LlmClient` Protocol.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise LlmCallError(
                "OPENAI_API_KEY is not set; cannot run the OpenAI client."
            )
        self._api_key = resolved_key
        self._client = client or httpx.Client(timeout=_DEFAULT_TIMEOUT_S)
        self._max_tokens = max_tokens

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
    ) -> LlmResponse:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        try:
            response = self._client.post(_OPENAI_API, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LlmCallError(f"OpenAI transport error: {exc}") from exc
        latency_ms = int((time.perf_counter() - t0) * 1000)
        if response.status_code >= 400:
            raise LlmCallError(
                f"OpenAI returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlmCallError(f"OpenAI response was not JSON: {exc}") from exc
        text = _extract_openai_text(payload)
        usage = payload.get("usage") or {}
        return LlmResponse(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    """Concatenate every ``text`` block from an Anthropic Messages response.

    Tool-use blocks (which the L3 baseline does not request) are skipped.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        raise LlmCallError("Anthropic response missing 'content' array.")
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_part = block.get("text")
            if isinstance(text_part, str):
                parts.append(text_part)
    if not parts:
        raise LlmCallError("Anthropic response had no text blocks.")
    return "".join(parts)


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmCallError("OpenAI response missing 'choices' array.")
    first = choices[0]
    if not isinstance(first, dict):
        raise LlmCallError("OpenAI 'choices[0]' is not an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LlmCallError("OpenAI 'choices[0].message' is missing.")
    content = message.get("content")
    if not isinstance(content, str):
        raise LlmCallError("OpenAI 'choices[0].message.content' is not a string.")
    return content


def _anthropic_tool(tool: AgentTool) -> dict[str, Any]:
    """Render an :class:`AgentTool` as Anthropic's tool descriptor."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _anthropic_message(message: AgentMessage) -> dict[str, Any]:
    """Render an :class:`AgentMessage` as one Anthropic ``messages[]`` entry.

    A ``user`` turn with tool results becomes a content array of
    ``tool_result`` blocks; an ``assistant`` turn merges any text +
    ``tool_use`` blocks. Plain user prompts stay as a string so the wire
    payload matches the format the simple ``complete`` path emits.
    """
    if message.role == "user":
        if message.tool_results:
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.id,
                        "content": r.content,
                        **({"is_error": True} if r.is_error else {}),
                    }
                    for r in message.tool_results
                ],
            }
        return {"role": "user", "content": message.text or ""}

    blocks: list[dict[str, Any]] = []
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
        )
    return {"role": "assistant", "content": blocks}


def _decode_anthropic_step(
    payload: dict[str, Any],
) -> tuple[str | None, tuple[AgentToolCall, ...]]:
    """Pull text and tool calls out of an Anthropic Messages response.

    Returns ``(text, tool_calls)``. ``text`` is the concatenated text
    blocks (may be empty when the model only emitted tool calls);
    ``tool_calls`` is empty when the response is purely text.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        raise LlmCallError("Anthropic response missing 'content' array.")
    text_parts: list[str] = []
    tool_calls: list[AgentToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_part = block.get("text")
            if isinstance(text_part, str):
                text_parts.append(text_part)
        elif block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            args = block.get("input")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise LlmCallError("Anthropic tool_use block missing 'id' or 'name'.")
            # ``input`` is contractually a JSON object on Anthropic's API
            # (no-arg tools still emit ``{}``). A missing or non-dict
            # value is a provider-side regression — silently coercing
            # it to ``{}`` would forward an empty-args tool call and
            # mask the bug. Fail loud so callers can record the
            # underlying provider response.
            if not isinstance(args, dict):
                raise LlmCallError(
                    f"Anthropic tool_use block 'input' for tool {name!r} "
                    f"is not a JSON object (got {type(args).__name__})."
                )
            tool_calls.append(
                AgentToolCall(id=call_id, name=name, arguments=dict(args))
            )
    text = "".join(text_parts) if text_parts else None
    return text, tuple(tool_calls)
