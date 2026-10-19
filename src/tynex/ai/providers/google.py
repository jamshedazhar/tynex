"""Google Generative AI (Gemini) provider, streaming via ``streamGenerateContent``."""

import json
import re
from collections.abc import Iterator
from typing import Any

from tynex.ai.models import Model, provider_base_url
from tynex.ai.providers.base import (
    Chunk,
    LlmContext,
    Provider,
    ProviderError,
    ProviderHttpError,
    StreamOptions,
    finish_chunks,
    raise_for_status,
    thinking_budget,
    tool_declarations,
)
from tynex.ai.types import ImageContent, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from tynex.http import iter_json_events, stream

# Gemini answers 404 for models retired for new API keys and names the
# replacement in the message: "... use models/gemini-3.8-flash ...".
_GONE_MODEL_SUGGESTION = re.compile(r"use models/([A-Za-z0-9._-]+)")


def _gone_model_hint(model: Model, error: ProviderHttpError) -> str | None:
    """Turn a deprecated-model 404 into an actionable switch instruction."""
    if error.status != 404:
        return None
    match = _GONE_MODEL_SUGGESTION.search(error.body or "")
    if match is None or match.group(1) == model.id:
        return None
    suggested = match.group(1)
    return (
        f"HTTP 404: model '{model.id}' is no longer available to this API key — the provider recommends "
        f"'{suggested}'. Switch with /model {model.provider}/{suggested} (or -m {model.provider}/{suggested})."
    )


class GoogleProvider(Provider):
    name = "google"

    # -- conversion -----------------------------------------------------------
    def _convert_contents(self, context: LlmContext) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for message in context.conversation():
            role = getattr(message, "role", None)
            if role == "user":
                assert isinstance(message, UserMessage)
                parts: list[dict[str, Any]] = []
                if isinstance(message.content, str):
                    if message.content:
                        parts.append({"text": message.content})
                else:
                    for part in message.content:
                        if isinstance(part, TextContent):
                            parts.append({"text": part.text})
                        elif isinstance(part, ImageContent):
                            parts.append({"inlineData": {"mimeType": part.mime_type, "data": part.data}})
                if parts:
                    contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                parts = []
                for part in message.content:
                    if isinstance(part, TextContent) and part.text:
                        item: dict[str, Any] = {"text": part.text}
                        if part.signature:
                            item["thoughtSignature"] = part.signature
                        parts.append(item)
                    elif isinstance(part, ToolCall):
                        item = {"functionCall": {"name": part.name, "args": part.arguments}}
                        # Gemini 3 rejects replayed function calls without their signature.
                        if part.signature:
                            item["thoughtSignature"] = part.signature
                        parts.append(item)
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "toolResult":
                assert isinstance(message, ToolResultMessage)
                response_payload: dict[str, Any] = {"result": message.text() or "(no output)"}
                if message.is_error:
                    response_payload = {"error": message.text() or "tool error"}
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"functionResponse": {"name": message.tool_name, "response": response_payload}}],
                    }
                )
        return contents

    def _convert_tools(self, context: LlmContext) -> list[dict[str, Any]]:
        declarations = tool_declarations(context)
        return [{"functionDeclarations": declarations}] if declarations else []

    # -- streaming --------------------------------------------------------------
    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        api_key = self.resolve_api_key(model, options)
        if not api_key:
            raise RuntimeError("No Google API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or run /login in tynex.")
        base_url = (options.base_url or model.base_url or provider_base_url("google") or "").rstrip("/")
        url = f"{base_url}/v1beta/models/{model.id}:streamGenerateContent?alt=sse"
        generation_config: dict[str, Any] = {"maxOutputTokens": options.max_tokens or model.max_tokens}
        budget = thinking_budget(options.thinking)
        if budget and model.reasoning:
            # Gemini's thinking budget tops out at 24576 (2.5 Flash) / 32768
            # (2.5 Pro); stay within the safest common bound.
            budget = min(budget, 24_576)
            generation_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": budget}
        if options.temperature is not None:
            generation_config["temperature"] = options.temperature
        body: dict[str, Any] = {"contents": self._convert_contents(context), "generationConfig": generation_config}
        system = context.system_text()
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        tools = self._convert_tools(context)
        if tools:
            body["tools"] = tools
        headers = {"content-type": "application/json", "accept": "text/event-stream", "x-goog-api-key": api_key}
        usage = Usage()
        finish_reason: str | None = None
        call_counter = 0
        saw_tool_call = False
        with stream("POST", url, json_body=body, headers=headers) as response:
            try:
                raise_for_status(response)
            except ProviderHttpError as error:
                hint = _gone_model_hint(model, error)
                if hint is None:
                    raise
                raise ProviderError(hint) from error
            for event in iter_json_events(response.iter_lines()):
                if event.get("error"):
                    err = event["error"]
                    message = err.get("message") if isinstance(err, dict) else str(err)
                    yield Chunk(kind="error", error=message)
                    return
                meta = event.get("usageMetadata") or {}
                if meta:
                    cached = meta.get("cachedContentTokenCount", 0) or 0
                    usage.input = max(0, meta.get("promptTokenCount", 0) - cached)
                    usage.output = meta.get("candidatesTokenCount", 0) or 0
                    usage.cache_read = cached
                    usage.total_tokens = usage.input + usage.output
                for candidate in event.get("candidates") or []:
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        signature = part.get("thoughtSignature")
                        fc = part.get("functionCall")
                        if fc:
                            saw_tool_call = True
                            call_counter += 1
                            yield Chunk(
                                kind="toolcall_start",
                                tool_id=f"call_{call_counter}_{fc.get('name', 'tool')}",
                                tool_name=fc.get("name", ""),
                                signature=signature,
                            )
                            yield Chunk(kind="toolcall_delta", json_fragment=json.dumps(fc.get("args") or {}))
                        elif part.get("thought") and part.get("text"):
                            yield Chunk(kind="thinking", text=part["text"])
                            if signature:
                                yield Chunk(kind="thinking_signature", signature=signature)
                        elif part.get("text"):
                            yield Chunk(kind="text", text=part["text"], signature=signature)
                        elif signature:
                            # Gemini 3 puts the response thought signature on a
                            # trailing empty text part.
                            yield Chunk(kind="signature", signature=signature)
                    if candidate.get("finishReason"):
                        finish_reason = candidate["finishReason"]
        stop_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "FINISH_REASON_UNSPECIFIED": "stop",
            "SAFETY": "stop",
            "RECITATION": "stop",
        }
        yield from finish_chunks(model, usage, finish_reason, stop_map, saw_tool_call)
