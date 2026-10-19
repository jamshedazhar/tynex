"""Scripted offline provider ("faux")."""

import json
import os
from collections.abc import Iterator

from tynex.ai.models import Model
from tynex.ai.providers.base import Chunk, LlmContext, Provider, StreamOptions, compute_cost, last_user_text
from tynex.ai.types import Usage

# Scripts for scripted/offline operation: a list of scenarios, each with events.
script: list[dict] | None = None
_script_index = 0


def reset_script(new_script: list[dict] | None = None) -> None:
    global script, _script_index
    script = new_script
    _script_index = 0


def _load_script_from_env(raw: str) -> list[dict] | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        if os.path.isfile(raw):
            with open(raw, encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FauxProvider(Provider):
    """Deterministic provider for offline end-to-end tests."""

    name = "faux"

    def _next_scenario(self) -> dict | None:
        global _script_index
        active = script
        if active is None:
            raw = os.environ.get("TYNEX_FAUX_SCRIPT")
            if raw:
                active = _load_script_from_env(raw)
        if not active:
            return None
        if _script_index >= len(active):
            return None
        scenario = active[_script_index]
        _script_index += 1
        return scenario

    def _chunks(self, model: Model, context: LlmContext, options: StreamOptions) -> Iterator[Chunk]:
        scenario = self._next_scenario()
        user_text = last_user_text(context)
        if scenario is None:
            text = f"[faux] {user_text}" if user_text else "[faux] no scripted response"
            yield Chunk(kind="text", text=text)
            input_tokens = _estimate_tokens("".join(str(getattr(m, "content", "")) for m in context.conversation()))
            usage = Usage(input=input_tokens, output=_estimate_tokens(text))
            usage.total_tokens = usage.input + usage.output
            compute_cost(model, usage)
            yield Chunk(kind="usage", usage=usage)
            yield Chunk(kind="stop", stop_reason="stop")
            return
        saw_usage = False
        saw_stop = False
        for event in scenario.get("events", []):
            etype = event.get("type")
            if etype == "text":
                text = event.get("text", "")
                # Emit in small deltas so streaming UIs get exercised.
                for i in range(0, len(text), 24):
                    yield Chunk(kind="text", text=text[i : i + 24])
            elif etype == "thinking":
                yield Chunk(kind="thinking", text=event.get("text", ""))
            elif etype == "thinking_signature":
                yield Chunk(kind="thinking_signature", signature=event.get("signature", ""))
            elif etype == "toolcall":
                yield Chunk(
                    kind="toolcall_start", tool_id=event.get("id", f"call_{id(event)}"), tool_name=event.get("name", "")
                )
                yield Chunk(kind="toolcall_delta", json_fragment=json.dumps(event.get("arguments", {})))
            elif etype == "usage":
                u = event.get("usage", {})
                usage = Usage(
                    input=u.get("input", 0),
                    output=u.get("output", 0),
                    cache_read=u.get("cacheRead", 0),
                    cache_write=u.get("cacheWrite", 0),
                )
                usage.total_tokens = usage.input + usage.output
                compute_cost(model, usage)
                saw_usage = True
                yield Chunk(kind="usage", usage=usage)
            elif etype == "stop":
                saw_stop = True
                yield Chunk(kind="stop", stop_reason=event.get("reason", "stop"))
            elif etype == "error":
                yield Chunk(kind="error", error=event.get("error", "faux error"))
                return
        if not saw_usage:
            input_tokens = _estimate_tokens("".join(str(getattr(m, "content", "")) for m in context.conversation()))
            usage = Usage(input=input_tokens, output=16)
            usage.total_tokens = usage.input + usage.output
            compute_cost(model, usage)
            yield Chunk(kind="usage", usage=usage)
        if not saw_stop:
            yield Chunk(kind="stop", stop_reason="stop")
