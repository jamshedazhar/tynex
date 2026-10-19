"""Gemini provider: retired models and thought signatures (offline)."""

import json

import httpx2

from tynex import http
from tynex.ai.models import Model
from tynex.ai.providers.base import LlmContext, StreamOptions
from tynex.ai.providers.google import GoogleProvider
from tynex.ai.types import TextContent, ToolCall, UserMessage, content_from_dict, content_to_dict

GONE_BODY = {
    "error": {
        "code": 404,
        "message": (
            "This model models/gemini-2.5-flash is no longer available to new users. "
            "Please update your code to use models/gemini-3.8-flash for the latest features and improvements."
        ),
        "status": "NOT_FOUND",
    }
}


def google_model(model_id: str = "gemini-3.8-flash") -> Model:
    return Model(id=model_id, name=model_id, provider="google", api="google-generative-ai", reasoning=True)


def sse(events: list[dict]) -> httpx2.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def stream_once(events: list[dict], model: Model | None = None):
    http.set_client(httpx2.Client(transport=httpx2.MockTransport(lambda request: sse(events))))
    provider = GoogleProvider()
    context = LlmContext(messages=[UserMessage(content="hello")])
    streamed = list(provider.stream(model or google_model(), context, StreamOptions(api_key="test")))
    return provider, streamed[-1]["partial"]


def stream_error(status: int, body: dict) -> str:
    http.set_client(httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(status, json=body))))
    model = google_model("gemini-2.5-flash")
    events = list(GoogleProvider().stream(model, LlmContext(), StreamOptions(api_key="test")))
    assert events[-1]["type"] == "error"
    return events[-1]["error"]


def test_retired_model_404_gets_an_actionable_switch_hint():
    message = stream_error(404, GONE_BODY)
    assert "no longer available" in message
    assert "/model google/gemini-3.8-flash" in message
    assert "-m google/gemini-3.8-flash" in message


def test_other_errors_keep_the_provider_body():
    message = stream_error(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    assert message.startswith("HTTP 400:")
    assert "API key not valid" in message


def test_repeated_usage_metadata_does_not_accumulate_cache_reads():
    events = [
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "A"}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 900, "candidatesTokenCount": 1},
        },
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "B"}]}}],
            "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 900, "candidatesTokenCount": 2},
        },
    ]
    _, assistant = stream_once(events)
    assert assistant.usage.input == 100
    assert assistant.usage.output == 2
    assert assistant.usage.cache_read == 900


def test_function_call_thought_signature_is_captured_and_replayed():
    events = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "bash", "args": {"command": "ls"}},
                                "thoughtSignature": "sig-tool",
                            }
                        ],
                    }
                }
            ]
        },
        {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP"}]},
    ]
    provider, assistant = stream_once(events)

    call = next(part for part in assistant.content if isinstance(part, ToolCall))
    assert call.name == "bash"
    assert call.arguments == {"command": "ls"}
    assert call.signature == "sig-tool"

    contents = provider._convert_contents(LlmContext(messages=[UserMessage(content="hello"), assistant]))
    replayed = contents[-1]["parts"][0]
    assert replayed == {"functionCall": {"name": "bash", "args": {"command": "ls"}}, "thoughtSignature": "sig-tool"}


def test_text_thought_signature_is_captured_and_replayed():
    events = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]},
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "", "thoughtSignature": "sig-text"}]},
                    "finishReason": "STOP",
                }
            ]
        },
    ]
    provider, assistant = stream_once(events)

    texts = [part for part in assistant.content if isinstance(part, TextContent)]
    assert len(texts) == 1
    assert texts[0].text == "Hello"
    assert texts[0].signature == "sig-text"

    contents = provider._convert_contents(LlmContext(messages=[UserMessage(content="hello"), assistant]))
    assert contents[-1]["parts"] == [{"text": "Hello", "thoughtSignature": "sig-text"}]


def test_thought_signatures_survive_session_serialization():
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"}, signature="sig-tool")
    text = TextContent(text="hi", signature="sig-text")
    assert content_from_dict(content_to_dict(call)) == call
    assert content_from_dict(content_to_dict(text)) == text
