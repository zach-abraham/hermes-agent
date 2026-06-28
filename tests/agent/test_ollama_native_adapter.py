"""Tests for the native Ollama chat adapter."""

from __future__ import annotations

import json


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def test_ollama_native_chat_url_maps_openai_compat_base_to_api_chat():
    from agent.ollama_native_adapter import ollama_native_chat_url

    assert (
        ollama_native_chat_url("http://100.68.160.14:11434/v1")
        == "http://100.68.160.14:11434/api/chat"
    )
    assert (
        ollama_native_chat_url("http://127.0.0.1:11434/api")
        == "http://127.0.0.1:11434/api/chat"
    )


def test_build_ollama_request_converts_tool_replay_and_strips_strict():
    from agent.ollama_native_adapter import build_ollama_request

    request = build_ollama_request(
        model="qwen3:8b",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "hal_probe_echo",
                            "arguments": '{"message":"ok","mode":"ping"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "hal_probe_echo",
                "content": '{"ok":true}',
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "hal_probe_echo",
                    "strict": True,
                    "parameters": {"type": "object"},
                },
            }
        ],
        max_tokens=64,
        temperature=0,
        extra_body={"num_ctx": 131072},
    )

    assert request["stream"] is False
    assert request["think"] is False
    assert request["options"]["num_predict"] == 64
    assert request["options"]["num_ctx"] == 131072
    assert request["messages"][0]["tool_calls"] == [
        {
            "function": {
                "name": "hal_probe_echo",
                "arguments": {"message": "ok", "mode": "ping"},
            }
        }
    ]
    assert request["messages"][1] == {"role": "tool", "content": '{"ok":true}'}
    assert "strict" not in request["tools"][0]["function"]


def test_translate_ollama_response_surfaces_openai_tool_calls():
    from agent.ollama_native_adapter import translate_ollama_response

    response = translate_ollama_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "hal_probe_echo",
                            "arguments": {"mode": "ping", "message": "ok"},
                        }
                    }
                ],
            },
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 2,
        },
        model="qwen3:8b",
    )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls[0].function.name == "hal_probe_echo"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {
        "message": "ok",
        "mode": "ping",
    }
    assert response.usage.total_tokens == 5


def test_native_client_posts_to_api_chat_with_openai_shaped_result():
    from agent.ollama_native_adapter import OllamaNativeClient

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            recorded["headers"] = headers
            return DummyResponse(
                payload={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "hal_probe_echo",
                                    "arguments": {"message": "ok", "mode": "ping"},
                                }
                            }
                        ],
                    },
                    "done_reason": "stop",
                }
            )

        def close(self):
            return None

    client = OllamaNativeClient(
        api_key="no-key-required",
        base_url="http://100.68.160.14:11434/v1",
        http_client=DummyHTTP(),
    )
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "call the tool"}],
        tools=[{"type": "function", "function": {"name": "hal_probe_echo"}}],
    )

    assert recorded["url"] == "http://100.68.160.14:11434/api/chat"
    assert "Authorization" not in recorded["headers"]
    assert recorded["json"]["model"] == "qwen3:8b"
    assert response.choices[0].message.tool_calls[0].function.name == "hal_probe_echo"


def test_ollama_native_transport_and_runtime_mode_are_registered():
    from agent.transports import get_transport
    from hermes_cli.runtime_provider import _parse_api_mode

    transport = get_transport("ollama_native_chat")

    assert transport is not None
    assert transport.api_mode == "ollama_native_chat"
    assert _parse_api_mode("ollama_native_chat") == "ollama_native_chat"


def test_ollama_native_transport_threads_detected_num_ctx():
    from agent.transports import get_transport

    transport = get_transport("ollama_native_chat")
    kwargs = transport.build_kwargs(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        ollama_num_ctx=131072,
    )

    assert kwargs["extra_body"]["num_ctx"] == 131072
