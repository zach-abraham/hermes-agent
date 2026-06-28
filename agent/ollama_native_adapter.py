"""OpenAI-compatible facade over Ollama's native ``/api/chat`` endpoint.

Hermes' normal Ollama route goes through Ollama's OpenAI-compatible ``/v1``
adapter.  That adapter can return normal chat text but, on the local qwen3:8b
route HAL uses, it drops tool calls.  The native ``/api/chat`` endpoint does
return tool calls, so this facade keeps Hermes' existing OpenAI-shaped agent
loop while sending the wire request to native Ollama.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaNativeError(Exception):
    """Error shape compatible with Hermes provider error handling."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response: Optional[httpx.Response] = None,
        code: str = "ollama_native_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.code = code


def ollama_native_chat_url(base_url: str | None) -> str:
    """Return the native Ollama ``/api/chat`` URL for a configured endpoint."""
    raw = (base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if not raw:
        raw = DEFAULT_OLLAMA_BASE_URL
    parsed = urlparse(raw)
    if not parsed.scheme:
        raw = f"http://{raw}"
        parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/chat"):
        new_path = path
    elif path.endswith("/api"):
        new_path = f"{path}/chat"
    elif path.endswith("/v1"):
        new_path = f"{path[:-3]}/api/chat"
    else:
        new_path = f"{path}/api/chat" if path else "/api/chat"
    return urlunparse(parsed._replace(path=new_path, params="", query="", fragment=""))


def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return str(content)


def _parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _openai_tool_call_to_ollama(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    fn = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(fn, dict):
        return None
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    return {
        "function": {
            "name": name,
            "arguments": _parse_tool_arguments(fn.get("arguments")),
        }
    }


def build_ollama_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages into native Ollama chat messages."""
    native: list[dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        item: dict[str, Any] = {
            "role": role,
            "content": _coerce_content_to_text(msg.get("content")),
        }
        if role == "assistant":
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                native_calls = [
                    converted
                    for call in calls
                    if isinstance(call, dict)
                    for converted in [_openai_tool_call_to_ollama(call)]
                    if converted is not None
                ]
                if native_calls:
                    item["tool_calls"] = native_calls
        native.append(item)
    return native


def build_ollama_tools(tools: Any) -> list[dict[str, Any]]:
    """Return Ollama-compatible tool declarations.

    Ollama accepts the OpenAI-style function-tool shape but does not need the
    OpenAI ``strict`` flag.  Removing it keeps the native request close to the
    live-local probe that passed on qwen3:8b.
    """
    if not isinstance(tools, list):
        return []
    native_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        copied = deepcopy(tool)
        fn = copied.get("function")
        if isinstance(fn, dict):
            fn.pop("strict", None)
        native_tools.append(copied)
    return native_tools


def build_ollama_request(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Any = None,
    stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Any = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": build_ollama_messages(messages),
        "stream": bool(stream),
    }
    native_tools = build_ollama_tools(tools)
    if native_tools:
        payload["tools"] = native_tools

    # HAL's local qwen3 probe was validated with thinking disabled.  Keep that
    # as the default unless a caller explicitly overrides it in extra_body.
    payload["think"] = False
    if isinstance(extra_body, dict) and "think" in extra_body:
        payload["think"] = bool(extra_body["think"])

    options: Dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if stop:
        options["stop"] = stop if isinstance(stop, list) else [str(stop)]
    if isinstance(extra_body, dict):
        if isinstance(extra_body.get("num_ctx"), int):
            options["num_ctx"] = int(extra_body["num_ctx"])
        extra_options = extra_body.get("options")
        if isinstance(extra_options, dict):
            options.update(extra_options)
    if options:
        payload["options"] = options
    return payload


def _tool_call_from_ollama(call: dict[str, Any], index: int) -> SimpleNamespace | None:
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict):
        return None
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    arguments = fn.get("arguments")
    try:
        args_str = json.dumps(
            arguments if isinstance(arguments, dict) else _parse_tool_arguments(arguments),
            ensure_ascii=False,
            sort_keys=True,
        )
    except (TypeError, ValueError):
        args_str = "{}"
    return SimpleNamespace(
        id=str(call.get("id") or f"call_ollama_{uuid.uuid4().hex[:12]}"),
        type="function",
        index=index,
        function=SimpleNamespace(name=name, arguments=args_str),
    )


def _map_done_reason(reason: Any) -> str:
    normalized = str(reason or "").strip().lower()
    if normalized in {"length", "max_tokens"}:
        return "length"
    if normalized in {"tool_calls", "tool_call"}:
        return "tool_calls"
    return "stop"


def translate_ollama_response(payload: Dict[str, Any], model: str) -> SimpleNamespace:
    message_obj = payload.get("message") if isinstance(payload, dict) else {}
    if not isinstance(message_obj, dict):
        message_obj = {}

    raw_calls = message_obj.get("tool_calls")
    tool_calls: list[SimpleNamespace] = []
    if isinstance(raw_calls, list):
        for index, call in enumerate(raw_calls):
            if isinstance(call, dict):
                converted = _tool_call_from_ollama(call, index)
                if converted is not None:
                    tool_calls.append(converted)

    finish_reason = "tool_calls" if tool_calls else _map_done_reason(payload.get("done_reason"))
    prompt_tokens = int(payload.get("prompt_eval_count") or 0)
    completion_tokens = int(payload.get("eval_count") or 0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    message = SimpleNamespace(
        role="assistant",
        content=message_obj.get("content") or None,
        tool_calls=tool_calls or None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=f"chatcmpl-ollama-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


class _OllamaStreamChunk(SimpleNamespace):
    pass


def _make_stream_chunk(
    *,
    model: str,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> _OllamaStreamChunk:
    delta = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _OllamaStreamChunk(
        id=f"chatcmpl-ollama-{uuid.uuid4().hex[:12]}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=None,
    )


class _OllamaChatCompletions:
    def __init__(self, client: "OllamaNativeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _OllamaChatNamespace:
    def __init__(self, client: "OllamaNativeClient"):
        self.completions = _OllamaChatCompletions(client)


class OllamaNativeClient:
    """Minimal OpenAI-SDK-compatible facade over Ollama native chat."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or ""
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.chat_url = ollama_native_chat_url(self.base_url)
        self._default_headers = dict(default_headers or {})
        self.chat = _OllamaChatNamespace(self)
        self.is_closed = False
        self._http = http_client or httpx.Client(
            timeout=timeout or httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)
        )

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-agent (ollama-native)",
        }
        if self.api_key and self.api_key != "no-key-required":
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._default_headers)
        return headers

    def _create_chat_completion(
        self,
        *,
        model: str = "qwen3:8b",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        if stream:
            return self._stream_completion(
                model=model,
                messages=messages or [],
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                extra_body=extra_body,
                timeout=timeout,
            )

        payload = build_ollama_request(
            model=model,
            messages=messages or [],
            tools=tools,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            extra_body=extra_body,
        )
        response = self._http.post(
            self.chat_url,
            json=payload,
            headers=self._headers(),
            timeout=timeout,
        )
        if response.status_code != 200:
            body = ""
            try:
                body = response.text
            except Exception:
                body = ""
            raise OllamaNativeError(
                f"Ollama native chat returned HTTP {response.status_code}: {body[:500]}",
                status_code=response.status_code,
                response=response,
                code=f"ollama_native_http_{response.status_code}",
            )
        try:
            loaded = response.json()
        except ValueError as exc:
            raise OllamaNativeError(
                f"Invalid JSON from Ollama native chat: {exc}",
                status_code=response.status_code,
                response=response,
                code="ollama_native_invalid_json",
            ) from exc
        if not isinstance(loaded, dict):
            raise OllamaNativeError("Ollama native chat returned a non-object JSON payload")
        return translate_ollama_response(loaded, model=model)

    def _stream_completion(self, **kwargs: Any) -> Iterator[_OllamaStreamChunk]:
        """Synthetic stream built from one native non-streaming response.

        Hermes crons currently run non-streaming, but gateway/CLI code may ask
        any chat client for a stream.  Ollama's native streaming tool-call deltas
        are provider-specific, so this keeps the adapter safe by making one
        non-streaming native call and yielding OpenAI-shaped chunks.
        """
        response = self._create_chat_completion(stream=False, **kwargs)
        choice = response.choices[0]
        message = choice.message
        if getattr(message, "content", None):
            yield _make_stream_chunk(model=response.model, content=message.content)
        if getattr(message, "tool_calls", None):
            yield _make_stream_chunk(model=response.model, tool_calls=message.tool_calls)
        finish_chunk = _make_stream_chunk(model=response.model, finish_reason=choice.finish_reason)
        finish_chunk.usage = response.usage
        yield finish_chunk
