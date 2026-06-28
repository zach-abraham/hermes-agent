"""Transport registration for Ollama native chat.

The native Ollama client exposes an OpenAI-shaped ``chat.completions`` facade,
so the existing chat-completions transport can build and normalize the agent
loop payloads.  This thin subclass only gives the route an explicit api_mode.
"""

from typing import Any

from agent.transports import register_transport
from agent.transports.chat_completions import ChatCompletionsTransport


class OllamaNativeChatTransport(ChatCompletionsTransport):
    @property
    def api_mode(self) -> str:
        return "ollama_native_chat"

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        kwargs = super().build_kwargs(model=model, messages=messages, tools=tools, **params)
        num_ctx = params.get("ollama_num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["num_ctx"] = num_ctx
            kwargs["extra_body"] = extra_body
        return kwargs


register_transport("ollama_native_chat", OllamaNativeChatTransport)
