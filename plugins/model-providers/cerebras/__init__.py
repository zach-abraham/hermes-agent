"""Cerebras provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

cerebras = ProviderProfile(
    name="cerebras",
    env_vars=("CEREBRAS_API_KEY",),
    display_name="Cerebras",
    description="Cerebras Inference - OpenAI-compatible inference",
    signup_url="https://cloud.cerebras.ai/",
    fallback_models=(
        "gpt-oss-120b",
        "qwen-3-coder-480b",
        "llama-3.3-70b",
    ),
    base_url="https://api.cerebras.ai/v1",
)

register_provider(cerebras)
