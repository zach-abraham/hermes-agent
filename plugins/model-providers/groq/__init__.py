"""Groq provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

groq = ProviderProfile(
    name="groq",
    aliases=("groqcloud",),
    env_vars=("GROQ_API_KEY",),
    display_name="Groq",
    description="Groq Cloud - OpenAI-compatible inference",
    signup_url="https://console.groq.com/keys",
    fallback_models=(
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ),
    base_url="https://api.groq.com/openai/v1",
)

register_provider(groq)
