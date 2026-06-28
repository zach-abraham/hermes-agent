"""Tests for native-tool cron allowlist routing."""

from cron.scheduler import _native_tool_cron_allowlist_route


def _routing(enabled=True):
    return {
        "native_tool_cron_allowlist": {
            "enabled": enabled,
            "provider": "ollama_native_local",
            "model": "gemma4:e4b",
            "adapter": "ollama_native_chat",
            "job_ids": ["job-1"],
            "job_names": ["lane-advancer"],
        }
    }


def test_native_tool_allowlist_routes_matching_unpinned_agent_job():
    route = _native_tool_cron_allowlist_route(
        {"id": "job-1", "name": "lane-advancer", "no_agent": False},
        _routing(),
    )

    assert route == {
        "provider": "ollama_native_local",
        "model": "gemma4:e4b",
        "adapter": "ollama_native_chat",
    }


def test_native_tool_allowlist_ignores_disabled_config():
    assert (
        _native_tool_cron_allowlist_route(
            {"id": "job-1", "name": "lane-advancer"},
            _routing(enabled=False),
        )
        is None
    )


def test_native_tool_allowlist_ignores_no_agent_jobs():
    assert (
        _native_tool_cron_allowlist_route(
            {"id": "job-1", "name": "lane-advancer", "no_agent": True},
            _routing(),
        )
        is None
    )


def test_native_tool_allowlist_ignores_pinned_provider_jobs():
    assert (
        _native_tool_cron_allowlist_route(
            {"id": "job-1", "name": "lane-advancer", "provider": "sambanova"},
            _routing(),
        )
        is None
    )


def test_native_tool_allowlist_requires_native_adapter_name():
    routing = _routing()
    routing["native_tool_cron_allowlist"]["adapter"] = "chat_completions"

    assert (
        _native_tool_cron_allowlist_route(
            {"id": "job-1", "name": "lane-advancer"},
            routing,
        )
        is None
    )
