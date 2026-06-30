import json


def test_cron_provider_gate_hook_selects_route_and_strips_secrets(tmp_path, monkeypatch):
    from cron import scheduler

    seen_path = tmp_path / "seen.json"
    gate_path = tmp_path / "gate.py"
    gate_path.write_text(
        "\n".join(
            [
                "import json, sys",
                "payload = json.loads(sys.stdin.read())",
                f"{str(seen_path)!r} and open({str(seen_path)!r}, 'w').write(json.dumps(payload))",
                "print(json.dumps({'admitted': True, 'selected': {'provider': 'ollama_local', 'model': 'qwen3:8b', 'base_url': 'http://127.0.0.1:11434/v1', 'source': 'fallback:0', 'tier': 'local'}}))",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_CRON_PROVIDER_GATE", str(gate_path))

    verdict = scheduler._run_cron_provider_gate(
        {"id": "plain", "name": "plain cron"},
        {
            "provider": "custom",
            "model": "claude-sonnet-4-6",
            "fallback_providers": [
                {
                    "provider": "ollama_local",
                    "model": "qwen3:8b",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "SHOULD_NOT_LEAK",
                }
            ],
        },
    )

    assert verdict["selected"]["provider"] == "ollama_local"
    seen = json.loads(seen_path.read_text(encoding="utf-8"))
    assert "SHOULD_NOT_LEAK" not in json.dumps(seen)


def test_cron_provider_gate_hook_can_be_disabled(monkeypatch):
    from cron import scheduler

    monkeypatch.setenv("HERMES_CRON_PROVIDER_GATE", "off")
    assert scheduler._run_cron_provider_gate({"id": "plain"}, {"provider": "custom"}) is None
