"""Photon adapter resilience to transient Spectrum/Envoy upstream overflow.

Covers the three behaviors that let the adapter ride through a Photon
"reset reason: overflow" event instead of degrading delivery and silently
dying (issue #50185):

  1. ``_is_retryable_error`` classifies the Envoy/sidecar overflow strings as
     retryable so ``_send_with_retry`` actually engages its backoff loop.
  2. ``send_typing`` is rate-gated per chat, and ``stop_typing`` resets the
     gate so the next turn's typing indicator fires immediately.
  3. ``_supervise_sidecar`` detects an unexpected sidecar exit and raises a
     ``retryable=True`` fatal so the gateway reconnect watcher revives the
     platform — instead of returning silently and leaving ``_inbound_loop``
     spinning against a dead port.
  4. ``_monitor_sidecar_health`` first restarts the sidecar in-place when
     ``/healthz`` reports degraded upstream stream health, and falls back to
     the retryable fatal path only if that local repair fails.

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- Gap 1: retryable classification of overflow errors ---------------------

@pytest.mark.parametrize(
    "error",
    [
        "UNAVAILABLE: internal sidecar error",
        "upstream connect error or disconnect/reset before headers",
        "reset reason: overflow",
        # Case-insensitive: real strings arrive with mixed case.
        "Internal Sidecar Error",
    ],
)
def test_overflow_strings_classified_retryable(error: str) -> None:
    assert PhotonAdapter._is_retryable_error(error) is True


def test_unrelated_error_not_retryable() -> None:
    # A genuine permanent failure must NOT be retried.
    assert PhotonAdapter._is_retryable_error("400 bad request: invalid spaceId") is False
    assert PhotonAdapter._is_retryable_error(None) is False


def test_base_network_patterns_still_match() -> None:
    # The override delegates to the base classifier first, so generic
    # network strings keep working.
    assert PhotonAdapter._is_retryable_error("ConnectError: connection refused") is True


# -- Gap 2: typing-indicator cooldown ---------------------------------------

@pytest.mark.asyncio
async def test_typing_cooldown_suppresses_rapid_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[Dict[str, Any]] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload)
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # First call fires; immediate repeats are suppressed by the cooldown.
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-1")

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_typing_cooldown_is_per_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls: list[str] = []

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        calls.append(payload["spaceId"])
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # Different chats have independent cooldowns.
    await adapter.send_typing("chat-1")
    await adapter.send_typing("chat-2")

    assert calls == ["chat-1", "chat-2"]


@pytest.mark.asyncio
async def test_stop_typing_resets_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    starts = 0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        nonlocal starts
        if payload.get("state") == "start":
            starts += 1
        return {"ok": True}

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)

    # A start, then a stop (end of turn), then a start for the next turn must
    # fire immediately — the cooldown only suppresses rapid consecutive starts
    # without an intervening stop.
    await adapter.send_typing("chat-1")
    await adapter.stop_typing("chat-1")
    await adapter.send_typing("chat-1")

    assert starts == 2


# -- Gap 3: sidecar crash detection -----------------------------------------

class _EofStdout:
    """A proc.stdout whose readline() reports immediate EOF (dead sidecar)."""

    def readline(self) -> bytes:
        return b""


class _DeadProc:
    """Minimal subprocess.Popen stand-in for a sidecar that has exited."""

    def __init__(self, exit_code: int = 1) -> None:
        self.stdout = _EofStdout()
        self.stdin = None
        self._exit_code = exit_code

    def poll(self) -> int:
        return self._exit_code


@pytest.mark.asyncio
async def test_unexpected_sidecar_exit_raises_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # Simulate a live session whose sidecar then dies underneath it.
    adapter._inbound_running = True

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=137))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "SIDECAR_CRASHED"
    # retryable=True routes the platform into the reconnect watcher rather
    # than crashing the whole gateway.
    assert adapter.fatal_error_retryable is True
    assert adapter._running is False
    assert notified == [True]


@pytest.mark.asyncio
async def test_clean_shutdown_does_not_raise_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    # disconnect() sets _inbound_running = False before stopping the sidecar,
    # so the detection block must NOT fire on a clean shutdown.
    adapter._inbound_running = False

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)

    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._supervise_sidecar(_DeadProc(exit_code=0))  # type: ignore[arg-type]

    assert adapter.has_fatal_error is False
    assert notified == []


@pytest.mark.asyncio
async def test_degraded_stream_health_restarts_sidecar_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 120000,
                "lastIssue": "[spectrum.stream] stream interrupted; reconnecting",
            },
        }

    calls: list[str] = []

    async def _fake_stop() -> None:
        calls.append("stop")

    async def _fake_start() -> None:
        calls.append("start")
        adapter._inbound_running = False

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_stop_sidecar", _fake_stop)
    monkeypatch.setattr(adapter, "_start_sidecar", _fake_start)
    monkeypatch.setattr(
        adapter,
        "_write_stream_recovery_state",
        lambda **_: None,
    )

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is False
    assert calls == ["stop", "start"]


@pytest.mark.asyncio
async def test_degraded_stream_restart_budget_suppresses_repeat_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._stream_restart_cooldown_seconds = 600.0

    writes: list[Dict[str, Any]] = []

    def _capture_state(
        *,
        action: str,
        failure_class: str,
        message: str,
        now: float | None = None,
    ) -> None:
        writes.append(
            {
                "action": action,
                "failure_class": failure_class,
                "message": message,
                "state": adapter._stream_recovery_state,
            }
        )

    calls: list[str] = []

    async def _fake_stop() -> None:
        calls.append("stop")

    async def _fake_start() -> None:
        calls.append("start")

    monkeypatch.setattr(adapter, "_write_stream_recovery_state", _capture_state)
    monkeypatch.setattr(adapter, "_stop_sidecar", _fake_stop)
    monkeypatch.setattr(adapter, "_start_sidecar", _fake_start)

    message = (
        "Photon upstream stream degraded: RetryableStreamError: Live stream ended"
    )

    assert await adapter._restart_sidecar_for_degraded_stream(message) is True
    assert calls == ["stop", "start"]

    assert await adapter._restart_sidecar_for_degraded_stream(message) is True
    assert calls == ["stop", "start"]
    assert writes[0]["action"] == "restart_allowed"
    assert writes[0]["failure_class"] == "upstream_reset"
    assert writes[1]["action"] == "restart_suppressed"
    assert writes[1]["state"] == "open"


def test_stream_recovered_resets_failure_state(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._stream_recovery_state = "open"
    adapter._stream_failure_count = 3

    writes: list[Dict[str, Any]] = []

    def _capture_state(
        *,
        action: str,
        failure_class: str,
        message: str,
        now: float | None = None,
    ) -> None:
        writes.append(
            {
                "action": action,
                "failure_class": failure_class,
                "message": message,
                "state": adapter._stream_recovery_state,
            }
        )

    monkeypatch.setattr(adapter, "_write_stream_recovery_state", _capture_state)

    adapter._mark_stream_recovered()

    assert adapter._stream_recovery_state == "closed"
    assert adapter._stream_failure_count == 0
    assert writes == [
        {
            "action": "stream_recovered",
            "failure_class": "none",
            "message": "Photon upstream stream healthy",
            "state": "closed",
        }
    ]


@pytest.mark.asyncio
async def test_degraded_stream_health_falls_back_to_retryable_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._inbound_running = True
    adapter._sidecar_health_interval = 0.0

    async def _fake_call(path: str, payload: Dict[str, Any]) -> Any:
        assert path == "/healthz"
        return {
            "ok": True,
            "stream": {
                "ok": False,
                "state": "degraded",
                "degradedForMs": 120000,
                "lastIssue": "[spectrum.stream] stream interrupted; reconnecting",
            },
        }

    async def _fake_restart(message: str) -> bool:
        return False

    notified: list[bool] = []

    async def _fake_notify() -> None:
        notified.append(True)
        adapter._inbound_running = False

    monkeypatch.setattr(adapter, "_sidecar_call", _fake_call)
    monkeypatch.setattr(adapter, "_restart_sidecar_for_degraded_stream", _fake_restart)
    monkeypatch.setattr(adapter, "_notify_fatal_error", _fake_notify)

    await adapter._monitor_sidecar_health()

    assert adapter.has_fatal_error is True
    assert adapter.fatal_error_code == "UPSTREAM_STREAM_DEGRADED"
    assert adapter.fatal_error_retryable is True
    assert notified == [True]
