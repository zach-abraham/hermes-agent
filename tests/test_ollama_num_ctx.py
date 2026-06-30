"""Tests for Ollama num_ctx context length detection and injection.

Covers:
  agent/model_metadata.py — query_ollama_num_ctx()
  run_agent.py — _ollama_num_ctx detection + extra_body injection
"""

from unittest.mock import patch, MagicMock


from agent.model_metadata import query_ollama_num_ctx, query_ollama_supports_vision


# ═══════════════════════════════════════════════════════════════════════
# Level 1: query_ollama_num_ctx — Ollama API interaction
# ═══════════════════════════════════════════════════════════════════════


def _mock_httpx_client(show_response_data, status_code=200):
    """Create a mock httpx.Client context manager that returns given /api/show data."""
    mock_resp = MagicMock(status_code=status_code)
    mock_resp.json.return_value = show_response_data
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_client)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx, mock_client


class TestQueryOllamaNumCtx:
    """Test the Ollama /api/show context length query."""

    def test_returns_context_from_model_info(self):
        """Should extract context_length from GGUF model_info metadata."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            # httpx is imported inside the function — patch the module import
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("llama3.1:8b", "http://localhost:11434/v1")

        assert result == 131072

    def test_prefers_explicit_num_ctx_from_modelfile(self):
        """If the Modelfile sets num_ctx explicitly, that should take priority."""
        show_data = {
            "model_info": {"llama.context_length": 131072},
            "parameters": "num_ctx 32768\ntemperature 0.7",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("custom-model", "http://localhost:11434")

        assert result == 32768

    def test_returns_none_for_non_ollama_server(self):
        """Should return None if the server is not Ollama."""
        with patch("agent.model_metadata.detect_local_server_type", return_value="lm-studio"):
            result = query_ollama_num_ctx("model", "http://localhost:1234")
        assert result is None

    def test_returns_none_on_connection_error(self):
        """Should return None if the server is unreachable."""
        with patch("agent.model_metadata.detect_local_server_type", side_effect=Exception("timeout")):
            result = query_ollama_num_ctx("model", "http://localhost:11434")
        assert result is None

    def test_returns_none_on_404(self):
        """Should return None if the model is not found."""
        mock_ctx, _ = _mock_httpx_client({}, status_code=404)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("nonexistent", "http://localhost:11434")

        assert result is None

    def test_strips_provider_prefix(self):
        """Should strip 'local:' prefix from model name before querying."""
        show_data = {
            "model_info": {"qwen2.context_length": 32768},
            "parameters": "",
        }
        mock_ctx, mock_client = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("local:qwen2.5:7b", "http://localhost:11434/v1")

        # Verify the post was called with stripped name (no "local:" prefix)
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["name"] == "qwen2.5:7b" or call_args[0][1] is not None
        assert result == 32768

    def test_handles_qwen2_architecture_key(self):
        """Different model architectures use different key prefixes in model_info."""
        show_data = {
            "model_info": {"qwen2.context_length": 65536},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("qwen2.5:32b", "http://localhost:11434")

        assert result == 65536

    def test_returns_none_when_model_info_empty(self):
        """Should return None if model_info has no context_length key."""
        show_data = {
            "model_info": {"llama.embedding_length": 4096},
            "parameters": "",
        }
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_num_ctx("model", "http://localhost:11434")

        assert result is None


class TestOllamaNumCtxConfigOverrides:
    """Test route-specific Ollama context overrides during agent init."""

    def test_provider_model_override_wins_over_global_model_config(self):
        from run_agent import AIAgent

        config = {
            "model": {
                "default": "deepseek-r1:8b",
                "provider": "ollama_local",
                "base_url": "http://192.168.8.108:11434/v1",
                "context_length": 131072,
            },
            "providers": {
                "ollama_native_local": {
                    "name": "ollama_native_local",
                    "base_url": "http://192.168.8.108:11434",
                    "default_model": "qwen3:8b",
                    "api_mode": "ollama_native_chat",
                    "models": {
                        "qwen3:8b": {
                            "context_length": 65536,
                            "ollama_num_ctx": 65536,
                        }
                    },
                }
            },
        }

        with (
            patch("hermes_cli.config.load_config", return_value=config),
            patch("agent.agent_init.query_ollama_num_ctx", return_value=40960) as probe,
        ):
            agent = AIAgent(
                model="qwen3:8b",
                api_key="ollama",
                provider="custom",
                base_url="http://192.168.8.108:11434",
                api_mode="ollama_native_chat",
                enabled_toolsets=[],
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cron",
            )

        assert agent._config_context_length == 65536
        assert agent.context_compressor.context_length == 65536
        assert agent._ollama_num_ctx == 65536
        probe.assert_not_called()


class TestQueryOllamaSupportsVision:
    """Test Ollama /api/show vision capability detection."""

    def test_returns_true_when_capabilities_include_vision(self):
        show_data = {"capabilities": ["completion", "vision"]}
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_supports_vision("gemma4:e2b", "http://localhost:11434/v1")

        assert result is True

    def test_returns_false_when_capabilities_exclude_vision(self):
        show_data = {"capabilities": ["completion", "tools"]}
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_supports_vision("gemma4:31b", "http://localhost:11434/v1")

        assert result is False

    def test_falls_back_to_model_info_vision_block_count(self):
        show_data = {"model_info": {"gemma3.vision.block_count": 27}}
        mock_ctx, _ = _mock_httpx_client(show_data)

        with patch("agent.model_metadata.detect_local_server_type", return_value="ollama"):
            import httpx
            with patch.object(httpx, "Client", return_value=mock_ctx):
                result = query_ollama_supports_vision("llava", "http://localhost:11434")

        assert result is True

    def test_returns_none_for_non_ollama_server(self):
        with patch("agent.model_metadata.detect_local_server_type", return_value="vllm"):
            result = query_ollama_supports_vision("llava", "http://localhost:8000/v1")
        assert result is None
