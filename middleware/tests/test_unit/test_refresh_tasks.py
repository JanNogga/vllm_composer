import pytest
import respx
from httpx import Response
from unittest.mock import AsyncMock, patch
from vllmComposer import vllmComposer

@pytest.mark.asyncio
async def test_refresh_models(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Mock servers
    composer.servers = [
        {"url": "http://server1:8000", "allowed_groups": ["group1"]},
        {"url": "http://server2:8000", "allowed_groups": ["group2"]},
    ]

    # Mock get_model_on_server
    async def mock_get_model_on_server(server_url):
        if server_url == "http://server1:8000":
            return {"id": "model123"}
        elif server_url == "http://server2:8000":
            return {"id": "model456"}
        return None

    composer.get_model_on_server = AsyncMock(side_effect=mock_get_model_on_server)

    # Call the original method directly (bypassing the decorator)
    await composer.refresh_models.__wrapped__(composer)

    # Validate that get_model_on_server was called for each server
    composer.get_model_on_server.assert_any_await("http://server1:8000")
    composer.get_model_on_server.assert_any_await("http://server2:8000")


@pytest.mark.asyncio
@respx.mock
async def test_refresh_metrics(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Mock servers
    composer.servers = [
        {"url": "http://server1:8000", "allowed_groups": ["group1"]},
        {"url": "http://server2:8000", "allowed_groups": ["group2"]},
    ]

    # Set up respx to mock the /metrics endpoint
    server1_route = respx.get("http://server1:8000/metrics").mock(
        return_value=Response(
            200, text="vllm:num_requests_running 5\nvllm:num_requests_waiting 3\n"
        )
    )
    server2_route = respx.get("http://server2:8000/metrics").mock(
        return_value=Response(
            200, text="vllm:num_requests_running 2\nvllm:num_requests_waiting 1\n"
        )
    )

    # Call the original method directly (bypassing the decorator)
    await composer.refresh_metrics.__wrapped__(composer)

    # Validate that metrics were fetched for each server
    assert server1_route.called
    assert server2_route.called
    assert respx.calls.call_count == 2  # Ensure both servers were called