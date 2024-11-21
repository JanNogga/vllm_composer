# test_routes.py

import pytest
import pytest_asyncio
from pathlib import Path
from datetime import datetime
import asyncio
import respx
import httpx
from unittest.mock import patch
from httpx import AsyncClient, Response
from asgi_lifespan import LifespanManager
from httpx._transports.asgi import ASGITransport
from tests.utils import create_mock_config_from_templates, create_mock_servers
from app import create_app

# Activate respx globally before any other code
@pytest.fixture(scope='session', autouse=True)
def respx_mock():
    with respx.mock(assert_all_called=False):
        yield

# Temporary config and secrets fixture (unchanged)
@pytest.fixture
def temporary_config_and_secrets(tmp_path: Path):
    return create_mock_config_from_templates(tmp_path)

# Create a fixture to set up mocks and create the app
@pytest_asyncio.fixture
async def app(temporary_config_and_secrets):
    config_path, secrets_path, _, _ = temporary_config_and_secrets

    # Set up respx mocks and create the app before any HTTP requests are made
    async with create_mock_servers(config_path) as expected_data:
        app = create_app(config_path, secrets_path)
        # Use LifespanManager to handle startup and shutdown events
        async with LifespanManager(app):
            yield app, expected_data  # Yield both app and expected_data

# Adjust the client fixture to use ASGITransport
@pytest_asyncio.fixture
async def client(app):
    app_instance, _ = app
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

# Test function using async with for create_mock_servers
@pytest.mark.asyncio
async def test_health_and_metrics_routes(client, app):
    app_instance, expected_data = app

    # Allow some time for background tasks to run
    await asyncio.sleep(4)
    
    # Test the /health endpoint
    response = await client.get("/health")
    assert response.status_code == 200, "Health route should return 200 OK"

    # Parse response JSON
    health_data = response.json()

    # Validate response structure
    assert "servers" in health_data, "Response should include 'servers' key"
    servers = health_data["servers"]
    assert len(servers) == len(expected_data), "Number of servers should match mocked servers"

    print("Health Route Data:", health_data)

    # Validate each server's health status
    for server, expected in zip(servers, expected_data):
        assert server["url"] == expected["url"], f"URL mismatch for server {server['url']}"
        assert server["healthy"] == expected["healthy"], f"Health mismatch for server {server['url']}"
        assert server["metrics_cached"] == expected["metrics_cached"], f"Metrics mismatch for {server['url']}, expected {expected['metrics_cached']} but got {server['metrics_cached']}"
        assert server["model_cached"] == expected["model_cached"], f"Model mismatch for {server['url']}, expected {expected['model_cached']} but got {server['model_cached']}"

    # Test the /metrics endpoint
    metrics_response = await client.get("/metrics")
    assert metrics_response.status_code == 200, "Metrics route should return 200 OK"

    # Parse metrics response JSON
    metrics_data = metrics_response.json()

    # Validate response structure for metrics
    for expected in expected_data:
        assert expected["url"] in metrics_data, f"Missing metrics data for server {expected['url']}"
        if expected["healthy"]:
            assert metrics_data[expected["url"]] == expected["raw_metrics"], f"Metrics data mismatch for {expected['url']}, expected {expected['raw_metrics']} but got {metrics_data[expected['url']]}"
        else:
            assert metrics_data[expected["url"]].startswith("Error"), f"Expected error for {expected['url']} but got {metrics_data[expected['url']]}"

    print("Metrics Route Data:", metrics_data)

@pytest.mark.asyncio
async def test_v1_routes(client, app):
    app_instance, expected_data = app

    # Mock the composer methods used in proxy_request
    with patch('vllmComposer.vllmComposer.get_group_for_token') as mock_get_group_for_token, \
         patch('vllmComposer.vllmComposer.get_compatible_servers') as mock_get_compatible_servers, \
         patch('vllmComposer.vllmComposer.get_least_utilized_server') as mock_get_least_utilized_server, \
         patch('vllmComposer.vllmComposer.handle_models_request') as mock_handle_models_request:

        # Define a valid user token and group
        valid_user_token = 'valid_token'
        user_group = 'test_group'

        # Configure the mocks
        mock_get_group_for_token.return_value = user_group
        mock_get_compatible_servers.return_value = [expected_data[0]['url'], expected_data[1]['url']]
        mock_get_least_utilized_server.return_value = expected_data[0]['url']
        mock_handle_models_request.return_value = {'models': ['shared-model']}

        # Prepare headers with valid token
        headers = {'Authorization': f'Bearer {valid_user_token}'}

        # Define the path and payload
        path = 'chat/completions'
        payload = {'model': 'shared-model', 'prompt': 'Hello, world!'}

        # Mock the backend server response for non-streaming request
        backend_response = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': int(datetime.utcnow().timestamp()),
            'choices': [{
                'message': {'role': 'assistant', 'content': 'Hello! How can I assist you today?'},
                'finish_reason': 'stop',
                'index': 0
            }],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}
        }

        # Test non-streaming response
        with respx.mock(assert_all_called=False) as mock:
            mock_route = mock.post(f"{expected_data[0]['url']}/v1/{path}").mock(
                return_value=Response(200, json=backend_response)
            )

            # Send the request to the proxy endpoint
            response = await client.post(f"/v1/{path}", headers=headers, json=payload)

            # Assert the response
            assert response.status_code == 200, "Expected 200 OK from proxy endpoint"
            assert response.json() == backend_response, "Response content mismatch"

            # Ensure the backend server was called
            assert mock_route.called, "Backend server was not called"

            # Verify that the request to the backend had the correct headers
            backend_request = mock_route.calls[0].request
            assert backend_request.headers['Authorization'] != f'Bearer {valid_user_token}', \
                "Backend request should use the internal vllm_token"

        # Test streaming response
        payload['stream'] = True

        # Define async generator for streaming content
        async def streaming_content():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'

        with respx.mock(assert_all_called=False) as mock:
            mock_route = mock.post(f"{expected_data[0]['url']}/v1/{path}").mock(
                return_value=Response(200, stream=streaming_content())
            )

            response = await client.post(f"/v1/{path}", headers=headers, json=payload)
            assert response.status_code == 200, "Expected 200 OK from proxy endpoint (streaming)"

            # Read the streaming response
            content = b""
            async for chunk in response.aiter_bytes():
                content += chunk

            expected_content = b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            assert content == expected_content, "Streaming response content mismatch"

            # Ensure the backend server was called
            assert mock_route.called, "Backend server was not called for streaming response"

        # Remove 'stream' from payload to avoid side effects
        del payload['stream']

        # Test streaming response with error
        payload['stream'] = True

        # Define async generator for streaming content with an error
        async def streaming_content_with_error():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            raise Exception("Stream error occurred")

        with respx.mock(assert_all_called=False) as mock:
            mock_route = mock.post(f"{expected_data[0]['url']}/v1/{path}").mock(
                return_value=Response(200, stream=streaming_content_with_error())
            )

            response = await client.post(f"/v1/{path}", headers=headers, json=payload)
            assert response.status_code == 200, "Expected 200 OK from proxy endpoint (streaming with error). Error should be in final chunk."

            content = b""
            async for chunk in response.aiter_bytes():
                content += chunk

            # Check that the initial content is correct and the error is included in the final chunk
            expected_content_start = b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            expected_error_chunk = b'event: error\ndata: {"error": "Streaming interrupted"}\n\n'
            assert expected_content_start in content, "Initial streaming content mismatch"
            assert expected_error_chunk in content, "Error chunk mismatch in streaming response"

            # Ensure the backend server was called
            assert mock_route.called, "Backend server was not called for streaming response"

        # Remove 'stream' from payload to avoid side effects
        del payload['stream']

        # Test /v1/models endpoint
        response = await client.get("/v1/models", headers=headers)
        assert response.status_code == 200, "Expected 200 OK from /v1/models endpoint"
        assert response.json() == {'models': ['shared-model']}, "Models response content mismatch"
