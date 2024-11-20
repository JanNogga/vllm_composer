# test_routes.py

import pytest
import pytest_asyncio
from pathlib import Path
from tests.utils import create_mock_config_from_templates, create_mock_servers
from app import create_app
import asyncio
import respx
from httpx import AsyncClient, Response
from asgi_lifespan import LifespanManager
from httpx._transports.asgi import ASGITransport

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
async def test_health_route(client, app):
    app_instance, expected_data = app

    # Allow some time for background tasks to run
    await asyncio.sleep(4)
    
    # Call the /health endpoint
    response = await client.get("/health")
    assert response.status_code == 200, "Health route should return 200 OK"

    # Parse response JSON
    health_data = response.json()

    # Validate response structure
    assert "servers" in health_data, "Response should include 'servers' key"
    servers = health_data["servers"]
    assert len(servers) == len(expected_data), "Number of servers should match mocked servers"

    print(health_data)

    # Validate each server's health status
    for server, expected in zip(servers, expected_data):
        assert server["url"] == expected["url"], f"URL mismatch for server {server['url']}"
        assert server["healthy"] == expected["healthy"], f"Health mismatch for server {server['url']}"
        assert server["metrics_cached"] == expected["metrics_cached"], f"Metrics mismatch for {server['url']}, expected {expected['metrics_cached']} but got {server['metrics_cached']}"
        assert server["model_cached"] == expected["model_cached"], f"Model mismatch for {server['url']}, expected {expected['model_cached']} but got {server['model_cached']}"
