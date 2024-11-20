import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from vllmComposer import vllmComposer

@pytest.mark.asyncio
async def test_get_compatible_servers_with_unhealthy_servers(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Mock servers
    composer.servers = [
        {"url": "http://server1:8000", "allowed_groups": ["group1"]},
        {"url": "http://server2:8000", "allowed_groups": ["group2"]},
        {"url": "http://server3:8000", "allowed_groups": ["group1", "group2"]},
    ]
    composer.server_health = {
        "http://server1:8000": {"healthy": True, "last_checked": None},
        "http://server2:8000": {"healthy": True, "last_checked": None},
        "http://server3:8000": {"healthy": False, "last_checked": None},  # Unhealthy server
    }

    # Mock get_model_on_server
    async def mock_get_model_on_server(server_url):
        if server_url == "http://server1:8000":
            return {"id": "model123"}
        elif server_url == "http://server2:8000":
            return {"id": "model456"}
        return None

    composer.get_model_on_server = AsyncMock(side_effect=mock_get_model_on_server)

    # Test valid user group and model with unhealthy server
    compatible_servers = await composer.get_compatible_servers(
        target_model_id="model123", user_group="group1"
    )
    assert compatible_servers == ["http://server1:8000"]  # Excludes unhealthy server3

    # Test valid user group but no matching model
    compatible_servers = await composer.get_compatible_servers(
        target_model_id="model123", user_group="group2"
    )
    assert compatible_servers == []  # server2 has a different model, server3 is unhealthy

    # Test with no compatible server
    compatible_servers = await composer.get_compatible_servers(
        target_model_id="model789", user_group="group1"
    )
    assert compatible_servers == []  # No server has the requested model

@pytest.mark.asyncio
async def test_get_least_utilized_server(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Mock compatible servers
    compatible_servers = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8002",
        "http://127.0.0.1:8004",
    ]

    # Mock get_server_load to simulate different server loads
    async def mock_get_server_load(server_url):
        server_loads = {
            "http://127.0.0.1:8000": 5.0,
            "http://127.0.0.1:8002": 0.0,
            "http://127.0.0.1:8004": 2.0,
        }
        return server_loads.get(server_url, None)

    composer.get_server_load = AsyncMock(side_effect=mock_get_server_load)

    # Test least utilized server
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    assert least_loaded_server == "http://127.0.0.1:8002"  # Server with the least load

    # Test when all servers have equal load
    async def mock_equal_load(server_url):
        return 3.0

    composer.get_server_load = AsyncMock(side_effect=mock_equal_load)
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    assert least_loaded_server == compatible_servers[0] # First one since none has been selected yet

    # Add a test for servers with equal load but different last utilization
    now = datetime.utcnow()
    composer.servers = [
        {
            "url": "http://127.0.0.1:8000",
            "allowed_groups": ["group1"],
            "last_utilization": now - timedelta(minutes=5),
        },
        {
            "url": "http://127.0.0.1:8002",
            "allowed_groups": ["group2"],
            "last_utilization": now - timedelta(minutes=10),
        },
        {
            "url": "http://127.0.0.1:8004",
            "allowed_groups": ["group3"],
            "last_utilization": now - timedelta(minutes=30),
        },
    ]

    # Debugging logs (can be removed later)
    print(f"Compatible servers: {compatible_servers}")
    print(f"Composer servers: {composer.servers}")
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    assert least_loaded_server == "http://127.0.0.1:8004"  # Longest unused

    # Test with no servers
    least_loaded_server = await composer.get_least_utilized_server([])
    assert least_loaded_server is None  # No servers available
