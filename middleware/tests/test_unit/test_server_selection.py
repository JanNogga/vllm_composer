import pytest
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
        "http://server1:8000",
        "http://server2:8000",
        "http://server3:8000",
    ]

    # Mock get_server_load to simulate different server loads
    async def mock_get_server_load(server_url):
        server_loads = {
            "http://server1:8000": 5.0,
            "http://server2:8000": 2.0,
            "http://server3:8000": 0.0,  # Least load
        }
        return server_loads.get(server_url, None)

    composer.get_server_load = AsyncMock(side_effect=mock_get_server_load)

    # Test least utilized server
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    assert least_loaded_server == "http://server3:8000"  # Server with the least load

    # Test when all servers have equal load
    async def mock_equal_load(server_url):
        return 3.0

    composer.get_server_load = AsyncMock(side_effect=mock_equal_load)
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    assert least_loaded_server == compatible_servers[0]  # Defaults to the first in case of a tie

    # Test with no servers
    least_loaded_server = await composer.get_least_utilized_server([])
    assert least_loaded_server is None  # No servers available
