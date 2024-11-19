import pytest
from datetime import datetime, timedelta
from vllmComposer import vllmComposer

@pytest.mark.asyncio
async def test_handle_server_failure(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)
    
    # Setup a mock server URL
    server_url = "http://localhost:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.server_health[server_url] = {"healthy": True, "last_checked": None}
    
    # Simulate failures
    for _ in range(composer.max_failures):
        await composer.handle_server_failure(server_url)
    
    # Assert that the failure count has reached max_failures
    assert composer.failure_counts[server_url] == composer.max_failures
    
    # Assert that the circuit breaker timeout is set correctly
    assert server_url in composer.circuit_breaker_timeout
    cooldown_end_time = composer.circuit_breaker_timeout[server_url]
    assert cooldown_end_time > datetime.utcnow()
    
    # Assert that the server is marked as unhealthy
    assert composer.server_health[server_url]["healthy"] is False

@pytest.mark.asyncio
async def test_is_server_healthy(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Setup a mock server
    server_url = "http://localhost:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.server_health[server_url] = {"healthy": True, "last_checked": None}
    
    # Scenario 1: Server is healthy and not in circuit breaker
    is_healthy = await composer.is_server_healthy(server_url)
    assert is_healthy is True

    # Scenario 2: Server is unhealthy
    composer.server_health[server_url] = {"healthy": False, "last_checked": datetime.utcnow()}
    is_healthy = await composer.is_server_healthy(server_url)
    assert is_healthy is False

    # Scenario 3: Server is in circuit breaker timeout
    composer.server_health[server_url] = {"healthy": True, "last_checked": datetime.utcnow()}
    composer.circuit_breaker_timeout[server_url] = datetime.utcnow() + timedelta(seconds=30)
    is_healthy = await composer.is_server_healthy(server_url)
    assert is_healthy is False

    # Scenario 4: Circuit breaker timeout has passed
    composer.circuit_breaker_timeout[server_url] = datetime.utcnow() - timedelta(seconds=30)
    is_healthy = await composer.is_server_healthy(server_url)
    assert is_healthy is True

@pytest.mark.asyncio
async def test_check_circuit_breaker_with_cooldown(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    server_url = "http://server1:8000"
    composer.circuit_breaker_timeout[server_url] = datetime.utcnow() + timedelta(seconds=10)

    assert not await composer.check_circuit_breaker(server_url)  # Server in cooldown

    # Test server not in cooldown
    composer.circuit_breaker_timeout[server_url] = datetime.utcnow() - timedelta(seconds=10)
    assert await composer.check_circuit_breaker(server_url)  # Cooldown expired