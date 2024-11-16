import pytest
import logging
from vllmComposer import vllmComposer
from pathlib import Path
from datetime import timedelta

@pytest.fixture
def mock_config_and_secrets(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    secrets_path = tmp_path / "secrets.yml"

    # Mock config.yml content
    config_content = """
    vllm_hosts:
      - hostname: "localhost"
        ports: {start: 8000, end: 8001}
        allowed_groups: ["group1"]
    app_settings:
      model_owner: "test-owner"
      max_failures: 3
      cooldown_period_minutes: 5
      log_level: "warning"
    """
    config_path.write_text(config_content)

    # Mock secrets.yml content
    secrets_content = """
    groups:
      group1: ["token1", "token2"]
      group2: ["token3"]
    vllm_token: "dummy_token"
    """
    secrets_path.write_text(secrets_content)

    return str(config_path), str(secrets_path)


def test_load_config(mock_config_and_secrets):
    # Use the fixture to get the mock config and secrets paths
    config_path, secrets_path = mock_config_and_secrets

    # Instantiate the composer with the mock config and secrets paths
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Validate servers list
    expected_servers = [
        {"url": "http://localhost:8000", "allowed_groups": ["group1"]},
        {"url": "http://localhost:8001", "allowed_groups": ["group1"]},
    ]
    assert composer.servers == expected_servers

    # Validate app settings
    assert composer.model_owner == "test-owner"
    assert composer.max_failures == 3
    assert composer.cooldown_period == timedelta(minutes=5)

    # Validate server health initialization
    expected_health = {server["url"]: {"healthy": True, "last_checked": None} for server in expected_servers}
    assert composer.server_health == expected_health

    # Validate logger settings
    assert composer.logger.level == logging.WARNING

def test_load_secrets(mock_config_and_secrets):
    # Use the fixture to get the mock config and secrets paths
    config_path, secrets_path = mock_config_and_secrets

    # Instantiate the composer with the mock paths
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Validate group tokens
    expected_group_tokens = {
        "group1": ["token1", "token2"],
        "group2": ["token3"],
    }
    assert composer.group_tokens == expected_group_tokens

    # Validate VLLM token
    assert composer.vllm_token == "dummy_token"

def test_get_group_for_token(mock_config_and_secrets):
    config_path, secrets_path = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    assert composer.get_group_for_token("token1") == "group1"
    assert composer.get_group_for_token("token3") == "group2"
    assert composer.get_group_for_token("invalid_token") is None
