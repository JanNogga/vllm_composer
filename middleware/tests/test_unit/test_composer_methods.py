import pytest
from vllmComposer import vllmComposer
from datetime import timedelta
import logging

from tests.utils import create_mock_config_from_templates

@pytest.fixture
def mock_config_and_secrets(tmp_path):
  return create_mock_config_from_templates(tmp_path)


def test_load_config(mock_config_and_secrets):
    config_path, secrets_path, config_dict, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Validate servers list
    expected_servers = [
        {"url": f"http://{host['hostname']}:{port}", "allowed_groups": host["allowed_groups"]}
        for host in config_dict["vllm_hosts"]
        for port in range(host["ports"]["start"], host["ports"]["end"] + 1)
    ]
    assert composer.servers == expected_servers

    # Validate app settings
    app_settings = config_dict["app_settings"]
    assert composer.model_owner == app_settings["model_owner"]
    assert composer.max_failures == app_settings["max_failures"]
    assert composer.cooldown_period == timedelta(minutes=app_settings["cooldown_period_minutes"])

    # Validate server health initialization
    expected_health = {server["url"]: {"healthy": True, "last_checked": None} for server in expected_servers}
    assert composer.server_health == expected_health

    # Validate logger settings
    log_levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET,
    }
    assert composer.logger.level == log_levels[app_settings["log_level"].lower()]

def test_load_secrets(mock_config_and_secrets):
    config_path, secrets_path, _, secrets_dict = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Normalize secrets_dict["groups"] to match the format of composer.group_tokens
    expected_group_tokens = {
        group: tokens for group_entry in secrets_dict["groups"] for group, tokens in group_entry.items()
    }

    # Validate group tokens
    assert composer.group_tokens == expected_group_tokens

    # Validate VLLM token
    assert composer.vllm_token == secrets_dict["vllm_token"]

def test_get_group_for_token(mock_config_and_secrets):
    config_path, secrets_path, _, secrets_dict = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    # Iterate through the list of group dictionaries
    for group_entry in secrets_dict["groups"]:
        # Each group_entry is a dictionary with one key (the group name)
        for group, tokens in group_entry.items():
            # Assert for each token in the group
            for token in tokens:
                assert composer.get_group_for_token(token) == group

    # Validate an invalid token
    assert composer.get_group_for_token("invalid_token") is None
