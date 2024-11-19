import pytest
from vllmComposer import vllmComposer
from tests.utils import create_mock_config_from_templates

@pytest.fixture
def mock_config_and_secrets(tmp_path):
    return create_mock_config_from_templates(tmp_path)

@pytest.fixture
def setup_composer(tmp_path):
    config_path, secrets_path, _, _ = create_mock_config_from_templates(tmp_path)
    return vllmComposer(config_path, secrets_path)