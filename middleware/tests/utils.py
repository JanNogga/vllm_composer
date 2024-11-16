from pathlib import Path
import shutil
import yaml

def create_mock_config_from_templates(tmp_path: Path):
    # Base directory for the project root
    base_dir = Path(__file__).parent.parent # Navigate to project root
    config_template = base_dir / "config.yml.template"
    secrets_template = base_dir / "secrets.yml.template"

    # Temporary mock file paths
    config_path = tmp_path / "config.yml"
    secrets_path = tmp_path / "secrets.yml"

    # Copy templates to temporary paths
    shutil.copy(config_template, config_path)
    shutil.copy(secrets_template, secrets_path)

    # Parse YAML into dictionaries
    with config_path.open("r") as config_file:
        config_dict = yaml.safe_load(config_file)

    with secrets_path.open("r") as secrets_file:
        secrets_dict = yaml.safe_load(secrets_file)

    return str(config_path), str(secrets_path), config_dict, secrets_dict