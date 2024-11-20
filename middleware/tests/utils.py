from pathlib import Path
import shutil
import yaml
import asyncio
import respx
from httpx import Response
from contextlib import asynccontextmanager

def load_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def create_mock_config_from_templates(tmp_path: Path):
    # Base directory for the project root
    base_dir = Path(__file__).parent
    config_template = base_dir / "test_config.yml"
    secrets_template = base_dir / "test_secrets.yml"

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

async def delayed_metrics_response(*args, **kwargs):
    """Simulates a delayed server response for metrics."""
    await asyncio.sleep(10)  # Simulate delay
    return Response(200, text="vllm:num_requests_running 1\nvllm:num_requests_waiting 0")

async def delayed_models_response(*args, **kwargs):
    """Simulates a delayed server response for models."""
    await asyncio.sleep(10)  # Simulate delay
    return Response(200, json={"data": [{"id": "unique-model-delayed", "created": 1234567890}]})

@asynccontextmanager
async def create_mock_servers(config_path):
    config = load_config(config_path)
    expected_data = []

    model_name, model_created = "shared-model", 1234567890
    with respx.mock(assert_all_called=False) as mock:
        for index, host in enumerate(config["vllm_hosts"]):
            hostname = "127.0.0.1"
            start_port = host["ports"]["start"]
            end_port = host["ports"]["end"]

            for port in range(start_port, end_port + 1):
                base_url = f"http://{hostname}:{port}"

                if index < 2:
                    mock.get(f"{base_url}/metrics").mock(
                        return_value=Response(200, text="vllm:num_requests_running 2\nvllm:num_requests_waiting 1")
                    )
                    mock.get(f"{base_url}/v1/models").mock(
                        return_value=Response(200, json={"data": [{"id": model_name, "created": model_created}]})
                    )
                    expected_data.append({
                        "url": base_url,
                        "healthy": True,
                        "metrics_cached": 3,  # 2 running + 1 waiting
                        "model_cached": {'created': model_created, 'id': model_name},
                        "raw_metrics": "vllm:num_requests_running 2\nvllm:num_requests_waiting 1"
                    })
                elif index == 2:
                    mock.get(f"{base_url}/metrics").mock(side_effect=delayed_metrics_response)
                    mock.get(f"{base_url}/v1/models").mock(side_effect=delayed_models_response)
                    expected_data.append({
                        "url": base_url,
                        "healthy": False,
                        "metrics_cached": None,
                        "model_cached": None,
                        "raw_metrics": "Error: Simulated delayed response"
                    })
                elif index == 3:
                    mock.get(f"{base_url}/metrics").mock(side_effect=ConnectionError("Host is unreachable"))
                    mock.get(f"{base_url}/v1/models").mock(side_effect=ConnectionError("Host is unreachable"))
                    expected_data.append({
                        "url": base_url,
                        "healthy": False,
                        "metrics_cached": None,
                        "model_cached": None,
                        "raw_metrics": "Error: Host is unreachable"
                    })
                else:
                    unique_model_name = f"unique-model-{port}"
                    mock.get(f"{base_url}/metrics").mock(
                        return_value=Response(200, text="vllm:num_requests_running 1\nvllm:num_requests_waiting 0")
                    )
                    mock.get(f"{base_url}/v1/models").mock(
                        return_value=Response(200, json={"data": [{"id": unique_model_name, "created": model_created}]})
                    )
                    expected_data.append({
                        "url": base_url,
                        "healthy": True,
                        "metrics_cached": 1,  # 1 running + 0 waiting
                        "model_cached": {'created': model_created, 'id': unique_model_name},
                        "raw_metrics": "vllm:num_requests_running 1\nvllm:num_requests_waiting 0"
                    })

        yield expected_data
