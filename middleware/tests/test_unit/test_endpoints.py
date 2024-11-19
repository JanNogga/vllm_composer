import pytest
import json
import httpx
from httpx import Request, Response
from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock, patch, MagicMock
from vllmComposer import vllmComposer


@pytest.mark.asyncio
async def test_get_server_load_success(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"
    mock_metrics_response = """
    # HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
    # TYPE vllm:num_preemptions_total counter
    vllm:num_preemptions_total{model_name="mymodelname"} 0.0
    # HELP vllm:prompt_tokens_total Number of prefill tokens processed.
    # TYPE vllm:prompt_tokens_total counter
    vllm:prompt_tokens_total{model_name="mymodelname"} 5.453083e+06
    # HELP vllm:generation_tokens_total Number of generation tokens processed.
    # TYPE vllm:generation_tokens_total counter
    vllm:generation_tokens_total{model_name="mymodelname"} 155722.0
    # HELP vllm:request_success_total Count of successfully processed requests.
    # TYPE vllm:request_success_total counter
    vllm:request_success_total{finished_reason="stop",model_name="mymodelname"} 6193.0
    vllm:request_success_total{finished_reason="abort",model_name="mymodelname"} 28.0
    vllm:request_success_total{finished_reason="length",model_name="mymodelname"} 5.0
    # HELP vllm:lora_requests_info Running stats on lora requests.
    # TYPE vllm:lora_requests_info gauge
    vllm:lora_requests_info{max_lora="0",running_lora_adapters="",waiting_lora_adapters=""} 1.0
    # HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
    # TYPE vllm:cache_config_info gauge
    vllm:cache_config_info{block_size="16",cache_dtype="fp8_e5m2",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",is_attention_free="False",num_cpu_blocks="2048",num_gpu_blocks="11560",num_gpu_blocks_override="None",sliding_window="None",swap_space_bytes="4294967296"} 1.0
    # HELP vllm:num_requests_running Number of requests currently running on GPU.
    # TYPE vllm:num_requests_running gauge
    vllm:num_requests_running{model_name="mymodelname"} 3.0
    # HELP vllm:num_requests_swapped Number of requests swapped to CPU.
    # TYPE vllm:num_requests_swapped gauge
    vllm:num_requests_swapped{model_name="mymodelname"} 0.0
    # HELP vllm:num_requests_waiting Number of requests waiting to be processed.
    # TYPE vllm:num_requests_waiting gauge
    vllm:num_requests_waiting{model_name="mymodelname"} 2.0
    # HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
    # TYPE vllm:gpu_cache_usage_perc gauge
    vllm:gpu_cache_usage_perc{model_name="mymodelname"} 0.0
    # HELP vllm:cpu_cache_usage_perc CPU KV-cache usage. 1 means 100 percent usage.
    # TYPE vllm:cpu_cache_usage_perc gauge
    vllm:cpu_cache_usage_perc{model_name="mymodelname"} 0.0
    # HELP vllm:cpu_prefix_cache_hit_rate CPU prefix cache block hit rate.
    # TYPE vllm:cpu_prefix_cache_hit_rate gauge
    vllm:cpu_prefix_cache_hit_rate{model_name="mymodelname"} -1.0
    # HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache block hit rate.
    # TYPE vllm:gpu_prefix_cache_hit_rate gauge
    vllm:gpu_prefix_cache_hit_rate{model_name="mymodelname"} -1.0
    # HELP vllm:avg_prompt_throughput_toks_per_s Average prefill throughput in tokens/s.
    # TYPE vllm:avg_prompt_throughput_toks_per_s gauge
    vllm:avg_prompt_throughput_toks_per_s{model_name="mymodelname"} 0.0
    # HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput in tokens/s.
    # TYPE vllm:avg_generation_throughput_toks_per_s gauge
    vllm:avg_generation_throughput_toks_per_s{model_name="mymodelname"} 0.0
    # HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
    # TYPE vllm:time_to_first_token_seconds histogram
    vllm:time_to_first_token_seconds_sum{model_name="mymodelname"} 5686.670262336731
    vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="mymodelname"} 0.0
    vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="mymodelname"} 0.0
    vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="mymodelname"} 0.0
    vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="mymodelname"} 0.0
    vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="mymodelname"} 19.0
    vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="mymodelname"} 83.0
    vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="mymodelname"} 133.0
    vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="mymodelname"} 247.0
    vllm:time_to_first_token_seconds_bucket{le="0.25",model_name="mymodelname"} 1006.0
    vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="mymodelname"} 3566.0
    vllm:time_to_first_token_seconds_bucket{le="0.75",model_name="mymodelname"} 6812.0
    vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="mymodelname"} 7511.0
    vllm:time_to_first_token_seconds_bucket{le="2.5",model_name="mymodelname"} 8297.0
    vllm:time_to_first_token_seconds_bucket{le="5.0",model_name="mymodelname"} 8432.0
    vllm:time_to_first_token_seconds_bucket{le="7.5",model_name="mymodelname"} 8482.0
    vllm:time_to_first_token_seconds_bucket{le="10.0",model_name="mymodelname"} 8494.0
    vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="mymodelname"} 8499.0
    vllm:time_to_first_token_seconds_count{model_name="mymodelname"} 8499.0
    # HELP vllm:time_per_output_token_seconds Histogram of time per output token in seconds.
    # TYPE vllm:time_per_output_token_seconds histogram
    vllm:time_per_output_token_seconds_sum{model_name="mymodelname"} 6772.65311050415
    vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="mymodelname"} 36.0
    vllm:time_per_output_token_seconds_bucket{le="0.025",model_name="mymodelname"} 99.0
    vllm:time_per_output_token_seconds_bucket{le="0.05",model_name="mymodelname"} 138458.0
    vllm:time_per_output_token_seconds_bucket{le="0.075",model_name="mymodelname"} 139802.0
    vllm:time_per_output_token_seconds_bucket{le="0.1",model_name="mymodelname"} 140057.0
    vllm:time_per_output_token_seconds_bucket{le="0.15",model_name="mymodelname"} 140759.0
    vllm:time_per_output_token_seconds_bucket{le="0.2",model_name="mymodelname"} 141246.0
    vllm:time_per_output_token_seconds_bucket{le="0.3",model_name="mymodelname"} 142037.0
    vllm:time_per_output_token_seconds_bucket{le="0.4",model_name="mymodelname"} 147174.0
    vllm:time_per_output_token_seconds_bucket{le="0.5",model_name="mymodelname"} 147219.0
    vllm:time_per_output_token_seconds_bucket{le="0.75",model_name="mymodelname"} 147223.0
    vllm:time_per_output_token_seconds_bucket{le="1.0",model_name="mymodelname"} 147223.0
    vllm:time_per_output_token_seconds_bucket{le="2.5",model_name="mymodelname"} 147223.0
    vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name="mymodelname"} 147223.0
    vllm:time_per_output_token_seconds_count{model_name="mymodelname"} 147223.0
    # HELP vllm:e2e_request_latency_seconds Histogram of end to end request latency in seconds.
    # TYPE vllm:e2e_request_latency_seconds histogram
    vllm:e2e_request_latency_seconds_sum{model_name="mymodelname"} 8560.867156982422
    vllm:e2e_request_latency_seconds_bucket{le="1.0",model_name="mymodelname"} 4406.0
    vllm:e2e_request_latency_seconds_bucket{le="2.5",model_name="mymodelname"} 5855.0
    vllm:e2e_request_latency_seconds_bucket{le="5.0",model_name="mymodelname"} 6120.0
    vllm:e2e_request_latency_seconds_bucket{le="10.0",model_name="mymodelname"} 6165.0
    vllm:e2e_request_latency_seconds_bucket{le="15.0",model_name="mymodelname"} 6174.0
    vllm:e2e_request_latency_seconds_bucket{le="20.0",model_name="mymodelname"} 6181.0
    vllm:e2e_request_latency_seconds_bucket{le="30.0",model_name="mymodelname"} 6189.0
    vllm:e2e_request_latency_seconds_bucket{le="40.0",model_name="mymodelname"} 6205.0
    vllm:e2e_request_latency_seconds_bucket{le="50.0",model_name="mymodelname"} 6211.0
    vllm:e2e_request_latency_seconds_bucket{le="60.0",model_name="mymodelname"} 6213.0
    vllm:e2e_request_latency_seconds_bucket{le="+Inf",model_name="mymodelname"} 6226.0
    vllm:e2e_request_latency_seconds_count{model_name="mymodelname"} 6226.0
    # HELP vllm:request_prompt_tokens Number of prefill tokens processed.
    # TYPE vllm:request_prompt_tokens histogram
    vllm:request_prompt_tokens_sum{model_name="mymodelname"} 3.928691e+06
    vllm:request_prompt_tokens_bucket{le="1.0",model_name="mymodelname"} 0.0
    vllm:request_prompt_tokens_bucket{le="2.0",model_name="mymodelname"} 0.0
    vllm:request_prompt_tokens_bucket{le="5.0",model_name="mymodelname"} 0.0
    vllm:request_prompt_tokens_bucket{le="10.0",model_name="mymodelname"} 1.0
    vllm:request_prompt_tokens_bucket{le="20.0",model_name="mymodelname"} 33.0
    vllm:request_prompt_tokens_bucket{le="50.0",model_name="mymodelname"} 90.0
    vllm:request_prompt_tokens_bucket{le="100.0",model_name="mymodelname"} 183.0
    vllm:request_prompt_tokens_bucket{le="200.0",model_name="mymodelname"} 520.0
    vllm:request_prompt_tokens_bucket{le="500.0",model_name="mymodelname"} 1345.0
    vllm:request_prompt_tokens_bucket{le="1000.0",model_name="mymodelname"} 6186.0
    vllm:request_prompt_tokens_bucket{le="2000.0",model_name="mymodelname"} 6197.0
    vllm:request_prompt_tokens_bucket{le="5000.0",model_name="mymodelname"} 6206.0
    vllm:request_prompt_tokens_bucket{le="10000.0",model_name="mymodelname"} 6221.0
    vllm:request_prompt_tokens_bucket{le="20000.0",model_name="mymodelname"} 6226.0
    vllm:request_prompt_tokens_bucket{le="50000.0",model_name="mymodelname"} 6226.0
    vllm:request_prompt_tokens_bucket{le="100000.0",model_name="mymodelname"} 6226.0
    vllm:request_prompt_tokens_bucket{le="+Inf",model_name="mymodelname"} 6226.0
    vllm:request_prompt_tokens_count{model_name="mymodelname"} 6226.0
    # HELP vllm:request_generation_tokens Number of generation tokens processed.
    # TYPE vllm:request_generation_tokens histogram
    vllm:request_generation_tokens_sum{model_name="mymodelname"} 124321.0
    vllm:request_generation_tokens_bucket{le="1.0",model_name="mymodelname"} 1466.0
    vllm:request_generation_tokens_bucket{le="2.0",model_name="mymodelname"} 2103.0
    vllm:request_generation_tokens_bucket{le="5.0",model_name="mymodelname"} 3701.0
    vllm:request_generation_tokens_bucket{le="10.0",model_name="mymodelname"} 4945.0
    vllm:request_generation_tokens_bucket{le="20.0",model_name="mymodelname"} 5676.0
    vllm:request_generation_tokens_bucket{le="50.0",model_name="mymodelname"} 6047.0
    vllm:request_generation_tokens_bucket{le="100.0",model_name="mymodelname"} 6129.0
    vllm:request_generation_tokens_bucket{le="200.0",model_name="mymodelname"} 6164.0
    vllm:request_generation_tokens_bucket{le="500.0",model_name="mymodelname"} 6181.0
    vllm:request_generation_tokens_bucket{le="1000.0",model_name="mymodelname"} 6197.0
    vllm:request_generation_tokens_bucket{le="2000.0",model_name="mymodelname"} 6215.0
    vllm:request_generation_tokens_bucket{le="5000.0",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_bucket{le="10000.0",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_bucket{le="20000.0",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_bucket{le="50000.0",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_bucket{le="100000.0",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_bucket{le="+Inf",model_name="mymodelname"} 6226.0
    vllm:request_generation_tokens_count{model_name="mymodelname"} 6226.0
    # HELP vllm:request_params_n Histogram of the n request parameter.
    # TYPE vllm:request_params_n histogram
    vllm:request_params_n_sum{model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="1.0",model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="2.0",model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="5.0",model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="10.0",model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="20.0",model_name="mymodelname"} 6226.0
    vllm:request_params_n_bucket{le="+Inf",model_name="mymodelname"} 6226.0
    vllm:request_params_n_count{model_name="mymodelname"} 6226.0
    """

    # Create a proper Response object with the associated Request
    request = Request(method="GET", url=f"{server_url}/metrics")
    mock_response = Response(status_code=200, request=request, text=mock_metrics_response)

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_response)):
        load = await composer.get_server_load(server_url)
        assert load == 5, f"The total load should be the sum of running and waiting requests. Got {load}."

@pytest.mark.asyncio
async def test_get_server_load_failure(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.HTTPStatusError("Error", request=None, response=None))):
        load = await composer.get_server_load(server_url)
        assert load is None, "Failed requests should return None."

@pytest.mark.asyncio
async def test_get_server_load_unhealthy_server(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    server_url = "http://server1:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.server_health[server_url] = {"healthy": False, "last_checked": None}  # Mark server as unhealthy

    server_load = await composer.get_server_load(server_url)

    # Validate that the load is None
    assert server_load is None

@pytest.mark.asyncio
async def test_get_server_load_cache_hit(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    server_url = "http://server1:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.metrics_cache[server_url] = 42.0  # Add a mock value to the cache

    server_load = await composer.get_server_load(server_url)

    # Validate that the cached value is returned
    assert server_load == 42.0

@pytest.mark.asyncio
async def test_get_model_on_server_success(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"
    mock_models_response = {
        "data": [
            {"id": "mygroup/my_modelname", "object": "model", "created": 123456789, "owner": "me", "some_other_field": "value"},
        ]
    }

    # Mock response with a proper `json` method
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.json = MagicMock(return_value=mock_models_response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_response)):
        model = await composer.get_model_on_server(server_url)
        assert model['id'] == "mygroup/my_modelname", "The returned model ID should match the mock response."
        assert model['created'] == 123456789, "The returned model should have the correct 'created' field."

@pytest.mark.asyncio
async def test_get_model_on_server_failure(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"

    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.HTTPStatusError("Error", request=None, response=None))):
        model = await composer.get_model_on_server(server_url)
        assert model is None, "Failed requests should return None."

@pytest.mark.asyncio
async def test_get_model_on_server_unhealthy_server(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    server_url = "http://server1:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.server_health[server_url] = {"healthy": False, "last_checked": None}  # Mark server as unhealthy

    model = await composer.get_model_on_server(server_url)

    # Validate that the model is None
    assert model is None


@pytest.mark.asyncio
async def test_get_model_on_server_cache_hit(mock_config_and_secrets):
    config_path, secrets_path, _, _ = mock_config_and_secrets
    composer = vllmComposer(config_path=config_path, secrets_path=secrets_path)

    server_url = "http://server1:8000"
    composer.servers = [{"url": server_url, "allowed_groups": ["group1"]}]
    composer.model_cache[server_url] = {"id": "mock_model"}  # Add a mock value to the cache

    model = await composer.get_model_on_server(server_url)

    # Validate that the cached value is returned
    assert model == {"id": "mock_model"}


@pytest.mark.asyncio
async def test_get_server_load_timeout(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"

    # Simulate a timeout exception
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))):
        load = await composer.get_server_load(server_url)
        assert load is None, "Timeout should result in a None return value."

@pytest.mark.asyncio
async def test_get_model_on_server_timeout(setup_composer):
    composer = setup_composer
    server_url = "http://mockserver:8080"

    # Simulate a timeout exception
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))):
        model = await composer.get_model_on_server(server_url)
        assert model is None, "Timeout should result in a None return value."

@pytest.mark.asyncio
async def test_handle_models_request(mock_config_and_secrets):
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
        "http://server3:8000": {"healthy": True, "last_checked": None},
    }

    # Mock get_model_on_server
    async def mock_get_model_on_server(server_url):
        models_data = {
            "http://server1:8000": [{"id": "model123", "created": 1650000000}],
            "http://server2:8000": [{"id": "model456", "created": 1650001000}],
            "http://server3:8000": [
                {"id": "model123", "created": 1650000500},
                {"id": "model789", "created": 1650002000},
            ],
        }
        return {"data": models_data.get(server_url, [])}

    composer.get_model_on_server = AsyncMock(side_effect=mock_get_model_on_server)

    # Call the handle_models_request method
    user_group = "group1"
    response = await composer.handle_models_request(user_group)

    # Decode the JSON response body
    assert isinstance(response, JSONResponse)
    response_data = json.loads(response.body.decode("utf-8"))

    # Validate the response structure and data
    assert response_data["object"] == "list"
    assert any(model["id"] == "model123" for model in response_data["data"])  # model123 should be included
    assert not any(model["id"] == "model456" for model in response_data["data"])  # model456 is not in group1
    assert any(model["id"] == "model789" for model in response_data["data"])  # model789 is compatible

    # Validate that models were fetched only from compatible servers
    composer.get_model_on_server.assert_any_await("http://server1:8000")
    composer.get_model_on_server.assert_any_await("http://server3:8000")

    # Ensure http://server2:8000 was not called
    called_urls = [call.args[0] for call in composer.get_model_on_server.call_args_list]
    assert "http://server2:8000" not in called_urls