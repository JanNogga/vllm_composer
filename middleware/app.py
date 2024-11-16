import re
import yaml
import httpx
from collections import defaultdict
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse


class vllmComposer:
    def __init__(self, config_path: str, secrets_path: str):
        self.servers = []
        self.model_owner = 'unknown'
        self.group_tokens = {}
        self.vllm_token = None
        self.load_config(config_path)
        self.load_secrets(secrets_path)

    def load_config(self, config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Parse vllm_hosts with allowed groups
        for host in config.get("vllm_hosts", []):
            hostname = host["hostname"]
            ports = range(host["ports"]["start"], host["ports"]["end"] + 1)
            allowed_groups = host["allowed_groups"]
            self.servers.extend([
                {"url": f"http://{hostname}:{port}", "allowed_groups": allowed_groups}
                for port in ports
            ])
        app_settings = config.get("app_settings", {})
        self.model_owner = app_settings.get("model_owner", "unknown")

    def load_secrets(self, secrets_path: str):
        with open(secrets_path, "r") as file:
            secrets = yaml.safe_load(file)
        
        # Parse group tokens
        self.group_tokens = secrets.get("groups", {})
        self.vllm_token = secrets.get("vllm_token")

    def get_group_for_token(self, token: str) -> Optional[str]:
        """Get the group associated with a token."""
        for group, tokens in self.group_tokens.items():
            if token in tokens:
                return group
        return None
    
    async def get_model_on_server(self, server_url: str) -> str | None:
        url = f"http://{server_url}/v1/models"
        headers = {
            "Authorization": f"Bearer {self.vllm_token}"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # Extract the model ID
                models = data.get("data", [])
                if models:
                    # Assuming the first model is the one to be determined
                    return models[0]
            except httpx.RequestError as exc:
                print(f"Request error while fetching model from {server_url}: {exc}")
            except KeyError:
                print(f"Unexpected response format from {server_url}: {response.text}")

        return None
    
    async def get_server_load(self, server_url: str) -> float | None:
        url = f"http://{server_url}/metrics"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                metrics_data = response.text

                # Process metrics data
                current_requests, pending_requests = 0, 0
                for line in metrics_data.splitlines():
                    if line.startswith("vllm:num_requests_running"):
                        current_requests += float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        pending_requests += float(line.split()[-1])

                total_load = current_requests + pending_requests
                return total_load
            except httpx.RequestError as exc:
                print(f"Request error while fetching load from {server_url}: {exc}")
            except ValueError:
                print(f"Unexpected metrics format from {server_url}: {response.text}")

        return None
    
    async def get_compatible_servers(self, target_model_id: str, user_group: str) -> list[str]:
        compatible_servers = []
        for server in self.servers:
            server_url = server["url"]
            allowed_groups = server["allowed_groups"]

            # Check if the server is available for the user's group
            if user_group not in allowed_groups:
                continue

            # Check if the server provides the target model
            model_info = await self.get_model_on_server(server_url)
            if model_info and model_info.get("id") == target_model_id:
                compatible_servers.append(server_url)

        return compatible_servers

    async def get_least_utilized_server(self, compatible_servers: list[str]) -> str | None:
        min_load = float('inf')
        least_loaded_server = None

        for server_url in compatible_servers:
            server_load = await self.get_server_load(server_url)
            if server_load == 0:
                return server_url
            if server_load is not None and server_load < min_load:
                min_load = server_load
                least_loaded_server = server_url

        return least_loaded_server
    
    async def handle_models_request(self, user_group: str) -> JSONResponse:
        """
        Handle /v1/models request by aggregating models that the user's group has access to.

        Args:
            user_group (str): The group of the user making the request.

        Returns:
            JSONResponse: A response containing the list of models accessible to the user's group,
                        formatted as specified.
        """
        models_data = []

        for server in self.servers:
            server_url = server["url"]
            allowed_groups = server["allowed_groups"]

            # Skip servers not accessible to the user's group
            if user_group not in allowed_groups:
                continue

            # Use get_model_on_server to fetch model data
            server_models = await self.get_model_on_server(server_url)
            if server_models:
                models_data.extend(server_models.get("data", []))

        # Format the response as specified
        formatted_models = {}
        for model in models_data:
            model_id = model['id']
            created_timestamp = model['created']

            if model_id in formatted_models:
                # Update the 'created' timestamp to the oldest one
                formatted_models[model_id]['created'] = min(
                    formatted_models[model_id]['created'], created_timestamp
                )
            else:
                # Add new entry to formatted models
                formatted_models[model_id] = {
                    'id': model_id,
                    'object': 'model',
                    'created': created_timestamp,
                    'owned_by': self.model_owner,
                }

        # Convert formatted models to a list
        models_list = list(formatted_models.values())

        return JSONResponse(content={"object": "list", "data": models_list})


# Instantiate the app and backend configuration
app = FastAPI()
composer = vllmComposer("config.yml")

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(path: str, request: Request):
    # Step 1: Extract the user's token from headers
    user_token = request.headers.get("Authorization")
    if not user_token or not user_token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token")

    user_token = user_token[len("Bearer "):]

    # Step 2: Determine the user's group
    user_group = composer.get_group_for_token(user_token)
    if not user_group:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token or unauthorized group")
    
    # Restrict to specific routes
    if path not in ["chat/completions", "completions", "models", "embeddings"]:
        raise HTTPException(status_code=404, detail=f"Not Found: The route '{path}' is not supported.")
    
    # Handle /v1/models differently
    if path == "models":
        return await composer.handle_models_request(user_group)
    
    # Step 3: Extract the target model name from the JSON payload
    try:
        payload = await request.json()
        target_model_name = payload.get("model")
        if not target_model_name:
            raise HTTPException(status_code=400, detail="Bad Request: Missing 'model' in payload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad Request: Invalid JSON payload ({str(exc)})")

    # Step 4: Find compatible servers
    compatible_servers = await composer.get_compatible_servers(target_model_name, user_group)
    if not compatible_servers:
        raise HTTPException(status_code=503, detail="Service Unavailable: No compatible servers found")
    
    # Step 5: Find the least utilized server
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    if not least_loaded_server:
        raise HTTPException(status_code=503, detail="Service Unavailable: No available servers with sufficient capacity")
    
    # Step 6: Replace user's token with self.vllm_token and forward the request
    url = f"http://{least_loaded_server}/v1/{path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() != "authorization"}
    headers["Authorization"] = f"Bearer {composer.vllm_token}"

    stream_mode = payload.get("stream", False)

    async with httpx.AsyncClient() as client:
        try:
            if stream_mode:
                # Handle streaming response
                response = await client.stream(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )

                async def stream_response():
                    async for chunk in response.aiter_bytes():
                        yield chunk

                return StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            else:
                # Handle non-streaming response
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    json=payload if request.method in ["POST", "PUT"] else None,
                    params=request.query_params
                )
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error communicating with backend server: {exc}")


def parse_metrics(raw_metrics: str, aggregated_metrics: dict):
    """
    Parse raw Prometheus-style metrics and aggregate them.

    Args:
        raw_metrics (str): Raw metrics data from a server.
        aggregated_metrics (dict): Dictionary to store aggregated metrics.
    """
    metrics_pattern = re.compile(r'(\w+:\w+)\{(.*?)\}\s+([\d\.\-eE]+)')
    label_pattern = re.compile(r'(\w+)="(.*?)"')
    metrics = defaultdict(list)

    # Match Prometheus-style metrics with labels
    for match in metrics_pattern.finditer(raw_metrics):
        metric_name = match.group(1)  # Metric name (e.g., "vllm:lora_requests_info")
        labels_raw = match.group(2)  # Raw labels string (e.g., 'max_lora="0",running_lora_adapters=""')
        metric_value = float(match.group(3))  # Metric value

        # Parse the labels into a dictionary
        labels = {m.group(1): m.group(2) for m in label_pattern.finditer(labels_raw)}

        # Use a tuple of the metric name and sorted labels as the aggregation key
        aggregation_key = (metric_name, tuple(sorted(labels.items())))

        metrics[aggregation_key].append(metric_value)

    # Aggregate metrics (e.g., sum for counters, average for gauges)
    for (metric_name, labels), values in metrics.items():
        if "counter" in metric_name:  # For counters, take the sum
            aggregated_value = sum(values)
        else:  # For gauges, take the average
            aggregated_value = sum(values) / len(values) if values else 0

        # Store the aggregated metric in the result, including its labels
        if metric_name not in aggregated_metrics:
            aggregated_metrics[metric_name] = []
        
        aggregated_metrics[metric_name].append({
            "labels": dict(labels),
            "value": aggregated_value
        })

@app.get("/metrics")
async def get_aggregated_metrics():
    """
    Aggregate metrics from all known servers and return the results.
    """

    metrics_data = {}

    async with httpx.AsyncClient() as client:
        for server in composer.servers:
            server_url = f"{server['url']}/metrics"
            try:
                response = await client.get(server_url)
                response.raise_for_status()
                # Parse the metrics data
                raw_metrics = response.text
                parse_metrics(raw_metrics, metrics_data)
            except httpx.RequestError as exc:
                print(f"Error fetching metrics from {server_url}: {exc}")
                continue

    return JSONResponse(content=metrics_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)