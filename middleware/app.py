import re
import yaml
import httpx
import asyncio
import logging
from typing import Optional
from cachetools import TTLCache
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, HTTPException
from fastapi_utils.tasks import repeat_every
from fastapi.responses import JSONResponse, StreamingResponse


class vllmComposer:
    def __init__(self, config_path: str, secrets_path: str):
        self.servers = []
        self.model_owner = 'unknown'
        self.group_tokens = {}
        self.vllm_token = None

        # Caches
        self.metrics_cache = TTLCache(maxsize=100, ttl=10)
        self.model_cache = TTLCache(maxsize=100, ttl=20)

        # Server health tracking
        self.server_health = {}
        self.failure_counts = defaultdict(int)
        self.circuit_breaker_timeout = {}
        self.max_failures = 3
        self.cooldown_period = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler("app.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

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
                {"url": f"{hostname if hostname.startswith(('http://', 'https://')) else f'http://{hostname}'}:{port}", "allowed_groups": allowed_groups}
                for port in ports
            ])
        app_settings = config.get("app_settings", {})
        self.model_owner = app_settings.get("model_owner", "unknown")
        # Load circuit breaker settings from config
        self.max_failures = app_settings.get("max_failures", 3)
        cooldown_minutes = app_settings.get("cooldown_period_minutes", 5)
        self.cooldown_period = timedelta(minutes=cooldown_minutes)

        # Configure logger
        log_level = app_settings.get("log_level", "INFO").upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Initialize health for each server
        self.server_health = {server["url"]: {"healthy": True, "last_checked": None} for server in self.servers}
        self.logger.info("Configuration loaded successfully.")

    def load_secrets(self, secrets_path: str):
        with open(secrets_path, "r") as file:
            secrets = yaml.safe_load(file)
        
        # Parse group tokens
        self.group_tokens = secrets.get("groups", {})
        self.vllm_token = secrets.get("vllm_token")
        self.logger.info("Secrets loaded successfully.")

    async def check_circuit_breaker(self, server_url: str) -> bool:
        now = datetime.utcnow()

        # Check if server is in cooldown
        if server_url in self.circuit_breaker_timeout and self.circuit_breaker_timeout[server_url] > now:
            return False

        # Reset failure count if cooldown period has passed
        if server_url in self.circuit_breaker_timeout:
            del self.circuit_breaker_timeout[server_url]
            self.failure_counts[server_url] = 0

        return True
    
    async def handle_server_failure(self, server_url: str):
        """Handle a server failure and open the circuit if necessary."""
        self.failure_counts[server_url] += 1

        if self.failure_counts[server_url] >= self.max_failures:
            self.circuit_breaker_timeout[server_url] = datetime.utcnow() + self.cooldown_period
            self.logger.warning(f"Server {server_url} disabled due to repeated failures.")
    
        # Mark the server as unhealthy
        await self.update_server_health(server_url, is_healthy=False)


    def get_group_for_token(self, token: str) -> Optional[str]:
        """Get the group associated with a token."""
        for group, tokens in self.group_tokens.items():
            if token in tokens:
                return group
        return None

    async def get_server_load(self, server_url: str) -> float | None:
        if not await self.is_server_healthy(server_url):
            return None
        # Check cache first
        if server_url in self.metrics_cache:
            return self.metrics_cache[server_url]

        url = f"{server_url}/metrics"
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
                self.metrics_cache[server_url] = total_load  # Update cache
                await self.update_server_health(server_url, is_healthy=True)
                self.logger.info(f"Metrics fetched for {server_url}: Load = {total_load}")
                return total_load
            except Exception as exc:
                await self.handle_server_failure(server_url)
                self.logger.error(f"Error fetching metrics from {server_url}: {exc}")
                return None

    async def get_model_on_server(self, server_url: str) -> str | None:
        if not await self.is_server_healthy(server_url):
            return None
        # Check cache first
        if server_url in self.model_cache:
            return self.model_cache[server_url]

        url = f"{server_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.vllm_token}"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # Extract the model ID
                models = data.get("data", [])
                if models:
                    model_id = models[0]
                    self.model_cache[server_url] = model_id  # Update cache
                    await self.update_server_health(server_url, is_healthy=True)
                    self.logger.info(f"Model fetched for {server_url}: Model ID = {model_id}")
                    return model_id
            except Exception as exc:
                await self.handle_server_failure(server_url)
                self.logger.error(f"Error fetching model from {server_url}: {exc}")
                return None

    async def update_server_health(self, server_url: str, is_healthy: bool):
        """Update the health status of a server."""
        self.server_health[server_url] = {
            "healthy": is_healthy,
            "last_checked": datetime.utcnow()
        }
        if is_healthy:
            self.failure_counts[server_url] = 0

    async def is_server_healthy(self, server_url: str) -> bool:
        """Check if a server is healthy."""
        if not await self.check_circuit_breaker(server_url):
            return False
        health_info = self.server_health.get(server_url, {"healthy": True})
        return health_info["healthy"]

    @repeat_every(seconds=20)
    async def refresh_models(self):
        """Periodically refresh model caches."""
        tasks = []
        for server in self.servers:
            tasks.append(self.get_model_on_server(server["url"]))
        await asyncio.gather(*tasks)

    @repeat_every(seconds=10)
    async def refresh_metrics(self):
        """Periodically refresh metrics and model caches."""
        tasks = []
        for server in self.servers:
            tasks.append(self.get_server_load(server["url"]))
        await asyncio.gather(*tasks)

    async def get_compatible_servers(self, target_model_id: str, user_group: str) -> list[str]:
        compatible_servers = []
        for server in self.servers:
            server_url = server["url"]
            allowed_groups = server["allowed_groups"]

            # Check if the server is available for the user's group
            if user_group not in allowed_groups or not await self.is_server_healthy(server_url):
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
        models_data = []

        for server in self.servers:
            server_url = server["url"]
            allowed_groups = server["allowed_groups"]

            if user_group not in allowed_groups or not await self.is_server_healthy(server_url):
                continue

            server_models = await self.get_model_on_server(server_url)
            if server_models:
                models_data.extend(server_models.get("data", []))

        formatted_models = {}
        for model in models_data:
            model_id = model['id']
            created_timestamp = model['created']

            if model_id in formatted_models:
                formatted_models[model_id]['created'] = min(
                    formatted_models[model_id]['created'], created_timestamp
                )
            else:
                formatted_models[model_id] = {
                    'id': model_id,
                    'object': 'model',
                    'created': created_timestamp,
                    'owned_by': self.model_owner,
                }

        models_list = list(formatted_models.values())
        return JSONResponse(content={"object": "list", "data": models_list})


# FastAPI app instantiation
app = FastAPI()
composer = vllmComposer("config.yml", "secrets.yml")

@app.get("/health")
async def health_check():
    """Return health and cache status for all servers."""
    health_data = []
    for server in composer.servers:
        server_url = server["url"]
        health_data.append({
            "url": server_url,
            "healthy": composer.server_health.get(server_url, {}).get("healthy", True),
            "metrics_cached": composer.metrics_cache.get(server_url),
            "model_cached": composer.model_cache.get(server_url)
        })
    return JSONResponse(content={"servers": health_data})


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
    composer.logger.info(f"Received request for model '{target_model_name}' from group '{user_group}'.")

    # Step 4: Find compatible servers
    compatible_servers = await composer.get_compatible_servers(target_model_name, user_group)
    if not compatible_servers:
        raise HTTPException(status_code=503, detail="Service Unavailable: No compatible servers found")
    
    # Step 5: Find the least utilized server
    least_loaded_server = await composer.get_least_utilized_server(compatible_servers)
    if not least_loaded_server:
        raise HTTPException(status_code=503, detail="Service Unavailable: No available servers with sufficient capacity")
    composer.logger.info(f"Least loaded server for model '{target_model_name}': {least_loaded_server}. Forwarding request...")
    
    # Step 6: Replace user's token with self.vllm_token and forward the request
    url = f"{least_loaded_server}/v1/{path}"
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
                composer.logger.error(f"Error fetching metrics from {server_url}: {exc}")
                continue

    return JSONResponse(content=metrics_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)