import re
import time
import yaml
import httpx
import asyncio
import logging
from typing import Optional
from cachetools import TTLCache
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse

class RateLimitFilter(logging.Filter):
    def __init__(self, min_interval):
        super().__init__()
        self.min_interval = min_interval  # Minimum interval between logs (in seconds)
        self.last_log_time = {}

    def filter(self, record):
        current_time = time.time()
        # Create a unique key for each log message based on name, level, and message
        record_key = (record.name, record.levelno, record.getMessage())
        last_time = self.last_log_time.get(record_key, 0)
        if current_time - last_time >= self.min_interval:
            self.last_log_time[record_key] = current_time
            return True  # Log the message
        else:
            return False  # Skip the message

class vllmComposer:
    def __init__(self, config_path: str, secrets_path: str):
        self.servers = []
        self.model_owner = 'unknown'
        self.group_tokens = {}
        self.vllm_token = None

        # Caches
        self.metrics_cache = TTLCache(maxsize=100, ttl=0.5)
        self.model_cache = TTLCache(maxsize=100, ttl=5)

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
        rate_limit_filter = RateLimitFilter(min_interval=1)
        self.logger.addFilter(rate_limit_filter)

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
        request_timeout = app_settings.get("request_timeout", 2.0)
        self.request_timeout = request_timeout

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
        groups = secrets.get("groups", [])
        self.group_tokens = {group: tokens for group_entry in groups for group, tokens in group_entry.items()}
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
            self.logger.warning(f"Server {server_url} is not healthy. Not fetching metrics.")
            return None
        # Check cache first
        if server_url in self.metrics_cache:
            return self.metrics_cache[server_url]

        url = f"{server_url}/metrics"
        async with httpx.AsyncClient() as client:
            try:
                self.logger.info(f"Fetching metrics from {server_url}")
                response = await asyncio.wait_for(client.get(url), self.request_timeout)
                response.raise_for_status()
                metrics_data = response.text

                # Process metrics data
                current_requests, pending_requests = 0, 0
                self.logger.debug("Processing metrics data:")
                for line in metrics_data.splitlines():
                    line = line.strip() 
                    self.logger.debug(f"Line: {line}")
                    if line.startswith("vllm:num_requests_running"):
                        match = re.search(r'(\d+(\.\d+)?)$', line)
                        if match:
                            current_requests += float(match.group(1))
                            self.logger.debug(f"Found running requests: {match.group(1)}")
                    elif line.startswith("vllm:num_requests_waiting"):
                        match = re.search(r'(\d+(\.\d+)?)$', line)
                        if match:
                            pending_requests += float(match.group(1))
                            self.logger.debug(f"Found waiting requests: {match.group(1)}")

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
            self.logger.warning(f"Server {server_url} is not healthy. Not fetching model.")
            return None
        # Check cache first
        if server_url in self.model_cache:
            return self.model_cache[server_url]

        url = f"{server_url}/v1/models"
        headers = {"Authorization": f"Bearer {self.vllm_token}"}

        async with httpx.AsyncClient() as client:
            try:
                self.logger.info(f"Fetching model from {server_url}")
                response = await asyncio.wait_for(client.get(url, headers=headers), self.request_timeout)
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
    
    async def refresh_models(self):
        """Periodically refresh model caches."""
        self.logger.info("Refreshing model caches.")
        tasks = []
        for server in self.servers:
            tasks.append(self.get_model_on_server(server["url"]))
        await asyncio.gather(*tasks)

    async def refresh_metrics(self):
        """Periodically refresh metrics and model caches."""
        self.logger.info("Refreshing metrics caches.")
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
